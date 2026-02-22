"""
Market data tools for fetching financial information.

Uses Yahoo Finance (yfinance) for real market data: price history,
financial statements, and recent news headlines.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf

from src.tools.twitter_data import get_recent_tweets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price data
# ---------------------------------------------------------------------------

def get_price_history(ticker: str, days: int = 90) -> List[dict]:
    """Fetch recent daily price history for a ticker via Yahoo Finance.

    Returns a list of dicts with keys: date, open, high, low, close, volume.
    Returns an empty list if the ticker is invalid or data is unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        period = "6mo" if days <= 180 else "1y"
        df = stock.history(period=period)

        if df.empty:
            logger.warning(f"No price data available for {ticker}")
            return []

        # Trim to requested number of days
        df = df.tail(days)

        prices = []
        for date, row in df.iterrows():
            prices.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })

        return prices

    except Exception as e:
        logger.error(f"Error fetching price data for {ticker}: {e}")
        return []


# ---------------------------------------------------------------------------
# Fundamental data
# ---------------------------------------------------------------------------

def get_financial_statements(ticker: str) -> dict:
    """Fetch key financial metrics from Yahoo Finance.

    Returns a dict with revenue, net_income, eps, pe_ratio, etc.
    Missing fields are set to None so agents can see what data is unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Get income statement data
        income_stmt = stock.income_stmt
        revenue = None
        net_income = None
        revenue_growth = None

        if income_stmt is not None and not income_stmt.empty:
            # income_stmt columns are dates, most recent first
            if "Total Revenue" in income_stmt.index:
                rev_series = income_stmt.loc["Total Revenue"].dropna()
                if len(rev_series) >= 1:
                    revenue = float(rev_series.iloc[0])
                if len(rev_series) >= 2:
                    prev_rev = float(rev_series.iloc[1])
                    if prev_rev != 0:
                        revenue_growth = (revenue - prev_rev) / abs(prev_rev)

            if "Net Income" in income_stmt.index:
                ni_series = income_stmt.loc["Net Income"].dropna()
                if len(ni_series) >= 1:
                    net_income = float(ni_series.iloc[0])

        # Get balance sheet data
        balance_sheet = stock.balance_sheet
        total_debt = None
        total_equity = None
        total_assets = None
        current_ratio = None

        if balance_sheet is not None and not balance_sheet.empty:
            bs = balance_sheet.iloc[:, 0]  # most recent

            if "Total Debt" in balance_sheet.index:
                total_debt = float(bs.get("Total Debt", 0) or 0) or None
            if "Stockholders Equity" in balance_sheet.index:
                total_equity = float(bs.get("Stockholders Equity", 0) or 0) or None
            elif "Total Stockholders Equity" in balance_sheet.index:
                total_equity = float(bs.get("Total Stockholders Equity", 0) or 0) or None
            if "Total Assets" in balance_sheet.index:
                total_assets = float(bs.get("Total Assets", 0) or 0) or None
            if "Current Assets" in balance_sheet.index and "Current Liabilities" in balance_sheet.index:
                ca = float(bs.get("Current Assets", 0) or 0)
                cl = float(bs.get("Current Liabilities", 0) or 0)
                if cl > 0:
                    current_ratio = round(ca / cl, 2)

        # Compute ratios
        eps = info.get("trailingEps")
        pe_ratio = info.get("trailingPE")
        debt_to_equity = None
        if total_debt and total_equity and total_equity != 0:
            debt_to_equity = round(total_debt / total_equity, 2)

        roe = info.get("returnOnEquity")
        if roe is not None:
            roe = round(float(roe), 4)

        # Market cap and forward metrics
        market_cap = info.get("marketCap")
        forward_pe = info.get("forwardPE")
        dividend_yield = info.get("dividendYield")
        sector = info.get("sector")
        industry = info.get("industry")
        company_name = info.get("shortName") or info.get("longName")

        result = {
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "revenue": revenue,
            "net_income": net_income,
            "eps": eps,
            "pe_ratio": round(float(pe_ratio), 1) if pe_ratio else None,
            "forward_pe": round(float(forward_pe), 1) if forward_pe else None,
            "revenue_growth": round(float(revenue_growth), 4) if revenue_growth is not None else None,
            "debt_to_equity": debt_to_equity,
            "return_on_equity": roe,
            "current_ratio": current_ratio,
            "total_debt": total_debt,
            "total_equity": total_equity,
            "total_assets": total_assets,
            "market_cap": market_cap,
            "dividend_yield": round(float(dividend_yield), 4) if dividend_yield else None,
            "data_source": "Yahoo Finance",
        }

        # Check if we actually got meaningful data
        if revenue is None and net_income is None and eps is None and pe_ratio is None:
            result["data_quality"] = "limited"
            logger.warning(f"Limited financial data available for {ticker}")
        else:
            result["data_quality"] = "good"

        return result

    except Exception as e:
        logger.error(f"Error fetching financial data for {ticker}: {e}")
        return {
            "ticker": ticker,
            "data_quality": "unavailable",
            "error": str(e),
            "data_source": "Yahoo Finance (error)",
        }


# ---------------------------------------------------------------------------
# News data
# ---------------------------------------------------------------------------

def get_recent_news(ticker: str) -> List[dict]:
    """Fetch recent news headlines for a ticker via Yahoo Finance.

    Returns a list of dicts with keys: title, date, source, link.
    Returns an empty list if no news is available.
    """
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news or []

        headlines = []
        for item in news_items[:10]:  # limit to 10 most recent
            # yfinance 1.2+ uses nested content structure
            content = item.get("content", item)  # fallback for older format

            title = content.get("title", "") or item.get("title", "")
            if not title:
                continue

            # Publisher / source
            provider = content.get("provider", {})
            if isinstance(provider, dict):
                publisher = provider.get("displayName", "Unknown")
            else:
                publisher = item.get("publisher", "Unknown")

            # Link
            click_through = content.get("clickThroughUrl") or content.get("canonicalUrl")
            if isinstance(click_through, dict):
                link = click_through.get("url", "")
            else:
                link = item.get("link", "")

            # Date
            pub_date_str = content.get("pubDate") or content.get("displayTime")
            if pub_date_str and isinstance(pub_date_str, str):
                try:
                    # Parse ISO format: "2026-02-21T16:33:00Z"
                    dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    date_str = datetime.now().strftime("%Y-%m-%d")
            else:
                # Fallback: try unix timestamp (older yfinance)
                pub_ts = item.get("providerPublishTime")
                if pub_ts:
                    try:
                        date_str = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d")
                    except (ValueError, TypeError, OSError):
                        date_str = datetime.now().strftime("%Y-%m-%d")
                else:
                    date_str = datetime.now().strftime("%Y-%m-%d")

            headlines.append({
                "title": title,
                "date": date_str,
                "source": publisher,
                "link": link,
            })

        if not headlines:
            logger.info(f"No news available for {ticker}")

        return headlines

    except Exception as e:
        logger.error(f"Error fetching news for {ticker}: {e}")
        return []


# ---------------------------------------------------------------------------
# Convenience: fetch all data for a ticker
# ---------------------------------------------------------------------------

def get_all_market_data(tickers: List[str]) -> Dict[str, dict]:
    """Fetch prices, fundamentals, news, and social media data for each ticker.

    Returns: {ticker: {prices, fundamentals, news, tweets}}
    """
    market_data = {}
    for ticker in tickers:
        logger.info(f"Fetching market data for {ticker}...")
        market_data[ticker] = {
            "prices": get_price_history(ticker),
            "fundamentals": get_financial_statements(ticker),
            "news": get_recent_news(ticker),
            "tweets": get_recent_tweets(ticker),
        }
    return market_data
