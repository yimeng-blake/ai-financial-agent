"""
Market data tools for fetching financial information.

Uses Yahoo Finance (yfinance) for real market data: price history,
financial statements, and recent news headlines.
"""

from __future__ import annotations

import logging
import math
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
# Fundamental data — helpers
# ---------------------------------------------------------------------------

def _safe_stmt_lookup(stmt, row_names: list[str], col_idx: int = 0):
    """Try multiple row names in a financial statement, return first found value.

    Args:
        stmt: pandas DataFrame (income_stmt, balance_sheet, or cashflow)
        row_names: list of possible row labels to try (yfinance naming varies)
        col_idx: column index (0 = most recent period)

    Returns:
        float value or None
    """
    if stmt is None or stmt.empty:
        return None
    for name in row_names:
        if name in stmt.index:
            try:
                val = stmt.iloc[:, col_idx][name]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    return float(val)
            except (IndexError, KeyError, TypeError, ValueError):
                continue
    return None


def _safe_divide(numerator, denominator, round_to: int = 4):
    """Safely divide two numbers, returning None on failure."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    try:
        return round(float(numerator) / float(denominator), round_to)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _safe_round(val, digits: int = 4):
    """Round a value if it's numeric, else return None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, digits)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Fundamental data
# ---------------------------------------------------------------------------

def get_financial_statements(ticker: str) -> dict:
    """Fetch comprehensive financial metrics from Yahoo Finance.

    Returns a dict with ~60 fields covering:
    - Core financials (revenue, net income, EPS, growth)
    - Profitability margins (gross, operating, EBITDA, FCF, net)
    - Cash flow (FCF, operating CF, capex)
    - Income statement detail (gross profit, R&D, SGA, EBITDA)
    - Balance sheet detail (cash, net debt, inventory, TBV)
    - Valuation multiples (EV/Revenue, EV/EBITDA, P/B, PEG)
    - Efficiency metrics (R&D%, SGA%, inventory turnover, Rule of 40)
    - Per-share metrics (book value, FCF, tangible BV)

    Missing fields are set to None so agents can see what data is unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # ==================================================================
        # INCOME STATEMENT
        # ==================================================================
        income_stmt = stock.income_stmt
        revenue = None
        net_income = None
        revenue_growth = None

        if income_stmt is not None and not income_stmt.empty:
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

        # Additional income statement items
        gross_profit = _safe_stmt_lookup(income_stmt, ["Gross Profit"])
        operating_income = _safe_stmt_lookup(
            income_stmt, ["Operating Income", "EBIT"]
        )
        cost_of_revenue = _safe_stmt_lookup(
            income_stmt, ["Cost Of Revenue", "Cost Of Goods Sold"]
        )
        research_development = _safe_stmt_lookup(
            income_stmt,
            ["Research And Development", "Research Development",
             "Research & Development"],
        )
        selling_general_admin = _safe_stmt_lookup(
            income_stmt,
            ["Selling General And Administration",
             "Selling General Administrative",
             "Selling General And Admin"],
        )
        ebitda_stmt = _safe_stmt_lookup(
            income_stmt, ["EBITDA", "Normalized EBITDA"]
        )
        interest_expense = _safe_stmt_lookup(
            income_stmt, ["Interest Expense", "Net Interest Income"]
        )

        # ==================================================================
        # BALANCE SHEET
        # ==================================================================
        balance_sheet = stock.balance_sheet
        total_debt = None
        total_equity = None
        total_assets = None
        current_ratio = None
        current_assets = None
        current_liabilities = None

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

            # Current ratio
            current_assets = _safe_stmt_lookup(balance_sheet, ["Current Assets"])
            current_liabilities = _safe_stmt_lookup(
                balance_sheet, ["Current Liabilities"]
            )
            if current_assets and current_liabilities and current_liabilities > 0:
                current_ratio = round(current_assets / current_liabilities, 2)

        # Additional balance sheet items
        cash_and_equivalents = _safe_stmt_lookup(
            balance_sheet,
            ["Cash And Cash Equivalents",
             "Cash Cash Equivalents And Short Term Investments",
             "Cash Financial"],
        )
        # Also try info dict for cash
        if cash_and_equivalents is None:
            cash_and_equivalents = info.get("totalCash")

        short_term_investments = _safe_stmt_lookup(
            balance_sheet,
            ["Other Short Term Investments", "Short Term Investments"],
        )
        inventory = _safe_stmt_lookup(balance_sheet, ["Inventory"])
        long_term_debt = _safe_stmt_lookup(
            balance_sheet, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
        )
        tangible_book_value = _safe_stmt_lookup(
            balance_sheet, ["Tangible Book Value"]
        )

        # ==================================================================
        # CASH FLOW STATEMENT
        # ==================================================================
        cashflow = stock.cashflow
        capital_expenditure = _safe_stmt_lookup(
            cashflow, ["Capital Expenditure"]
        )
        operating_cashflow_stmt = _safe_stmt_lookup(
            cashflow, ["Operating Cash Flow"]
        )
        free_cashflow_stmt = _safe_stmt_lookup(
            cashflow, ["Free Cash Flow"]
        )

        # ==================================================================
        # INFO DICT — pre-computed by Yahoo Finance
        # ==================================================================
        eps = info.get("trailingEps")
        pe_ratio = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        market_cap = info.get("marketCap")
        dividend_yield = info.get("dividendYield")
        sector = info.get("sector")
        industry = info.get("industry")
        company_name = info.get("shortName") or info.get("longName")

        # Margins from info (pre-calculated, most reliable)
        gross_margin = _safe_round(info.get("grossMargins"))
        operating_margin = _safe_round(info.get("operatingMargins"))
        ebitda_margin = _safe_round(info.get("ebitdaMargins"))
        net_margin = _safe_round(info.get("profitMargins"))

        # Cash flow from info (fallback to statement values)
        free_cashflow = info.get("freeCashflow") or free_cashflow_stmt
        operating_cashflow = info.get("operatingCashflow") or operating_cashflow_stmt

        # Valuation from info
        enterprise_value = info.get("enterpriseValue")
        ev_to_revenue = _safe_round(info.get("enterpriseToRevenue"), 2)
        ev_to_ebitda = _safe_round(info.get("enterpriseToEbitda"), 2)
        price_to_book = _safe_round(info.get("priceToBook"), 2)
        peg_ratio = _safe_round(info.get("pegRatio"), 2)

        # EBITDA from info (fallback to statement)
        ebitda = info.get("ebitda") or ebitda_stmt

        # Other info fields
        roe = _safe_round(info.get("returnOnEquity"))
        roa = _safe_round(info.get("returnOnAssets"))
        book_value_per_share = _safe_round(info.get("bookValue"), 2)
        beta = _safe_round(info.get("beta"), 2)
        shares_outstanding = info.get("sharesOutstanding")
        short_ratio = _safe_round(info.get("shortRatio"), 2)

        # ==================================================================
        # COMPUTED / DERIVED METRICS
        # ==================================================================
        debt_to_equity = None
        if total_debt and total_equity and total_equity != 0:
            debt_to_equity = round(total_debt / total_equity, 2)

        # Capital expenditure as positive number for display
        capex_abs = abs(capital_expenditure) if capital_expenditure else None

        # Net cash / net debt (positive = net cash, negative = net debt)
        net_cash = None
        if cash_and_equivalents is not None and total_debt is not None:
            net_cash = cash_and_equivalents - total_debt

        # Margin-based computations
        fcf_margin = _safe_divide(free_cashflow, revenue)
        operating_cf_margin = _safe_divide(operating_cashflow, revenue)
        fcf_conversion = _safe_divide(free_cashflow, net_income)

        # Efficiency metrics
        rd_pct_revenue = _safe_divide(research_development, revenue)
        sga_pct_revenue = _safe_divide(selling_general_admin, revenue)
        inventory_turnover = _safe_divide(cost_of_revenue, inventory, 1)
        debt_to_ebitda = _safe_divide(total_debt, ebitda, 2)

        # Per-share metrics
        fcf_per_share = _safe_divide(free_cashflow, shares_outstanding, 2)
        tangible_bv_per_share = _safe_divide(
            tangible_book_value, shares_outstanding, 2
        )

        # FCF yield = FCF / market_cap
        fcf_yield = _safe_divide(free_cashflow, market_cap)

        # Rule of 40 (revenue_growth% + FCF_margin% or operating_margin%)
        rule_of_40 = None
        if revenue_growth is not None and fcf_margin is not None:
            rule_of_40 = round(revenue_growth * 100 + fcf_margin * 100, 1)
        elif revenue_growth is not None and operating_margin is not None:
            rule_of_40 = round(revenue_growth * 100 + operating_margin * 100, 1)

        # Interest coverage
        interest_coverage = None
        if operating_income and interest_expense and interest_expense < 0:
            interest_coverage = _safe_divide(
                operating_income, abs(interest_expense), 2
            )

        # ==================================================================
        # BUILD RESULT DICT
        # ==================================================================
        result = {
            # --- Identity ---
            "ticker": ticker,
            "company_name": company_name,
            "sector": sector,
            "industry": industry,

            # --- Core financials (existing, unchanged) ---
            "revenue": revenue,
            "net_income": net_income,
            "eps": eps,
            "pe_ratio": _safe_round(pe_ratio, 1),
            "forward_pe": _safe_round(forward_pe, 1),
            "revenue_growth": _safe_round(revenue_growth),
            "debt_to_equity": debt_to_equity,
            "return_on_equity": roe,
            "current_ratio": current_ratio,
            "total_debt": total_debt,
            "total_equity": total_equity,
            "total_assets": total_assets,
            "market_cap": market_cap,
            "dividend_yield": _safe_round(dividend_yield),

            # --- Profitability margins ---
            "gross_margin": gross_margin,
            "operating_margin": operating_margin,
            "ebitda_margin": ebitda_margin,
            "net_margin": net_margin,
            "fcf_margin": fcf_margin,

            # --- Cash flow ---
            "free_cashflow": free_cashflow,
            "operating_cashflow": operating_cashflow,
            "capital_expenditure": capex_abs,
            "fcf_conversion": fcf_conversion,
            "operating_cf_margin": operating_cf_margin,
            "fcf_yield": fcf_yield,

            # --- Income statement detail ---
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "ebitda": ebitda,
            "cost_of_revenue": cost_of_revenue,
            "research_development": research_development,
            "selling_general_admin": selling_general_admin,

            # --- Balance sheet detail ---
            "cash_and_equivalents": cash_and_equivalents,
            "short_term_investments": short_term_investments,
            "net_cash": net_cash,
            "inventory": inventory,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "long_term_debt": long_term_debt,
            "tangible_book_value": tangible_book_value,

            # --- Valuation multiples ---
            "enterprise_value": enterprise_value,
            "ev_to_revenue": ev_to_revenue,
            "ev_to_ebitda": ev_to_ebitda,
            "price_to_book": price_to_book,
            "peg_ratio": peg_ratio,
            "debt_to_ebitda": debt_to_ebitda,

            # --- Efficiency metrics ---
            "rd_pct_revenue": rd_pct_revenue,
            "sga_pct_revenue": sga_pct_revenue,
            "inventory_turnover": inventory_turnover,
            "rule_of_40": rule_of_40,

            # --- Per-share metrics ---
            "book_value_per_share": book_value_per_share,
            "fcf_per_share": fcf_per_share,
            "tangible_bv_per_share": tangible_bv_per_share,

            # --- Other ---
            "return_on_assets": roa,
            "beta": beta,
            "shares_outstanding": shares_outstanding,
            "short_ratio": short_ratio,
            "interest_coverage": interest_coverage,

            "data_source": "Yahoo Finance",
        }

        # --- Data quality assessment ---
        core_metrics = [
            revenue, net_income, eps, pe_ratio,
            gross_margin, operating_margin, free_cashflow, ebitda,
        ]
        available_count = sum(1 for m in core_metrics if m is not None)

        if available_count >= 6:
            result["data_quality"] = "comprehensive"
        elif available_count >= 3:
            result["data_quality"] = "good"
        elif available_count >= 1:
            result["data_quality"] = "limited"
            logger.warning(f"Limited financial data available for {ticker}")
        else:
            result["data_quality"] = "minimal"
            logger.warning(f"Minimal financial data available for {ticker}")

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
