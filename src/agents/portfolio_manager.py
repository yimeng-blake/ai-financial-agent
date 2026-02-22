"""
Portfolio Manager Agent — makes final trade decisions.

Synthesizes all agent signals and risk assessments to produce
actionable trade decisions (buy, sell, or hold).
"""

from src.state import AgentState, TradeDecision
from src.llm.models import call_llm


SYSTEM_PROMPT = """You are a portfolio manager responsible for making final trading decisions.
You synthesize inputs from the fundamentals, technicals, and sentiment analysts,
along with the risk manager's assessment.

Your approach:
- Weight each analyst's signal based on their confidence level
- Respect the risk manager's position limits strictly — never exceed them
- Consider the current portfolio state (existing positions, cash)
- Only trade when you have high conviction from multiple analysts
- Prefer to hold when signals are mixed, weak, or when data was limited
- Size positions proportionally to conviction and risk budget

CRITICAL RULES:
- If any analyst had low confidence due to data unavailability, weight their signal less.
- Do NOT make bold claims about a stock's prospects unless strongly supported by the data.
- Be transparent about limitations in your reasoning.
- Frame your analysis as based on the available data, not as investment advice.

For buy decisions, calculate a specific share quantity based on:
  max_spend = portfolio_cash * max_position_size
  quantity = floor(max_spend / current_price)

For sell decisions, specify how many shares to sell from current holdings.
For hold, set quantity to 0."""


def portfolio_manager_agent(state: AgentState) -> AgentState:
    """Make final trade decisions for each ticker."""
    decisions = []
    signals = state.get("signals", [])
    risk_assessments = state.get("risk_assessments", [])
    portfolio_cash = state.get("portfolio_cash", 100_000)
    positions = state.get("portfolio_positions", {})

    for ticker in state["tickers"]:
        ticker_signals = [s for s in signals if s.ticker == ticker]
        ticker_risk = next((r for r in risk_assessments if r.ticker == ticker), None)

        signals_text = "\n".join(
            f"- {s.agent_name}: {s.signal} (confidence: {s.confidence:.0%})\n  {s.reasoning[:400]}"
            for s in ticker_signals
        )

        risk_text = "No risk assessment available."
        if ticker_risk:
            risk_text = (
                f"Risk Score: {ticker_risk.risk_score:.0%}\n"
                f"Max Position Size: {ticker_risk.max_position_size:.0%} of portfolio\n"
                f"Risk Factors: {', '.join(ticker_risk.risk_factors)}\n"
                f"Assessment: {ticker_risk.reasoning[:400]}"
            )

        data = state["market_data"].get(ticker, {})
        prices = data.get("prices", [])
        fundamentals = data.get("fundamentals", {})
        current_price = prices[-1]["close"] if prices else None
        current_shares = positions.get(ticker, 0)
        data_quality = fundamentals.get("data_quality", "unknown")
        company_name = fundamentals.get("company_name") or ticker

        prompt = f"""{SYSTEM_PROMPT}

TICKER: {ticker} ({company_name})
CURRENT PRICE: {f'${current_price:.2f}' if current_price else 'N/A'}
DATA QUALITY: {data_quality}

PORTFOLIO STATE:
- Cash Available: ${portfolio_cash:,.2f}
- Current Position in {ticker}: {current_shares} shares {f'(${current_shares * current_price:,.2f})' if current_price else ''}

ANALYST SIGNALS:
{signals_text}

RISK ASSESSMENT:
{risk_text}

Make your final trade decision as a TradeDecision with:
- action: "buy", "sell", or "hold"
- quantity: number of shares to trade (0 if hold)
- confidence: your overall conviction in this decision
- reasoning: full synthesis of all inputs (2-3 paragraphs). Be transparent about
  data limitations and do not overstate conclusions.
"""

        decision = call_llm(prompt, response_model=TradeDecision)
        decision.ticker = ticker
        decisions.append(decision)

    return {"decisions": decisions}
