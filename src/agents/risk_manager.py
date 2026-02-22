"""
Risk Manager Agent — evaluates risk and sets position limits.

Reviews the signals from other agents, considers portfolio exposure,
volatility, and concentration risk to produce risk assessments.
"""

from src.state import AgentState, RiskAssessment
from src.llm.models import call_llm


SYSTEM_PROMPT = """You are a risk manager for an investment portfolio. Your job is to
evaluate the risk of each potential trade and set appropriate position limits.

Your approach:
- Review all agent signals and their confidence levels
- Assess price volatility from recent data
- Consider portfolio concentration risk
- Evaluate downside scenarios
- Set position limits that protect capital while allowing upside

CRITICAL RULES:
- Be conservative — capital preservation is your top priority.
- When agents disagree, lean toward smaller positions.
- When agent confidence is low (especially if data was unavailable), further reduce position sizes.
- Do NOT fabricate risk factors. Only cite risks supported by the data provided.
- If an agent reported "neutral with low confidence" due to data unavailability,
  treat this as additional uncertainty and factor it into your risk score."""


def risk_manager_agent(state: AgentState) -> AgentState:
    """Produce risk assessments for each ticker based on all available signals."""
    risk_assessments = []
    signals = state.get("signals", [])
    portfolio_cash = state.get("portfolio_cash", 100_000)
    positions = state.get("portfolio_positions", {})

    for ticker in state["tickers"]:
        # Gather all signals for this ticker
        ticker_signals = [s for s in signals if s.ticker == ticker]
        signals_summary = "\n".join(
            f"- {s.agent_name}: {s.signal} (confidence: {s.confidence:.0%})\n  Reasoning: {s.reasoning[:300]}..."
            for s in ticker_signals
        )

        # Get price data for volatility assessment
        data = state["market_data"].get(ticker, {})
        prices = data.get("prices", [])
        fundamentals = data.get("fundamentals", {})
        data_quality = fundamentals.get("data_quality", "unknown")

        if len(prices) >= 20:
            closes = [p["close"] for p in prices[-20:]]
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
            avg_price = closes[-1]
        else:
            volatility = None
            avg_price = prices[-1]["close"] if prices else None

        current_position = positions.get(ticker, 0)

        prompt = f"""{SYSTEM_PROMPT}

TICKER: {ticker}
CURRENT PRICE: {f'${avg_price:.2f}' if avg_price else 'N/A'}
DAILY VOLATILITY: {f'{volatility:.2%}' if volatility else 'N/A (insufficient price data)'}
DATA QUALITY: {data_quality}
CURRENT POSITION: {current_position} shares {f'(${current_position * avg_price:,.2f})' if avg_price else ''}
PORTFOLIO CASH: ${portfolio_cash:,.2f}

AGENT SIGNALS:
{signals_summary}

Note: If any agent reported low confidence due to data unavailability, factor that
uncertainty into your risk assessment. Lack of data is itself a risk factor.

Produce a RiskAssessment with:
- risk_score: 0.0 (very low risk) to 1.0 (very high risk)
- max_position_size: as a fraction of total portfolio (0.0 to 0.25)
- risk_factors: list of key risks (3-5 items)
- reasoning: your detailed risk analysis
"""

        assessment = call_llm(prompt, response_model=RiskAssessment)
        assessment.ticker = ticker
        risk_assessments.append(assessment)

    return {"risk_assessments": risk_assessments}
