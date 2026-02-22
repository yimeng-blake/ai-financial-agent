"""
Fundamentals Agent — sector-aware financial statement and valuation analysis.

Dynamically selects a specialized analysis framework based on each ticker's
sector/industry classification (e.g., SaaS metrics for software, cycle analysis
for semiconductors, pipeline assessment for biotech).
"""

from src.state import AgentState, TradingSignal
from src.llm.models import call_llm
from src.agents.sector_prompts import get_sector_config


BASE_RULES = """CRITICAL RULES:
- ONLY reference data that is explicitly provided below. Never fabricate or assume numbers.
- If a metric is "N/A" or missing, acknowledge it is unavailable — do NOT guess.
- If data quality is "limited" or "unavailable", state this clearly and lower your confidence accordingly.
- Base your reasoning strictly on the numbers given. Do not reference external knowledge about recent
  earnings reports, guidance, or news unless that data is explicitly provided.
- Be precise: cite the exact figures from the data below.

You must produce a clear trading signal: bullish, bearish, or neutral."""


def _format_research_section(docs: list[dict[str, str]]) -> str:
    """Format uploaded research documents into a prompt section."""
    if not docs:
        return ""
    lines = [
        "\n=== UPLOADED RESEARCH DOCUMENTS ===",
        "The following sell-side/buy-side research documents were provided by the user.",
        "Treat these as supplementary analyst perspectives. Cross-reference their claims",
        "against the hard financial data above. Note any discrepancies. Do NOT treat",
        "research opinions as facts — they are analyst viewpoints.\n",
    ]
    for idx, doc in enumerate(docs[:3], 1):
        lines.append(f"--- Document {idx}: {doc['filename']} ---")
        lines.append(doc["text"])
        lines.append("")
    return "\n".join(lines)


def fundamentals_agent(state: AgentState) -> AgentState:
    """Analyze each ticker's fundamentals using sector-specific frameworks."""
    signals = []
    research_docs = state.get("research_documents", {})

    for ticker in state["tickers"]:
        data = state["market_data"].get(ticker, {})
        fundamentals = data.get("fundamentals", {})
        prices = data.get("prices", [])

        current_price = prices[-1]["close"] if prices else None
        data_quality = fundamentals.get("data_quality", "unknown")
        company_name = fundamentals.get("company_name") or ticker
        sector = fundamentals.get("sector") or "N/A"
        industry = fundamentals.get("industry") or "N/A"

        # Resolve sector-specific analysis framework
        sector_config = get_sector_config(
            fundamentals.get("industry"),
            fundamentals.get("sector"),
        )

        def fmt(val, spec):
            """Format a numeric value, returning 'N/A' if missing."""
            if isinstance(val, (int, float)):
                return f"{val:{spec}}"
            return "N/A"

        # Handle case where we have no fundamental data at all
        if data_quality == "unavailable":
            signals.append(TradingSignal(
                agent_name=f"Fundamentals Agent ({sector_config.label})",
                ticker=ticker,
                signal="neutral",
                confidence=0.1,
                reasoning=(
                    f"Fundamental data is unavailable for {ticker}. "
                    f"Cannot perform meaningful fundamental analysis without financial statements. "
                    f"Error: {fundamentals.get('error', 'Unknown')}. "
                    f"Defaulting to neutral with very low confidence."
                ),
            ))
            continue

        prompt = f"""You are a fundamental analysis expert specializing in {sector_config.label} companies.

=== SECTOR-SPECIFIC ANALYSIS FRAMEWORK: {sector_config.label} ===
{sector_config.analysis_framework}

{BASE_RULES}

Analyze the following financial data for {ticker} ({company_name}):

DATA SOURCE: {fundamentals.get('data_source', 'Unknown')}
DATA QUALITY: {data_quality}
SECTOR: {sector}
INDUSTRY: {industry}

FUNDAMENTALS:
- Revenue: ${fmt(fundamentals.get('revenue'), ',.0f')}
- Net Income: ${fmt(fundamentals.get('net_income'), ',.0f')}
- EPS (trailing): ${fundamentals.get('eps', 'N/A')}
- P/E Ratio (trailing): {fundamentals.get('pe_ratio', 'N/A')}
- Forward P/E: {fundamentals.get('forward_pe', 'N/A')}
- Revenue Growth (YoY): {fmt(fundamentals.get('revenue_growth'), '.1%')}
- Debt-to-Equity: {fundamentals.get('debt_to_equity', 'N/A')}
- Return on Equity: {fmt(fundamentals.get('return_on_equity'), '.1%')}
- Current Ratio: {fundamentals.get('current_ratio', 'N/A')}
- Market Cap: ${fmt(fundamentals.get('market_cap'), ',.0f')}
- Dividend Yield: {fmt(fundamentals.get('dividend_yield'), '.2%')}

Current Stock Price: ${current_price if current_price else 'N/A'}
{_format_research_section(research_docs.get(ticker, []))}
Produce your analysis as a TradingSignal with:
- signal: "bullish", "bearish", or "neutral"
- confidence: 0.0 to 1.0 (lower if data quality is limited)
- reasoning: your detailed fundamental analysis using the {sector_config.label} framework above.
  Apply the sector-specific metrics and valuation approach. Only cite numbers provided above.
  {"If research documents were provided, integrate relevant analyst insights into your reasoning, noting where they align with or contradict the financial data." if research_docs.get(ticker) else ""}
"""

        signal = call_llm(prompt, response_model=TradingSignal)
        signal.agent_name = f"Fundamentals Agent ({sector_config.label})"
        signal.ticker = ticker
        signals.append(signal)

    return {"signals": signals}
