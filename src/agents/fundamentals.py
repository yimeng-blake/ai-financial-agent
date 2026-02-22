"""
Fundamentals Agent — sector-aware financial statement and valuation analysis.

Dynamically selects a specialized analysis framework based on each ticker's
sector/industry classification (e.g., SaaS metrics for software, cycle analysis
for semiconductors, pipeline assessment for biotech).
"""

from src.state import AgentState, TradingSignal
from src.llm.models import call_llm
from src.agents.sector_prompts import get_sector_config


BASE_RULES = """关键规则：
- 只引用下方明确提供的数据。绝不编造或假设数字。
- 如果某个指标为"N/A"或缺失，请承认该数据不可用——不要猜测。
- 如果数据质量为"limited"或"unavailable"，请明确说明并相应降低置信度。
- 严格基于给定数据进行推理。除非数据中明确提供，否则不要引用关于近期财报、指引或新闻的外部知识。
- 做到精确：引用下方数据中的确切数字。

你必须给出一个明确的交易信号：bullish（看多）、bearish（看空）或 neutral（中性）。"""


def _format_research_section(docs: list[dict[str, str]]) -> str:
    """Format uploaded research documents into a prompt section."""
    if not docs:
        return ""
    lines = [
        "\n=== 用户上传的研究报告 ===",
        "以下是用户提供的卖方/买方研究报告。",
        "将这些内容视为补充性的分析师观点。请将其论点与上方的硬性财务数据进行交叉验证。",
        "标注任何不一致之处。不要将研究报告中的观点视为事实——它们是分析师的看法。\n",
    ]
    for idx, doc in enumerate(docs[:3], 1):
        lines.append(f"--- 文档 {idx}: {doc['filename']} ---")
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
                agent_name=f"基本面分析师（{sector_config.label}）",
                ticker=ticker,
                signal="neutral",
                confidence=0.1,
                reasoning=(
                    f"{ticker} 的基本面数据不可用。"
                    f"缺少财务报表，无法进行有意义的基本面分析。"
                    f"错误信息：{fundamentals.get('error', '未知')}。"
                    f"默认输出中性信号，置信度极低。"
                ),
            ))
            continue

        prompt = f"""你是一位专注于{sector_config.label}公司的基本面分析专家。

=== 行业专属分析框架：{sector_config.label} ===
{sector_config.analysis_framework}

{BASE_RULES}

请分析以下 {ticker}（{company_name}）的财务数据：

数据来源：{fundamentals.get('data_source', '未知')}
数据质量：{data_quality}
板块：{sector}
行业：{industry}

基本面数据：
- 营收：${fmt(fundamentals.get('revenue'), ',.0f')}
- 净利润：${fmt(fundamentals.get('net_income'), ',.0f')}
- EPS（滚动）：${fundamentals.get('eps', 'N/A')}
- P/E（滚动）：{fundamentals.get('pe_ratio', 'N/A')}
- Forward P/E：{fundamentals.get('forward_pe', 'N/A')}
- 营收增长率（同比）：{fmt(fundamentals.get('revenue_growth'), '.1%')}
- Debt-to-Equity：{fundamentals.get('debt_to_equity', 'N/A')}
- ROE：{fmt(fundamentals.get('return_on_equity'), '.1%')}
- 流动比率：{fundamentals.get('current_ratio', 'N/A')}
- 总市值：${fmt(fundamentals.get('market_cap'), ',.0f')}
- 股息率：{fmt(fundamentals.get('dividend_yield'), '.2%')}

当前股价：${current_price if current_price else 'N/A'}
{_format_research_section(research_docs.get(ticker, []))}
请输出 TradingSignal，包含：
- signal："bullish"、"bearish" 或 "neutral"
- confidence：0.0 到 1.0（数据质量受限时应降低）
- reasoning：使用上述{sector_config.label}框架进行详细的基本面分析（用中文撰写）。
  运用行业专属指标和估值方法。只引用上方提供的数据。
  {"如果提供了研究报告，请将相关分析师观点整合到你的推理中，指出它们与财务数据的一致或矛盾之处。" if research_docs.get(ticker) else ""}
"""

        signal = call_llm(prompt, response_model=TradingSignal)
        signal.agent_name = f"基本面分析师（{sector_config.label}）"
        signal.ticker = ticker
        signals.append(signal)

    return {"signals": signals}
