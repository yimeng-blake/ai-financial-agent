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

        # Helper alias for cleaner template
        g = fundamentals.get

        prompt = f"""你是一位专注于{sector_config.label}公司的基本面分析专家。

=== 行业专属分析框架：{sector_config.label} ===
{sector_config.analysis_framework}

{BASE_RULES}

请分析以下 {ticker}（{company_name}）的财务数据：

数据来源：{g('data_source', '未知')}
数据质量：{data_quality}
板块：{sector}
行业：{industry}

=== 核心财务数据 ===
- 营收：${fmt(g('revenue'), ',.0f')}
- 营收增长率（同比）：{fmt(g('revenue_growth'), '.1%')}
- 净利润：${fmt(g('net_income'), ',.0f')}
- EPS（滚动）：{g('eps', 'N/A')}

=== 盈利能力 ===
- 毛利率：{fmt(g('gross_margin'), '.1%')}
- 营业利润率：{fmt(g('operating_margin'), '.1%')}
- EBITDA利润率：{fmt(g('ebitda_margin'), '.1%')}
- 净利率：{fmt(g('net_margin'), '.1%')}
- FCF利润率：{fmt(g('fcf_margin'), '.1%')}
- ROE：{fmt(g('return_on_equity'), '.1%')}
- ROA：{fmt(g('return_on_assets'), '.1%')}

=== 现金流 ===
- 自由现金流（FCF）：${fmt(g('free_cashflow'), ',.0f')}
- 经营性现金流：${fmt(g('operating_cashflow'), ',.0f')}
- 资本支出：${fmt(g('capital_expenditure'), ',.0f')}
- FCF转化率（FCF/净利润）：{fmt(g('fcf_conversion'), '.1%')}
- FCF收益率（FCF/市值）：{fmt(g('fcf_yield'), '.2%')}

=== 利润表明细 ===
- 毛利润：${fmt(g('gross_profit'), ',.0f')}
- 营业利润：${fmt(g('operating_income'), ',.0f')}
- EBITDA：${fmt(g('ebitda'), ',.0f')}
- 营业成本：${fmt(g('cost_of_revenue'), ',.0f')}
- 研发费用：${fmt(g('research_development'), ',.0f')}
- 销售管理费用：${fmt(g('selling_general_admin'), ',.0f')}

=== 资产负债表 ===
- 总资产：${fmt(g('total_assets'), ',.0f')}
- 总权益：${fmt(g('total_equity'), ',.0f')}
- 总负债：${fmt(g('total_debt'), ',.0f')}
- 现金及等价物：${fmt(g('cash_and_equivalents'), ',.0f')}
- 短期投资：${fmt(g('short_term_investments'), ',.0f')}
- 净现金/净负债：${fmt(g('net_cash'), ',.0f')}
- 库存：${fmt(g('inventory'), ',.0f')}
- 流动比率：{g('current_ratio', 'N/A')}
- 长期负债：${fmt(g('long_term_debt'), ',.0f')}

=== 估值倍数 ===
- P/E（滚动）：{g('pe_ratio', 'N/A')}
- Forward P/E：{g('forward_pe', 'N/A')}
- EV/Revenue：{g('ev_to_revenue', 'N/A')}
- EV/EBITDA：{g('ev_to_ebitda', 'N/A')}
- P/B（市净率）：{g('price_to_book', 'N/A')}
- PEG比率：{g('peg_ratio', 'N/A')}
- 企业价值：${fmt(g('enterprise_value'), ',.0f')}
- Debt-to-Equity：{g('debt_to_equity', 'N/A')}
- Debt-to-EBITDA：{g('debt_to_ebitda', 'N/A')}

=== 效率指标 ===
- 研发占营收比：{fmt(g('rd_pct_revenue'), '.1%')}
- 销售管理费占营收比：{fmt(g('sga_pct_revenue'), '.1%')}
- 库存周转率：{fmt(g('inventory_turnover'), '.1f')}
- Rule of 40：{g('rule_of_40', 'N/A')}

=== 每股指标 ===
- 每股账面价值：${g('book_value_per_share', 'N/A')}
- 每股FCF：${fmt(g('fcf_per_share'), '.2f')}
- 每股有形账面价值：${fmt(g('tangible_bv_per_share'), '.2f')}

=== 市场数据 ===
- 总市值：${fmt(g('market_cap'), ',.0f')}
- 股息率：{fmt(g('dividend_yield'), '.2%')}
- Beta：{g('beta', 'N/A')}
- 做空比率：{g('short_ratio', 'N/A')}
- 利息覆盖率：{g('interest_coverage', 'N/A')}

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
