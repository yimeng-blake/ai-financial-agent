"""
Risk Manager Agent — evaluates risk and sets position limits.

Reviews the signals from other agents, considers portfolio exposure,
volatility, and concentration risk to produce risk assessments.
"""

from src.state import AgentState, RiskAssessment
from src.llm.models import call_llm


SYSTEM_PROMPT = """你是一位投资组合的风险管理经理。你的工作是评估每笔潜在交易的风险并设定适当的仓位限制。

你的方法：
- 审查所有智能体信号及其置信度
- 根据近期数据评估价格波动率
- 考虑投资组合集中度风险
- 评估下行情景
- 设定在保护资本的同时允许上行空间的仓位限制

关键规则：
- 保守为上——资本保全是你的首要任务。
- 当智能体意见分歧时，倾向于更小的仓位。
- 当智能体置信度低（尤其是因为数据不可用时），进一步减小仓位。
- 不要编造风险因素。只引用数据支持的风险。
- 如果某个智能体因数据不可用而报告"中性、低置信度"，将此视为额外的不确定性并纳入风险评分。"""


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

股票代码：{ticker}
当前价格：{f'${avg_price:.2f}' if avg_price else 'N/A'}
日波动率：{f'{volatility:.2%}' if volatility else 'N/A（价格数据不足）'}
数据质量：{data_quality}
当前持仓：{current_position} 股 {f'(${current_position * avg_price:,.2f})' if avg_price else ''}
投资组合现金：${portfolio_cash:,.2f}

智能体信号：
{signals_summary}

注意：如果有任何智能体因数据不可用而报告低置信度，请将该不确定性纳入你的风险评估。
数据缺失本身就是一个风险因素。

请输出 RiskAssessment，包含：
- risk_score：0.0（极低风险）到 1.0（极高风险）
- max_position_size：占总投资组合的比例（0.0 到 0.25）
- risk_factors：主要风险列表（3-5项）
- reasoning：你的详细风险分析（用中文撰写）
"""

        assessment = call_llm(prompt, response_model=RiskAssessment)
        assessment.ticker = ticker
        risk_assessments.append(assessment)

    return {"risk_assessments": risk_assessments}
