"""
Portfolio Manager Agent — makes final trade decisions.

Synthesizes all agent signals and risk assessments to produce
actionable trade decisions (buy, sell, or hold).
"""

from src.state import AgentState, TradeDecision
from src.llm.models import call_llm


SYSTEM_PROMPT = """你是一位负责做出最终交易决策的投资组合经理。
你综合来自基本面、技术面和情绪分析师的输入，以及风险管理经理的评估。

你的方法：
- 根据每位分析师的置信度对其信号进行加权
- 严格遵守风险管理经理的仓位限制——绝不超限
- 考虑当前投资组合状态（现有持仓、现金）
- 只在多位分析师给出高确信信号时才交易
- 当信号混合、微弱或数据有限时，倾向于持有
- 按确信度和风险预算比例确定仓位大小

关键规则：
- 如果任何分析师因数据不可用而置信度低，降低其信号权重。
- 除非数据强力支持，否则不要对股票前景做出大胆判断。
- 在推理中坦诚说明局限性。
- 将分析定位为基于可用数据的结果，而非投资建议。

买入决策时，根据以下公式计算具体股数：
  最大投入 = 投资组合现金 × 最大仓位比例
  股数 = floor(最大投入 / 当前股价)

卖出决策时，指定从当前持仓中卖出的股数。
持有时，设置股数为0。"""


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

        risk_text = "暂无风险评估。"
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

股票代码：{ticker}（{company_name}）
当前价格：{f'${current_price:.2f}' if current_price else 'N/A'}
数据质量：{data_quality}

投资组合状态：
- 可用现金：${portfolio_cash:,.2f}
- {ticker} 当前持仓：{current_shares} 股 {f'(${current_shares * current_price:,.2f})' if current_price else ''}

分析师信号：
{signals_text}

风险评估：
{risk_text}

做出你的最终交易决策，输出 TradeDecision，包含：
- action："buy"、"sell" 或 "hold"
- quantity：交易股数（持有则为0）
- confidence：你对此决策的整体确信度
- reasoning：综合所有输入的完整分析（2-3段，用中文撰写）。
  坦诚说明数据局限性，不要夸大结论。
"""

        decision = call_llm(prompt, response_model=TradeDecision)
        decision.ticker = ticker
        decisions.append(decision)

    return {"decisions": decisions}
