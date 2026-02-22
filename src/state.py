"""
Shared state for the multi-agent financial analysis pipeline.

The AgentState flows through the LangGraph workflow. Each agent reads from it,
performs analysis, and writes its signal back into the state.
"""

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class TradingSignal(BaseModel):
    """A structured trading signal produced by an agent."""

    agent_name: str = Field(description="Name of the agent that produced this signal")
    ticker: str = Field(description="Stock ticker symbol")
    signal: str = Field(description="Trading signal: 'bullish', 'bearish', or 'neutral'")
    confidence: float = Field(
        description="Confidence level from 0.0 to 1.0", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Detailed reasoning behind the signal")


class RiskAssessment(BaseModel):
    """Risk assessment produced by the risk manager."""

    ticker: str = Field(description="Stock ticker symbol")
    risk_score: float = Field(
        description="Overall risk score from 0.0 (low) to 1.0 (high)", ge=0.0, le=1.0
    )
    max_position_size: float = Field(
        description="Recommended max position size as fraction of portfolio"
    )
    risk_factors: list[str] = Field(description="Key risk factors identified")
    reasoning: str = Field(description="Detailed risk reasoning")


class TradeDecision(BaseModel):
    """Final trade decision from the portfolio manager."""

    ticker: str = Field(description="Stock ticker symbol")
    action: str = Field(description="'buy', 'sell', or 'hold'")
    quantity: int = Field(description="Number of shares (0 if hold)")
    confidence: float = Field(description="Decision confidence", ge=0.0, le=1.0)
    reasoning: str = Field(description="Full reasoning incorporating all agent signals")


class AgentState(TypedDict):
    """The shared state that flows through the LangGraph pipeline."""

    # Input
    tickers: list[str]
    portfolio_cash: float
    portfolio_positions: dict[str, int]  # ticker -> shares held

    # Market data (populated by data tools)
    market_data: dict  # ticker -> {prices, fundamentals, news}

    # Agent signals (populated by each agent — merged across parallel branches)
    signals: Annotated[list[TradingSignal], operator.add]

    # Risk assessment (populated by risk manager)
    risk_assessments: Annotated[list[RiskAssessment], operator.add]

    # Final decisions (populated by portfolio manager)
    decisions: Annotated[list[TradeDecision], operator.add]

    # Research documents (optional) — uploaded PDFs keyed by ticker
    # Structure: {ticker: [{"filename": str, "text": str}, ...]}
    research_documents: dict[str, list[dict[str, str]]]
