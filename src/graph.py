"""
LangGraph workflow — orchestrates the multi-agent financial analysis pipeline.

Pipeline:
  1. Fetch market data (prices, fundamentals, news)
  2. Run analysis agents in parallel: Fundamentals, Technicals, Sentiment
  3. Risk Manager evaluates all signals
  4. Portfolio Manager makes final trade decisions
"""

from __future__ import annotations

from typing import Dict, List, Optional

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.tools.market_data import get_all_market_data
from src.agents.fundamentals import fundamentals_agent
from src.agents.technicals import technicals_agent
from src.agents.sentiment import sentiment_agent
from src.agents.risk_manager import risk_manager_agent
from src.agents.portfolio_manager import portfolio_manager_agent


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def fetch_market_data(state: AgentState) -> AgentState:
    """Node: fetch all market data for the tickers."""
    market_data = get_all_market_data(state["tickers"])
    return {"market_data": market_data}


def run_fundamentals(state: AgentState) -> AgentState:
    """Node: run the fundamentals agent."""
    return fundamentals_agent(state)


def run_technicals(state: AgentState) -> AgentState:
    """Node: run the technicals agent."""
    return technicals_agent(state)


def run_sentiment(state: AgentState) -> AgentState:
    """Node: run the sentiment agent."""
    return sentiment_agent(state)


def run_risk_manager(state: AgentState) -> AgentState:
    """Node: run the risk manager agent."""
    return risk_manager_agent(state)


def run_portfolio_manager(state: AgentState) -> AgentState:
    """Node: run the portfolio manager agent."""
    return portfolio_manager_agent(state)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow.

    The graph structure:

        fetch_data
            |
      +-----+-----+
      |     |     |
    fund  tech  sent    (parallel analysis)
      |     |     |
      +-----+-----+
            |
       risk_manager
            |
     portfolio_manager
            |
           END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("fetch_data", fetch_market_data)
    workflow.add_node("fundamentals", run_fundamentals)
    workflow.add_node("technicals", run_technicals)
    workflow.add_node("sentiment", run_sentiment)
    workflow.add_node("risk_manager", run_risk_manager)
    workflow.add_node("portfolio_manager", run_portfolio_manager)

    # Set entry point
    workflow.set_entry_point("fetch_data")

    # Data fetching -> all three analysis agents
    workflow.add_edge("fetch_data", "fundamentals")
    workflow.add_edge("fetch_data", "technicals")
    workflow.add_edge("fetch_data", "sentiment")

    # All analysis agents -> risk manager
    workflow.add_edge("fundamentals", "risk_manager")
    workflow.add_edge("technicals", "risk_manager")
    workflow.add_edge("sentiment", "risk_manager")

    # Risk manager -> portfolio manager -> END
    workflow.add_edge("risk_manager", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    return workflow.compile()


def run_hedge_fund(
    tickers: List[str],
    portfolio_cash: float = 100_000.0,
    portfolio_positions: Optional[Dict[str, int]] = None,
    research_documents: Optional[Dict[str, list]] = None,
) -> AgentState:
    """Run the full hedge fund analysis pipeline.

    Args:
        tickers: List of stock ticker symbols to analyze.
        portfolio_cash: Available cash in the portfolio.
        portfolio_positions: Current holdings {ticker: shares}.
        research_documents: Uploaded research PDFs {ticker: [{filename, text}]}.

    Returns:
        The final AgentState with all signals, assessments, and decisions.
    """
    graph = build_graph()

    initial_state: AgentState = {
        "tickers": tickers,
        "portfolio_cash": portfolio_cash,
        "portfolio_positions": portfolio_positions or {},
        "market_data": {},
        "signals": [],
        "risk_assessments": [],
        "decisions": [],
        "research_documents": research_documents or {},
    }

    return graph.invoke(initial_state)