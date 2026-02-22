"""
Rich terminal display for the hedge fund results.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from src.state import AgentState


console = Console()


def display_results(state: AgentState) -> None:
    """Pretty-print the full analysis results to the terminal."""

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]AI Financial Agent — Analysis Report[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    # ---- Agent Signals Table ----
    signals_table = Table(
        title="Agent Trading Signals",
        show_header=True,
        header_style="bold magenta",
    )
    signals_table.add_column("Ticker", style="cyan", width=8)
    signals_table.add_column("Agent", width=20)
    signals_table.add_column("Signal", width=10)
    signals_table.add_column("Confidence", width=12)
    signals_table.add_column("Reasoning", width=60)

    for signal in state.get("signals", []):
        signal_color = {
            "bullish": "green",
            "bearish": "red",
            "neutral": "yellow",
        }.get(signal.signal, "white")

        signals_table.add_row(
            signal.ticker,
            signal.agent_name,
            Text(signal.signal.upper(), style=f"bold {signal_color}"),
            f"{signal.confidence:.0%}",
            signal.reasoning[:80] + "..." if len(signal.reasoning) > 80 else signal.reasoning,
        )

    console.print(signals_table)
    console.print()

    # ---- Risk Assessments Table ----
    risk_table = Table(
        title="Risk Assessments",
        show_header=True,
        header_style="bold magenta",
    )
    risk_table.add_column("Ticker", style="cyan", width=8)
    risk_table.add_column("Risk Score", width=12)
    risk_table.add_column("Max Position", width=14)
    risk_table.add_column("Key Risks", width=50)

    for risk in state.get("risk_assessments", []):
        risk_color = "green" if risk.risk_score < 0.3 else "yellow" if risk.risk_score < 0.6 else "red"
        risk_table.add_row(
            risk.ticker,
            Text(f"{risk.risk_score:.0%}", style=f"bold {risk_color}"),
            f"{risk.max_position_size:.0%}",
            ", ".join(risk.risk_factors[:3]),
        )

    console.print(risk_table)
    console.print()

    # ---- Final Decisions Table ----
    decisions_table = Table(
        title="Portfolio Decisions",
        show_header=True,
        header_style="bold magenta",
    )
    decisions_table.add_column("Ticker", style="cyan", width=8)
    decisions_table.add_column("Action", width=8)
    decisions_table.add_column("Quantity", width=10)
    decisions_table.add_column("Confidence", width=12)
    decisions_table.add_column("Reasoning", width=60)

    for decision in state.get("decisions", []):
        action_color = {
            "buy": "bold green",
            "sell": "bold red",
            "hold": "bold yellow",
        }.get(decision.action, "white")

        decisions_table.add_row(
            decision.ticker,
            Text(decision.action.upper(), style=action_color),
            str(decision.quantity),
            f"{decision.confidence:.0%}",
            decision.reasoning[:80] + "..." if len(decision.reasoning) > 80 else decision.reasoning,
        )

    console.print(decisions_table)
    console.print()

    # ---- Detailed reasoning for each decision ----
    for decision in state.get("decisions", []):
        action_color = {"buy": "green", "sell": "red", "hold": "yellow"}.get(decision.action, "white")
        console.print(
            Panel(
                decision.reasoning,
                title=f"[bold]{decision.ticker}[/bold] — {decision.action.upper()} {decision.quantity} shares",
                border_style=action_color,
            )
        )
        console.print()
