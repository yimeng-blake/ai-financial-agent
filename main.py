"""
AI Financial Agent — Main entry point.

Usage:
    python main.py --tickers AAPL,MSFT,NVDA
    python main.py --tickers TSLA --cash 50000
"""

import argparse
import sys
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from src.graph import run_hedge_fund
from src.display import display_results

load_dotenv()
console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="AI Financial Agent — Multi-agent investment analysis powered by Claude"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated list of stock tickers (e.g., AAPL,MSFT,NVDA)",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100_000.0,
        help="Portfolio cash available (default: $100,000)",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        default=True,
        help="Show detailed reasoning from each agent (default: True)",
    )

    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]AI Financial Agent[/bold cyan]\n"
            f"Analyzing: [bold]{', '.join(tickers)}[/bold]\n"
            f"Portfolio: [bold green]${args.cash:,.2f}[/bold green]",
            border_style="cyan",
        )
    )
    console.print()

    # Show pipeline stages
    stages = [
        ("Fetching market data", "prices, fundamentals, news"),
        ("Fundamentals Agent", "financial statement analysis"),
        ("Technicals Agent", "price action & indicators"),
        ("Sentiment Agent", "news & market sentiment"),
        ("Risk Manager", "risk assessment & position limits"),
        ("Portfolio Manager", "final trade decisions"),
    ]

    console.print("[bold]Pipeline stages:[/bold]")
    for i, (name, desc) in enumerate(stages, 1):
        console.print(f"  {i}. [cyan]{name}[/cyan] — {desc}")
    console.print()

    start_time = time.time()

    with console.status("[bold green]Running analysis pipeline...") as status:
        try:
            result = run_hedge_fund(
                tickers=tickers,
                portfolio_cash=args.cash,
            )
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            console.print(
                "\nMake sure your ANTHROPIC_API_KEY is set in your .env file."
            )
            sys.exit(1)

    elapsed = time.time() - start_time
    console.print(f"[dim]Analysis completed in {elapsed:.1f}s[/dim]\n")

    display_results(result)


if __name__ == "__main__":
    main()
