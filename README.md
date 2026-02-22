# AI Financial Agent

A multi-agent investment analysis system powered by Claude and orchestrated with LangGraph.

Inspired by [@virattt's ai-hedge-fund](https://github.com/virattt/ai-hedge-fund).

## How It Works

Five specialized AI agents collaborate to analyze stocks and make trading decisions:

```
         Fetch Market Data
               |
     +---------+---------+
     |         |         |
Fundamentals  Technicals  Sentiment    (parallel analysis)
     |         |         |
     +---------+---------+
               |
         Risk Manager
               |
       Portfolio Manager
               |
         Trade Decisions
```

### Agents

| Agent | Role |
|-------|------|
| **Fundamentals** | Analyzes financial statements, valuation, and business quality |
| **Technicals** | Computes SMA, RSI, MACD and reads price action |
| **Sentiment** | Reads news headlines and gauges market narrative |
| **Risk Manager** | Evaluates risk, sets position limits, flags concerns |
| **Portfolio Manager** | Synthesizes all signals into buy/sell/hold decisions |

## Setup

### 1. Clone and install

```bash
git clone <your-repo>
cd ai-financial-agent

# Using Poetry (recommended)
poetry install

# Or using pip
pip install langgraph langchain-anthropic langchain-core pydantic rich python-dotenv requests
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 3. Run

```bash
# Analyze specific stocks
python main.py --tickers AAPL,MSFT,NVDA

# Custom portfolio size
python main.py --tickers TSLA,GOOGL --cash 50000
```

## Example Output

```
AI Financial Agent
Analyzing: AAPL, NVDA
Portfolio: $100,000.00

┌──────────────────────────────────────────────────┐
│              Agent Trading Signals                │
├────────┬──────────────────┬────────┬─────────────┤
│ Ticker │ Agent            │ Signal │ Confidence  │
├────────┼──────────────────┼────────┼─────────────┤
│ AAPL   │ Fundamentals     │ BULLISH│ 75%         │
│ AAPL   │ Technicals       │ NEUTRAL│ 60%         │
│ AAPL   │ Sentiment        │ BULLISH│ 70%         │
│ NVDA   │ Fundamentals     │ BULLISH│ 85%         │
│ NVDA   │ Technicals       │ BULLISH│ 72%         │
│ NVDA   │ Sentiment        │ BULLISH│ 80%         │
└────────┴──────────────────┴────────┴─────────────┘
```

## Project Structure

```
ai-financial-agent/
├── main.py                    # CLI entry point
├── src/
│   ├── state.py               # Shared state (Pydantic models)
│   ├── graph.py               # LangGraph workflow orchestration
│   ├── display.py             # Rich terminal output
│   ├── llm/
│   │   └── models.py          # Claude LLM client
│   ├── agents/
│   │   ├── fundamentals.py    # Financial statement analysis
│   │   ├── technicals.py      # Technical indicators (SMA, RSI, MACD)
│   │   ├── sentiment.py       # News sentiment analysis
│   │   ├── risk_manager.py    # Risk assessment & position limits
│   │   └── portfolio_manager.py # Final trade decisions
│   └── tools/
│       └── market_data.py     # Market data fetching (API + synthetic fallback)
├── pyproject.toml
├── .env.example
└── README.md
```

## Data Sources

- **Financial Datasets API** (optional) — real market data from [financialdatasets.ai](https://financialdatasets.ai)
- **Synthetic fallback** — generates plausible demo data when no API key is provided, so you can run it immediately

## Disclaimer

This is for **educational purposes only**. It is not financial advice and is not intended for real trading or investment decisions.
