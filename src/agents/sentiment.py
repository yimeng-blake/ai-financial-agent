"""
Sentiment Agent — sector-aware news and social media sentiment analysis.

Reads recent news headlines and X/Twitter posts to assess the overall
sentiment landscape for each ticker, with sector-specific context for
interpreting what types of signals matter most.
"""

from src.state import AgentState, TradingSignal
from src.llm.models import call_llm
from src.agents.sector_prompts import get_sector_config


SYSTEM_PROMPT = """You are a market sentiment analyst. You specialize in reading
news headlines and social media posts to gauge investor sentiment.

Your approach:
- Analyze the tone and content of recent news headlines
- Analyze social media posts for real-time market sentiment and retail investor mood
- Identify potential catalysts (positive or negative)
- Assess whether sentiment is shifting (improving or deteriorating)
- Consider the reliability and significance of the sources
- Weigh how sentiment might impact short-to-medium term price action

SOURCE WEIGHTING (CRITICAL):
- INSTITUTIONAL NEWS (Reuters, Bloomberg, WSJ, etc.) carries MORE weight than social media.
  News from established financial outlets represents vetted, fact-checked reporting.
- SOCIAL MEDIA (X/Twitter) provides supplementary real-time sentiment signals.
  Tweets are useful for gauging retail investor mood and spotting emerging narratives,
  but individual tweets can be speculative, biased, or misleading.
- When news and social media sentiment CONFLICT, lean toward the news signal.
- When they ALIGN, this reinforces the direction and increases your confidence.
- Tweets from verified accounts with high follower counts are more credible.
- High engagement (likes, retweets) indicates the post resonated widely but
  does NOT guarantee accuracy.

CRITICAL RULES:
- ONLY analyze the headlines and tweets provided below. Do NOT fabricate or invent data.
- If no headlines AND no tweets are available, clearly state that sentiment analysis cannot be
  performed and default to neutral with low confidence.
- Do NOT claim specific earnings results, guidance, or corporate actions unless
  a headline or tweet explicitly states them.
- Be precise: reference specific headlines or tweets when building your argument.
- Distinguish between factual reporting and opinion/speculation.
- Headlines are from real news sources and tweets are from real X/Twitter posts —
  treat them as real data."""


def _format_tweets_section(tweets: list) -> str:
    """Format tweets into a labeled prompt section with author metadata."""
    lines = []
    for t in tweets:
        verified_tag = " [VERIFIED]" if t.get("verified") else ""
        engagement = f"{t['likes']} likes, {t['retweets']} RTs"
        lines.append(
            f"- [{t['date']}] {t['author']}{verified_tag} "
            f"({t['author_followers']:,} followers, {engagement}): "
            f"{t['text']}"
        )
    return "\n".join(lines)


def sentiment_agent(state: AgentState) -> AgentState:
    """Analyze news and social media sentiment for each ticker
    with sector-aware interpretation."""
    signals = []

    for ticker in state["tickers"]:
        data = state["market_data"].get(ticker, {})
        fundamentals = data.get("fundamentals", {})
        news = data.get("news", [])
        tweets = data.get("tweets", [])

        # Resolve sector config for context-aware interpretation
        sector_config = get_sector_config(
            fundamentals.get("industry"),
            fundamentals.get("sector"),
        )

        # Handle case where neither news nor tweets are available
        if not news and not tweets:
            signals.append(TradingSignal(
                agent_name=f"Sentiment Agent ({sector_config.label})",
                ticker=ticker,
                signal="neutral",
                confidence=0.15,
                reasoning=(
                    f"No recent news headlines or social media posts are available "
                    f"for {ticker}. Without any sentiment data, analysis cannot be "
                    f"meaningfully performed. Defaulting to neutral with low confidence. "
                    f"The absence of data is itself neutral — it does not indicate "
                    f"positive or negative sentiment."
                ),
            ))
            continue

        # --- Build the news section ---
        if news:
            headlines_text = "\n".join(
                f"- [{item['date']}] ({item['source']}) {item['title']}"
                for item in news
            )
            news_section = (
                f"=== INSTITUTIONAL NEWS ({len(news)} headlines) ===\n"
                f"[WEIGHT: PRIMARY — these are vetted news sources]\n\n"
                f"{headlines_text}"
            )
        else:
            news_section = (
                "=== INSTITUTIONAL NEWS ===\n"
                "No news headlines available. Rely on social media signals "
                "with REDUCED confidence."
            )

        # --- Build the tweets section ---
        if tweets:
            tweets_text = _format_tweets_section(tweets)
            tweets_section = (
                f"=== SOCIAL MEDIA / X POSTS ({len(tweets)} posts) ===\n"
                f"[WEIGHT: SUPPLEMENTARY — real-time sentiment, may be speculative]\n\n"
                f"{tweets_text}"
            )
        else:
            tweets_section = (
                "=== SOCIAL MEDIA / X POSTS ===\n"
                "No social media posts available. Base analysis on news headlines only."
            )

        prompt = f"""{SYSTEM_PROMPT}

SECTOR CONTEXT: This is a {sector_config.label} company.
{sector_config.sentiment_context}

Recent sentiment data for {ticker}:

{news_section}

{tweets_section}

Based on the above data and sector context, produce a TradingSignal with your
assessment of whether sentiment is bullish, bearish, or neutral for this stock.

Weight the sources according to their reliability — institutional news is your primary
signal, social media is supplementary. If both sources align, increase confidence.
If they conflict, favor the news signal and note the divergence in your reasoning.

Weight the content according to its relevance to the {sector_config.label} sector —
some items may be more material than others depending on the industry context.

Remember: only reference the data provided above. Do not add information from your
own knowledge about recent events — analyze strictly what is given.
"""

        signal = call_llm(prompt, response_model=TradingSignal)
        signal.agent_name = f"Sentiment Agent ({sector_config.label})"
        signal.ticker = ticker
        signals.append(signal)

    return {"signals": signals}
