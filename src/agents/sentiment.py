"""
Sentiment Agent — sector-aware news and social media sentiment analysis.

Reads recent news headlines and X/Twitter posts to assess the overall
sentiment landscape for each ticker, with sector-specific context for
interpreting what types of signals matter most.
"""

from src.state import AgentState, TradingSignal
from src.llm.models import call_llm
from src.agents.sector_prompts import get_sector_config


SYSTEM_PROMPT = """你是一位市场情绪分析师。你擅长解读新闻标题和社交媒体帖子来判断投资者情绪。

你的分析方法：
- 分析近期新闻标题的基调和内容
- 分析社交媒体帖子以获取实时市场情绪和散户投资者情绪
- 识别潜在催化剂（正面或负面）
- 评估情绪是否在转变（改善或恶化）
- 考虑来源的可靠性和重要性
- 权衡情绪对中短期价格走势的可能影响

来源权重（关键）：
- 机构新闻（Reuters, Bloomberg, WSJ等）比社交媒体权重更高。
  来自权威财经媒体的报道经过审核和事实核查。
- 社交媒体（X/Twitter）提供补充性的实时情绪信号。
  推文有助于判断散户投资者情绪和发现新兴叙事，
  但个别推文可能具有投机性、偏见或误导性。
- 当新闻和社交媒体情绪矛盾时，倾向于新闻信号。
- 当两者一致时，这会增强方向确认并提高置信度。
- 来自认证账号且粉丝数多的推文更可信。
- 高互动（点赞、转发）表明帖子引起广泛共鸣，但不代表准确。

关键规则：
- 只分析下方提供的标题和推文。不要编造或捏造数据。
- 如果既没有标题也没有推文，请明确说明无法进行情绪分析，并默认输出中性、低置信度的信号。
- 除非标题或推文明确提及，否则不要声称具体的财报结果、指引或公司行动。
- 做到精确：在构建论点时引用具体的标题或推文。
- 区分事实报道和观点/猜测。
- 标题来自真实新闻来源，推文来自真实的X/Twitter帖子——将它们视为真实数据。"""


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
                agent_name=f"情绪分析师（{sector_config.label}）",
                ticker=ticker,
                signal="neutral",
                confidence=0.15,
                reasoning=(
                    f"{ticker} 没有可用的近期新闻标题或社交媒体帖子。"
                    f"缺少情绪数据，无法进行有意义的分析。"
                    f"默认输出中性信号，置信度较低。"
                    f"数据缺失本身是中性的——并不意味着正面或负面情绪。"
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
                f"=== 机构新闻（{len(news)} 条标题） ===\n"
                f"[权重：主要来源——经过审核的新闻源]\n\n"
                f"{headlines_text}"
            )
        else:
            news_section = (
                "=== 机构新闻 ===\n"
                "没有可用的新闻标题。仅依据社交媒体信号分析，"
                "请降低置信度。"
            )

        # --- Build the tweets section ---
        if tweets:
            tweets_text = _format_tweets_section(tweets)
            tweets_section = (
                f"=== 社交媒体 / X 帖子（{len(tweets)} 条） ===\n"
                f"[权重：补充来源——实时情绪，可能具有投机性]\n\n"
                f"{tweets_text}"
            )
        else:
            tweets_section = (
                "=== 社交媒体 / X 帖子 ===\n"
                "没有可用的社交媒体帖子。仅基于新闻标题进行分析。"
            )

        prompt = f"""{SYSTEM_PROMPT}

行业背景：这是一家{sector_config.label}公司。
{sector_config.sentiment_context}

{ticker} 的近期情绪数据：

{news_section}

{tweets_section}

基于以上数据和行业背景，请输出 TradingSignal，评估该股票的情绪是
bullish（看多）、bearish（看空）还是 neutral（中性）。

按可靠性对来源进行权重分配——机构新闻是你的主要信号，社交媒体是补充。
如果两个来源一致，提高置信度。如果矛盾，倾向于新闻信号，并在推理中指出分歧。

按内容与{sector_config.label}行业的相关性进行权重分配——
根据行业背景，某些信息可能比其他信息更重要。

请记住：只引用上方提供的数据。不要添加你自己关于近期事件的知识——
严格分析给定的内容。请用中文撰写分析。
"""

        signal = call_llm(prompt, response_model=TradingSignal)
        signal.agent_name = f"情绪分析师（{sector_config.label}）"
        signal.ticker = ticker
        signals.append(signal)

    return {"signals": signals}
