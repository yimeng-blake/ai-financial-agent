"""
Twitter/X data tools for fetching social media sentiment signals.

Uses the X API v2 recent search endpoint to find tweets about
stock tickers, filtered for credible accounts and meaningful engagement.
Requires X_BEARER_TOKEN environment variable.

If the token is not set or the API fails, all functions gracefully
return empty lists — the pipeline continues with news-only sentiment.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

X_API_BASE = "https://api.x.com/2"
DEFAULT_MIN_FOLLOWERS = 1000      # Minimum follower count for credibility
DEFAULT_MIN_ENGAGEMENT = 5        # Minimum likes + retweets
DEFAULT_MAX_TWEETS = 15           # Top N tweets per ticker after filtering
DEFAULT_API_MAX_RESULTS = 50      # Results per API request (10-100)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_bearer_token() -> Optional[str]:
    """Return the X bearer token from env, or None if not configured."""
    return os.getenv("X_BEARER_TOKEN")


def _build_query(ticker: str) -> str:
    """Build the X API v2 search query for a ticker.

    Uses both plain ticker and "$TICKER" as text to maximize coverage
    on Basic tier (the cashtag operator requires Academic/Enterprise).
    Filters: English only, no retweets, no replies.
    """
    # For very short tickers (1-2 chars), use only the cashtag form
    # to reduce false positives
    if len(ticker) <= 2:
        return f'"${ticker}" lang:en -is:retweet -is:reply'

    return f'({ticker} OR "${ticker}") lang:en -is:retweet -is:reply'


def _parse_tweet_response(response_json: dict) -> List[dict]:
    """Parse the X API v2 response into a flat list of tweet dicts
    with author metadata joined in.

    The X API v2 returns tweets in `data[]` and user objects separately
    in `includes.users[]`, linked by `author_id`. This function joins them.

    Returns raw parsed tweets (before credibility filtering).
    """
    tweets_data = response_json.get("data", [])
    if not tweets_data:
        return []

    # Build user lookup: user_id -> user object
    includes = response_json.get("includes", {})
    users_by_id: Dict[str, dict] = {}
    for user in includes.get("users", []):
        users_by_id[user["id"]] = user

    parsed = []
    for tweet in tweets_data:
        author_id = tweet.get("author_id", "")
        user = users_by_id.get(author_id, {})
        user_metrics = user.get("public_metrics", {})
        tweet_metrics = tweet.get("public_metrics", {})

        # Parse created_at: "2026-02-21T14:30:00.000Z"
        created_at = tweet.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_str = datetime.now().strftime("%Y-%m-%d")

        username = user.get("username", "unknown")

        parsed.append({
            "text": tweet.get("text", "").strip(),
            "date": date_str,
            "author": f"@{username}",
            "author_followers": user_metrics.get("followers_count", 0),
            "verified": user.get("verified", False),
            "likes": tweet_metrics.get("like_count", 0),
            "retweets": tweet_metrics.get("retweet_count", 0),
            "url": f"https://x.com/{username}/status/{tweet.get('id', '')}",
        })

    return parsed


def _filter_and_rank_tweets(
    tweets: List[dict],
    min_followers: int = DEFAULT_MIN_FOLLOWERS,
    min_engagement: int = DEFAULT_MIN_ENGAGEMENT,
    max_tweets: int = DEFAULT_MAX_TWEETS,
) -> List[dict]:
    """Filter tweets by credibility thresholds and return the top N
    sorted by engagement descending.

    Filtering criteria:
      - Author must have >= min_followers followers
      - Tweet must have >= min_engagement total engagement (likes + RTs)

    Sort order:
      - Verified accounts first
      - Then by total engagement (likes + retweets) descending
    """
    filtered = [
        t for t in tweets
        if t["author_followers"] >= min_followers
        and (t["likes"] + t["retweets"]) >= min_engagement
    ]

    # Sort: verified first, then by total engagement descending
    filtered.sort(
        key=lambda t: (t["verified"], t["likes"] + t["retweets"]),
        reverse=True,
    )

    return filtered[:max_tweets]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recent_tweets(
    ticker: str,
    min_followers: int = DEFAULT_MIN_FOLLOWERS,
    min_engagement: int = DEFAULT_MIN_ENGAGEMENT,
    max_tweets: int = DEFAULT_MAX_TWEETS,
) -> List[dict]:
    """Fetch and filter recent tweets about a ticker from X API v2.

    Returns a list of tweet dicts with keys:
        text, date, author, author_followers, verified, likes, retweets, url

    Returns an empty list if:
        - X_BEARER_TOKEN is not set (graceful degradation)
        - The API call fails for any reason
        - No tweets pass the credibility filters
    """
    bearer_token = _get_bearer_token()
    if not bearer_token:
        logger.info(
            f"X_BEARER_TOKEN not configured — skipping tweet fetch for {ticker}"
        )
        return []

    query = _build_query(ticker)

    params = {
        "query": query,
        "max_results": DEFAULT_API_MAX_RESULTS,
        "tweet.fields": "created_at,public_metrics,author_id",
        "expansions": "author_id",
        "user.fields": "verified,public_metrics,username",
    }
    headers = {
        "Authorization": f"Bearer {bearer_token}",
    }

    try:
        resp = requests.get(
            f"{X_API_BASE}/tweets/search/recent",
            params=params,
            headers=headers,
            timeout=10,
        )

        if resp.status_code == 401:
            logger.error("X API authentication failed — check X_BEARER_TOKEN")
            return []

        if resp.status_code == 429:
            logger.warning(f"X API rate limited for {ticker} — skipping tweets")
            return []

        if resp.status_code != 200:
            logger.warning(
                f"X API returned {resp.status_code} for {ticker}: "
                f"{resp.text[:200]}"
            )
            return []

        raw_tweets = _parse_tweet_response(resp.json())
        filtered = _filter_and_rank_tweets(
            raw_tweets,
            min_followers=min_followers,
            min_engagement=min_engagement,
            max_tweets=max_tweets,
        )

        logger.info(
            f"Fetched {len(raw_tweets)} tweets for {ticker}, "
            f"{len(filtered)} passed credibility filters"
        )
        return filtered

    except requests.RequestException as e:
        logger.error(f"X API request failed for {ticker}: {e}")
        return []
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing X API response for {ticker}: {e}")
        return []
