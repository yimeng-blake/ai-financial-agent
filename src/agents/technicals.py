"""
Technicals Agent — comprehensive price action and indicator analysis.

Computes a full suite of technical indicators from real price data:
trend (EMA/SMA, Bollinger Bands), momentum (RSI, MACD, Stochastic),
volatility (ATR, Bollinger width), volume analysis (OBV, accumulation/
distribution), and structure (support/resistance, Fibonacci retracement,
52-week positioning).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.state import AgentState, TradingSignal
from src.llm.models import call_llm


# ---------------------------------------------------------------------------
# Exponential Moving Average (core building block)
# ---------------------------------------------------------------------------

def _ema(data: List[float], span: int) -> List[float]:
    """Compute exponential moving average series."""
    if not data:
        return []
    multiplier = 2.0 / (span + 1)
    result = [data[0]]
    for val in data[1:]:
        result.append((val - result[-1]) * multiplier + result[-1])
    return result


def _compute_ema_value(prices: List[dict], span: int) -> Optional[float]:
    """Compute current EMA value from price history."""
    closes = [p["close"] for p in prices]
    if len(closes) < span:
        return None
    ema_series = _ema(closes, span)
    return ema_series[-1]


# ---------------------------------------------------------------------------
# Simple Moving Average
# ---------------------------------------------------------------------------

def _compute_sma(prices: List[dict], window: int) -> Optional[float]:
    """Compute simple moving average of close prices."""
    closes = [p["close"] for p in prices]
    if len(closes) < window:
        return None
    return sum(closes[-window:]) / window


# ---------------------------------------------------------------------------
# RSI — Wilder's smoothed (industry standard)
# ---------------------------------------------------------------------------

def _compute_rsi(prices: List[dict], period: int = 14) -> Optional[float]:
    """Compute RSI using Wilder's smoothing method (the standard).

    Wilder's smoothing uses: avg = prev_avg * (period-1)/period + current/period
    This matches what TradingView, Bloomberg, etc. display.
    """
    closes = [p["close"] for p in prices]
    if len(closes) < period + 1:
        return None

    # Calculate price changes
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Initial average gain/loss (simple average of first 'period' changes)
    gains = [max(d, 0) for d in deltas[:period]]
    losses = [max(-d, 0) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder's smoothing for remaining periods
    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# MACD (12, 26, 9)
# ---------------------------------------------------------------------------

def _compute_macd(prices: List[dict]) -> Optional[dict]:
    """Compute MACD line, signal line, and histogram."""
    closes = [p["close"] for p in prices]
    if len(closes) < 35:
        return None

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal_line = _ema(macd_line, 9)
    histogram = macd_line[-1] - signal_line[-1]

    # Detect crossover direction
    if len(macd_line) >= 2 and len(signal_line) >= 2:
        prev_diff = macd_line[-2] - signal_line[-2]
        curr_diff = macd_line[-1] - signal_line[-1]
        if prev_diff <= 0 and curr_diff > 0:
            crossover = "看涨交叉（MACD上穿信号线）"
        elif prev_diff >= 0 and curr_diff < 0:
            crossover = "看跌交叉（MACD下穿信号线）"
        else:
            crossover = "无近期交叉"
    else:
        crossover = "数据不足，无法检测交叉"

    return {
        "macd_line": round(macd_line[-1], 4),
        "signal_line": round(signal_line[-1], 4),
        "histogram": round(histogram, 4),
        "crossover": crossover,
    }


# ---------------------------------------------------------------------------
# Bollinger Bands (20-period, 2 std dev)
# ---------------------------------------------------------------------------

def _compute_bollinger_bands(
    prices: List[dict], window: int = 20, num_std: float = 2.0
) -> Optional[dict]:
    """Compute Bollinger Bands: middle (SMA), upper, lower, bandwidth, %B."""
    closes = [p["close"] for p in prices]
    if len(closes) < window:
        return None

    recent = closes[-window:]
    middle = sum(recent) / window
    std_dev = (sum((x - middle) ** 2 for x in recent) / window) ** 0.5
    upper = middle + num_std * std_dev
    lower = middle - num_std * std_dev

    current_price = closes[-1]
    bandwidth = (upper - lower) / middle if middle != 0 else 0

    # %B: where price sits within the bands (0 = lower, 1 = upper, >1 = above upper)
    pct_b = (current_price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5

    # Squeeze detection: bandwidth < 20-day average bandwidth
    if len(closes) >= window * 2:
        # Compute historical bandwidths
        bw_history = []
        for i in range(window, len(closes) + 1):
            w = closes[i - window:i]
            m = sum(w) / window
            s = (sum((x - m) ** 2 for x in w) / window) ** 0.5
            bw = (2 * num_std * s) / m if m != 0 else 0
            bw_history.append(bw)
        avg_bw = sum(bw_history) / len(bw_history) if bw_history else bandwidth
        squeeze = bandwidth < avg_bw * 0.75
    else:
        squeeze = False

    return {
        "upper": round(upper, 2),
        "middle": round(middle, 2),
        "lower": round(lower, 2),
        "bandwidth": round(bandwidth, 4),
        "pct_b": round(pct_b, 2),
        "squeeze": squeeze,
    }


# ---------------------------------------------------------------------------
# ATR (Average True Range) — volatility measurement
# ---------------------------------------------------------------------------

def _compute_atr(prices: List[dict], period: int = 14) -> Optional[dict]:
    """Compute Average True Range using Wilder's smoothing.

    Returns ATR value and ATR as a percentage of price.
    """
    if len(prices) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(prices)):
        high = prices[i]["high"]
        low = prices[i]["low"]
        prev_close = prices[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    # Wilder's smoothing
    atr = sum(true_ranges[:period]) / period
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period

    current_price = prices[-1]["close"]
    atr_pct = (atr / current_price) * 100 if current_price != 0 else 0

    return {
        "atr": round(atr, 2),
        "atr_pct": round(atr_pct, 2),
    }


# ---------------------------------------------------------------------------
# Stochastic Oscillator (%K, %D)
# ---------------------------------------------------------------------------

def _compute_stochastic(
    prices: List[dict], k_period: int = 14, d_period: int = 3
) -> Optional[dict]:
    """Compute Stochastic %K and %D."""
    if len(prices) < k_period + d_period:
        return None

    k_values = []
    for i in range(k_period - 1, len(prices)):
        window = prices[i - k_period + 1: i + 1]
        high = max(p["high"] for p in window)
        low = min(p["low"] for p in window)
        close = prices[i]["close"]
        if high - low == 0:
            k_values.append(50.0)
        else:
            k_values.append(((close - low) / (high - low)) * 100)

    # %D is the SMA of %K
    pct_k = k_values[-1]
    pct_d = sum(k_values[-d_period:]) / d_period if len(k_values) >= d_period else pct_k

    return {
        "pct_k": round(pct_k, 1),
        "pct_d": round(pct_d, 1),
    }


# ---------------------------------------------------------------------------
# On-Balance Volume (OBV) — volume-price correlation
# ---------------------------------------------------------------------------

def _compute_obv(prices: List[dict]) -> Optional[dict]:
    """Compute On-Balance Volume and its trend.

    OBV adds volume on up days and subtracts on down days.
    Rising OBV = accumulation, falling OBV = distribution.
    """
    if len(prices) < 20:
        return None

    obv = 0
    obv_series = [0]
    for i in range(1, len(prices)):
        if prices[i]["close"] > prices[i - 1]["close"]:
            obv += prices[i]["volume"]
        elif prices[i]["close"] < prices[i - 1]["close"]:
            obv -= prices[i]["volume"]
        obv_series.append(obv)

    # OBV trend over last 20 days
    obv_20 = obv_series[-20:]
    obv_slope = (obv_20[-1] - obv_20[0]) / 20 if len(obv_20) >= 20 else 0

    # Price trend over same period
    price_20 = [p["close"] for p in prices[-20:]]
    price_slope = (price_20[-1] - price_20[0]) / 20 if len(price_20) >= 20 else 0

    # Divergence detection
    if obv_slope > 0 and price_slope < 0:
        divergence = "看涨背离（OBV上升而价格下跌——吸筹）"
    elif obv_slope < 0 and price_slope > 0:
        divergence = "看跌背离（OBV下降而价格上涨——派发）"
    elif obv_slope > 0 and price_slope > 0:
        divergence = "确认上升趋势（价格和OBV同步上涨）"
    elif obv_slope < 0 and price_slope < 0:
        divergence = "确认下降趋势（价格和OBV同步下跌）"
    else:
        divergence = "无明显背离"

    return {
        "obv": obv,
        "trend": "rising" if obv_slope > 0 else "falling" if obv_slope < 0 else "flat",
        "divergence": divergence,
    }


# ---------------------------------------------------------------------------
# Volume analysis (enhanced)
# ---------------------------------------------------------------------------

def _analyze_volume(prices: List[dict], window: int = 20) -> Optional[dict]:
    """Enhanced volume analysis: trend, relative volume, up/down day correlation."""
    if len(prices) < window:
        return None

    volumes = [p["volume"] for p in prices]
    avg_volume = sum(volumes[-window:]) / window
    recent_avg = sum(volumes[-5:]) / 5
    relative_volume = recent_avg / avg_volume if avg_volume > 0 else 1.0

    # Volume on up days vs down days (last 20 days)
    up_day_volume = []
    down_day_volume = []
    for i in range(-window, 0):
        if prices[i]["close"] > prices[i - 1]["close"]:
            up_day_volume.append(prices[i]["volume"])
        elif prices[i]["close"] < prices[i - 1]["close"]:
            down_day_volume.append(prices[i]["volume"])

    avg_up_vol = sum(up_day_volume) / len(up_day_volume) if up_day_volume else 0
    avg_down_vol = sum(down_day_volume) / len(down_day_volume) if down_day_volume else 0

    if avg_up_vol > avg_down_vol * 1.3:
        volume_character = "吸筹（上涨日成交量更大）"
    elif avg_down_vol > avg_up_vol * 1.3:
        volume_character = "派发（下跌日成交量更大）"
    else:
        volume_character = "均衡（上涨日和下跌日成交量相近）"

    return {
        "avg_20d": int(avg_volume),
        "recent_5d_avg": int(recent_avg),
        "relative_volume": round(relative_volume, 2),
        "character": volume_character,
    }


# ---------------------------------------------------------------------------
# Support / Resistance detection
# ---------------------------------------------------------------------------

def _find_support_resistance(
    prices: List[dict], num_levels: int = 3
) -> Optional[dict]:
    """Identify key support and resistance levels from price history.

    Uses pivot point clustering: finds local highs/lows and clusters
    nearby levels together.
    """
    if len(prices) < 20:
        return None

    closes = [p["close"] for p in prices]
    highs = [p["high"] for p in prices]
    lows = [p["low"] for p in prices]
    current = closes[-1]

    # Find local maxima and minima (using 5-bar lookback/forward)
    pivot_highs = []
    pivot_lows = []
    lookback = 5

    for i in range(lookback, len(prices) - lookback):
        # Local high: highest high in the window
        if highs[i] == max(highs[i - lookback: i + lookback + 1]):
            pivot_highs.append(highs[i])
        # Local low: lowest low in the window
        if lows[i] == min(lows[i - lookback: i + lookback + 1]):
            pivot_lows.append(lows[i])

    # Cluster nearby levels (within 1.5% of each other)
    def cluster_levels(levels: List[float], threshold: float = 0.015) -> List[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        clusters.append(sum(current_cluster) / len(current_cluster))
        return clusters

    resistance_levels = [
        r for r in cluster_levels(pivot_highs) if r > current
    ][:num_levels]

    support_levels = [
        s for s in reversed(cluster_levels(pivot_lows)) if s < current
    ][:num_levels]

    # 52-week high/low from available data
    all_highs = max(highs) if highs else None
    all_lows = min(lows) if lows else None
    pct_from_high = ((current - all_highs) / all_highs * 100) if all_highs else None
    pct_from_low = ((current - all_lows) / all_lows * 100) if all_lows else None

    return {
        "resistance": [round(r, 2) for r in resistance_levels],
        "support": [round(s, 2) for s in support_levels],
        "period_high": round(all_highs, 2) if all_highs else None,
        "period_low": round(all_lows, 2) if all_lows else None,
        "pct_from_high": round(pct_from_high, 1) if pct_from_high is not None else None,
        "pct_from_low": round(pct_from_low, 1) if pct_from_low is not None else None,
    }


# ---------------------------------------------------------------------------
# Fibonacci Retracement
# ---------------------------------------------------------------------------

def _compute_fibonacci(prices: List[dict]) -> Optional[dict]:
    """Compute Fibonacci retracement levels from recent swing high/low.

    Uses the period high and low to calculate standard Fibonacci levels.
    """
    if len(prices) < 20:
        return None

    highs = [p["high"] for p in prices]
    lows = [p["low"] for p in prices]
    swing_high = max(highs)
    swing_low = min(lows)
    current = prices[-1]["close"]

    if swing_high == swing_low:
        return None

    diff = swing_high - swing_low
    levels = {
        "0.0% (high)": round(swing_high, 2),
        "23.6%": round(swing_high - 0.236 * diff, 2),
        "38.2%": round(swing_high - 0.382 * diff, 2),
        "50.0%": round(swing_high - 0.500 * diff, 2),
        "61.8%": round(swing_high - 0.618 * diff, 2),
        "78.6%": round(swing_high - 0.786 * diff, 2),
        "100.0% (low)": round(swing_low, 2),
    }

    # Find nearest Fibonacci level
    nearest_level = None
    nearest_dist = float("inf")
    for label, level in levels.items():
        dist = abs(current - level)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_level = label

    return {
        "levels": levels,
        "nearest_level": nearest_level,
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
    }


# ---------------------------------------------------------------------------
# Trend strength assessment
# ---------------------------------------------------------------------------

def _assess_trend(prices: List[dict]) -> Optional[dict]:
    """Assess overall trend direction and strength using multiple timeframes."""
    closes = [p["close"] for p in prices]
    if len(closes) < 50:
        return None

    # Short-term trend (5 days)
    short_change = (closes[-1] - closes[-5]) / closes[-5]

    # Medium-term trend (20 days)
    med_change = (closes[-1] - closes[-20]) / closes[-20]

    # Long-term trend (50 days)
    long_change = (closes[-1] - closes[-50]) / closes[-50]

    # EMA alignment check
    ema_9 = _ema(closes, 9)[-1]
    ema_21 = _ema(closes, 21)[-1]
    sma_50 = sum(closes[-50:]) / 50

    if ema_9 > ema_21 > sma_50:
        alignment = "看涨排列（EMA-9 > EMA-21 > SMA-50）"
    elif ema_9 < ema_21 < sma_50:
        alignment = "看跌排列（EMA-9 < EMA-21 < SMA-50）"
    else:
        alignment = "混合排列（无明确趋势层级）"

    # Count up days vs down days (last 20)
    up_days = sum(1 for i in range(-20, 0) if closes[i] > closes[i - 1])
    down_days = 20 - up_days

    return {
        "short_5d": round(short_change, 4),
        "medium_20d": round(med_change, 4),
        "long_50d": round(long_change, 4),
        "ma_alignment": alignment,
        "up_days_20": up_days,
        "down_days_20": down_days,
    }


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一位专业的技术分析师。你通过分析价格走势、动量、波动率、成交量形态和市场结构来生成交易信号。

你的分析框架（按重要性排序）：

1. 趋势结构：均线排列、多时间周期趋势方向。
   顺势而为——首先确定主要趋势方向。

2. 动量：RSI（超买>70 / 超卖<30）、MACD交叉、
   Stochastic随机指标。观察动量是否确认或背离趋势。

3. 波动率：Bollinger Bands（缩口=即将突破，%B判断带内位置），
   ATR（平均真实波幅——用于止损设置和风险评估）。

4. 成交量：OBV背离是有力的领先指标。
   吸筹（上涨日放量）vs. 派发（下跌日放量）。
   成交量确认价格走势——低量突破值得怀疑。

5. 结构：支撑/阻力位、Fibonacci回撤区间。
   价格往往尊重这些关键位——观察在关键位的反应。

信号质量检查清单：
- 强信号需要多个指标确认同一方向
- 背离（如价格创新高但RSI下降）是警告信号
- Bollinger Band缩口往往预示大幅波动——警惕方向选择
- 成交量应确认价格走势（突破+放量=确认）

关键规则：
- 只引用下方提供的技术数据。不要编造指标。
- 如果数据标注为"N/A"或不足，请承认这一局限。
- 做到精确：引用具体的指标数值、价位和百分比。
- 不要引用新闻、财报或基本面数据——你是纯粹的技术分析师。
- 诚实陈述置信度——混合信号应降低置信度。"""


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

def technicals_agent(state: AgentState) -> AgentState:
    """Analyze each ticker's technical indicators and produce signals."""
    signals = []

    for ticker in state["tickers"]:
        data = state["market_data"].get(ticker, {})
        prices = data.get("prices", [])

        if not prices:
            signals.append(TradingSignal(
                agent_name="技术面分析师",
                ticker=ticker,
                signal="neutral",
                confidence=0.1,
                reasoning=(
                    f"{ticker} 没有可用的价格数据。"
                    f"缺少历史价格，无法进行技术分析。"
                    f"默认输出中性信号，置信度极低。"
                ),
            ))
            continue

        current_price = prices[-1]["close"]
        date_range = f"{prices[0]['date']} to {prices[-1]['date']}"

        # --- Compute all indicators ---
        ema_9 = _compute_ema_value(prices, 9)
        ema_21 = _compute_ema_value(prices, 21)
        sma_50 = _compute_sma(prices, 50)
        rsi = _compute_rsi(prices)
        macd = _compute_macd(prices)
        stochastic = _compute_stochastic(prices)
        bollinger = _compute_bollinger_bands(prices)
        atr = _compute_atr(prices)
        obv = _compute_obv(prices)
        volume = _analyze_volume(prices)
        sr = _find_support_resistance(prices)
        fib = _compute_fibonacci(prices)
        trend = _assess_trend(prices)

        # --- Build the prompt sections ---

        # Trend section
        trend_text = "N/A — 数据不足"
        if trend:
            trend_text = f"""5日: {trend['short_5d']:+.2%} | 20日: {trend['medium_20d']:+.2%} | 50日: {trend['long_50d']:+.2%}
  均线排列: {trend['ma_alignment']}
  近20日: {trend['up_days_20']} 涨, {trend['down_days_20']} 跌"""

        # Moving averages section
        ma_lines = []
        if ema_9:
            pos = "上方" if current_price > ema_9 else "下方"
            ma_lines.append(f"EMA-9:  ${ema_9:.2f}（价格在其{pos}）")
        if ema_21:
            pos = "上方" if current_price > ema_21 else "下方"
            ma_lines.append(f"EMA-21: ${ema_21:.2f}（价格在其{pos}）")
        if sma_50:
            pos = "上方" if current_price > sma_50 else "下方"
            ma_lines.append(f"SMA-50: ${sma_50:.2f}（价格在其{pos}）")
        if ema_9 and ema_21:
            if ema_9 > ema_21:
                ma_lines.append("EMA-9/21 状态：看涨（短期EMA在长期上方）")
            else:
                ma_lines.append("EMA-9/21 状态：看跌（短期EMA在长期下方）")
        ma_text = "\n  ".join(ma_lines) if ma_lines else "N/A"

        # RSI section
        rsi_text = "N/A"
        if rsi is not None:
            if rsi > 70:
                zone = "超买"
            elif rsi > 60:
                zone = "偏多区域"
            elif rsi < 30:
                zone = "超卖"
            elif rsi < 40:
                zone = "偏空区域"
            else:
                zone = "中性区域"
            rsi_text = f"{rsi:.1f} ({zone})"

        # MACD section
        macd_text = "N/A"
        if macd:
            macd_text = (
                f"MACD Line: {macd['macd_line']}, Signal: {macd['signal_line']}, "
                f"Histogram: {macd['histogram']}\n  "
                f"Status: {macd['crossover']}"
            )

        # Stochastic section
        stoch_text = "N/A"
        if stochastic:
            k, d = stochastic["pct_k"], stochastic["pct_d"]
            if k > 80:
                zone = "超买"
            elif k < 20:
                zone = "超卖"
            else:
                zone = "中性"
            stoch_text = f"%K: {k:.1f}, %D: {d:.1f} ({zone})"

        # Bollinger Bands section
        bb_text = "N/A"
        if bollinger:
            squeeze_tag = " ** 检测到布林带挤压——预期波动率扩张 **" if bollinger["squeeze"] else ""
            bb_text = (
                f"Upper: ${bollinger['upper']}, Middle: ${bollinger['middle']}, Lower: ${bollinger['lower']}\n  "
                f"Bandwidth: {bollinger['bandwidth']:.4f}, %B: {bollinger['pct_b']:.2f} "
                f"(0=lower band, 1=upper band){squeeze_tag}"
            )

        # ATR section
        atr_text = "N/A"
        if atr:
            atr_text = f"${atr['atr']} ({atr['atr_pct']:.2f}% of price)"

        # Volume section
        vol_text = "N/A"
        if volume:
            vol_text = (
                f"20-day avg: {volume['avg_20d']:,} | Recent 5-day avg: {volume['recent_5d_avg']:,}\n  "
                f"Relative volume: {volume['relative_volume']:.2f}x average\n  "
                f"Character: {volume['character']}"
            )

        # OBV section
        obv_text = "N/A"
        if obv:
            obv_text = (
                f"Trend: {obv['trend']}\n  "
                f"{obv['divergence']}"
            )

        # Support/Resistance section
        sr_text = "N/A"
        if sr:
            res = ", ".join(f"${r}" for r in sr["resistance"]) or "none identified"
            sup = ", ".join(f"${s}" for s in sr["support"]) or "none identified"
            sr_text = (
                f"Resistance: {res}\n  "
                f"Support: {sup}\n  "
                f"Period high: ${sr['period_high']} ({sr['pct_from_high']:+.1f}% from current)\n  "
                f"Period low: ${sr['period_low']} ({sr['pct_from_low']:+.1f}% from current)"
            )

        # Fibonacci section
        fib_text = "N/A"
        if fib:
            level_lines = ", ".join(
                f"{label}: ${val}" for label, val in fib["levels"].items()
            )
            fib_text = (
                f"Swing range: ${fib['swing_low']} — ${fib['swing_high']}\n  "
                f"Levels: {level_lines}\n  "
                f"Price nearest to: {fib['nearest_level']}"
            )

        prompt = f"""{SYSTEM_PROMPT}

{ticker} 的技术数据（{len(prices)} 个交易日，{date_range}）：

当前价格: ${current_price:.2f}

1. 趋势:
  {trend_text}

2. 均线:
  {ma_text}

3. RSI (14, Wilder): {rsi_text}

4. MACD (12, 26, 9):
  {macd_text}

5. STOCHASTIC (14, 3):
  {stoch_text}

6. BOLLINGER BANDS (20, 2σ):
  {bb_text}

7. ATR (14): {atr_text}

8. 成交量:
  {vol_text}

9. OBV（能量潮）:
  {obv_text}

10. 支撑 / 阻力:
  {sr_text}

11. FIBONACCI 回撤:
  {fib_text}

请输出 TradingSignal。综合以上所有指标——
寻找趋势、动量、波动率和成交量之间的确认和背离信号。
在推理中引用具体数值。请用中文撰写分析。
"""

        signal = call_llm(prompt, response_model=TradingSignal)
        signal.agent_name = "技术面分析师"
        signal.ticker = ticker
        signals.append(signal)

    return {"signals": signals}
