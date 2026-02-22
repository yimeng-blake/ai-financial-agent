/* ==========================================================================
   AI Financial Agent — SSE Client & DOM Renderer
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {

    // -----------------------------------------------------------------------
    // Research document upload UI (index / form page only)
    // -----------------------------------------------------------------------

    const analysisForm = document.getElementById("analysis-form");
    if (analysisForm) {
        const tickersInput = document.getElementById("tickers");
        const addDocBtn    = document.getElementById("add-doc-btn");
        const docList      = document.getElementById("doc-list");

        function parseTickers() {
            return tickersInput.value
                .split(",")
                .map((t) => t.trim().toUpperCase())
                .filter((t) => t.length > 0);
        }

        function updateTickerDropdowns() {
            const tickers = parseTickers();
            addDocBtn.disabled = tickers.length === 0;

            document.querySelectorAll(".doc-ticker-select").forEach((select) => {
                const currentVal = select.value;
                select.innerHTML = '<option value="">选择股票...</option>';
                tickers.forEach((t) => {
                    const opt = document.createElement("option");
                    opt.value = t;
                    opt.textContent = t;
                    if (t === currentVal) opt.selected = true;
                    select.appendChild(opt);
                });
            });
        }

        tickersInput.addEventListener("input", updateTickerDropdowns);

        addDocBtn.addEventListener("click", () => {
            const tickers = parseTickers();
            if (tickers.length === 0) return;

            const row = document.createElement("div");
            row.className = "doc-row fade-in";

            // Ticker dropdown
            const select = document.createElement("select");
            select.name = "doc_tickers";
            select.className = "doc-ticker-select";
            select.required = true;
            select.innerHTML = '<option value="">选择代码...</option>';
            tickers.forEach((t) => {
                const opt = document.createElement("option");
                opt.value = t;
                opt.textContent = t;
                select.appendChild(opt);
            });

            // Styled file input
            const fileLabel = document.createElement("label");
            fileLabel.className = "doc-file-label";
            const fileName = document.createElement("span");
            fileName.className = "doc-file-name";
            fileName.textContent = "选择PDF文件...";
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.name = "doc_files";
            fileInput.accept = ".pdf";
            fileInput.required = true;
            fileInput.className = "doc-file-input";
            fileInput.addEventListener("change", () => {
                fileName.textContent = fileInput.files[0]
                    ? fileInput.files[0].name
                    : "选择PDF文件...";
            });
            fileLabel.appendChild(fileName);
            fileLabel.appendChild(fileInput);

            // Remove button
            const removeBtn = document.createElement("button");
            removeBtn.type = "button";
            removeBtn.className = "doc-remove-btn";
            removeBtn.textContent = "\u00d7";
            removeBtn.addEventListener("click", () => row.remove());

            row.appendChild(select);
            row.appendChild(fileLabel);
            row.appendChild(removeBtn);
            docList.appendChild(row);
        });
    }

    // -----------------------------------------------------------------------
    // Results page — SSE streaming & rendering
    // -----------------------------------------------------------------------

    const config = document.getElementById("sse-config");
    if (!config) return;  // Not on the results page.

    const tickers = config.dataset.tickers;
    const cash    = config.dataset.cash;
    const sid     = config.dataset.sid || "";

    const statusEl  = document.getElementById("progress-status");
    const errorEl   = document.getElementById("error-container");
    const bannerEl  = document.getElementById("complete-banner");

    const completedStages = new Set();
    let highestStage = 0;

    // Analysis data accumulators — populated during SSE, used for chat context.
    const analysisContext = {
        tickers: [],
        cash: 0,
        signals: [],
        risk_assessments: [],
        decisions: [],
        price_data: {},
    };

    // -----------------------------------------------------------------------
    // SSE connection
    // -----------------------------------------------------------------------

    let url = `/stream?tickers=${encodeURIComponent(tickers)}&cash=${encodeURIComponent(cash)}`;
    if (sid) {
        url += `&sid=${encodeURIComponent(sid)}`;
    }
    const source = new EventSource(url);

    source.addEventListener("start", (e) => {
        const data = JSON.parse(e.data);
        statusEl.textContent = `正在分析 ${data.tickers.join(", ")}...`;
        setStageActive(1);
        // Store for chat context
        analysisContext.tickers = data.tickers;
        analysisContext.cash = data.cash;
    });

    source.addEventListener("node_complete", (e) => {
        const data = JSON.parse(e.data);
        const stage = data.stage;

        // Update progress
        markStageComplete(stage);
        completedStages.add(stage);
        statusEl.textContent = `${data.label} 完成`;

        // Activate the next pending stage
        const nextStage = findNextPending();
        if (nextStage) setStageActive(nextStage);

        // Render data sections and accumulate for chat context
        if (data.signals) {
            renderSignals(data.signals);
            analysisContext.signals.push(...data.signals);
        }
        if (data.risk_assessments) {
            renderRiskAssessments(data.risk_assessments);
            analysisContext.risk_assessments.push(...data.risk_assessments);
        }
        if (data.decisions) {
            renderDecisions(data.decisions);
            analysisContext.decisions.push(...data.decisions);
        }
        if (data.market_data_loaded) {
            statusEl.textContent = `已加载 ${data.tickers_loaded.join(", ")} 的市场数据`;
        }
        if (data.price_data) {
            analysisContext.price_data = data.price_data;
            renderCharts(data.price_data);
        }
    });

    source.addEventListener("complete", () => {
        source.close();
        statusEl.textContent = "所有阶段已完成";
        bannerEl.classList.remove("hidden");
        for (let i = 1; i <= 6; i++) markStageComplete(i);
        // Show chat panel
        initChat();
    });

    source.addEventListener("error", (e) => {
        source.close();
        // Check if it's a custom error event with data
        if (e.data) {
            try {
                const data = JSON.parse(e.data);
                showError(data.error || "发生未知错误。");
            } catch {
                showError("连接中断，请重试。");
            }
        } else {
            showError("连接中断，请重试。");
        }
    });

    // -----------------------------------------------------------------------
    // Progress helpers
    // -----------------------------------------------------------------------

    function setStageActive(stage) {
        const el = document.getElementById(`stage-${stage}`);
        if (el && !el.classList.contains("complete")) {
            el.classList.add("active");
        }
        if (stage > highestStage) highestStage = stage;
    }

    function markStageComplete(stage) {
        const el = document.getElementById(`stage-${stage}`);
        if (!el) return;
        el.classList.remove("active");
        el.classList.add("complete");
        completedStages.add(stage);

        // Also mark the line before this stage as complete
        const lines = document.querySelectorAll(".progress-line");
        if (stage >= 2 && lines[stage - 2]) {
            lines[stage - 2].classList.add("complete");
        }
    }

    function findNextPending() {
        for (let i = 1; i <= 6; i++) {
            if (!completedStages.has(i)) return i;
        }
        return null;
    }

    function showError(msg) {
        errorEl.textContent = msg;
        errorEl.classList.remove("hidden");
        statusEl.textContent = "发生错误";
    }

    // -----------------------------------------------------------------------
    // Signal rendering
    // -----------------------------------------------------------------------

    function renderSignals(signals) {
        const section = document.getElementById("signals-section");
        const tbody   = document.getElementById("signals-body");
        section.classList.remove("hidden");

        signals.forEach((s) => {
            const signalClass = {
                bullish: "signal-bullish",
                bearish: "signal-bearish",
                neutral: "signal-neutral",
            }[s.signal] || "";

            const pct = Math.round(s.confidence * 100);
            const preview = s.reasoning.length > 120
                ? s.reasoning.substring(0, 120) + "..."
                : s.reasoning;

            const row = document.createElement("tr");
            row.classList.add("fade-in");
            row.innerHTML = `
                <td><span class="ticker-badge">${esc(s.ticker)}</span></td>
                <td>${esc(s.agent_name)}</td>
                <td><span class="signal-pill ${signalClass}">${esc(s.signal)}</span></td>
                <td>
                    <div class="confidence-bar">
                        <div class="confidence-track">
                            <div class="confidence-fill" style="width:${pct}%"></div>
                        </div>
                        <span class="confidence-value">${pct}%</span>
                    </div>
                </td>
                <td class="reasoning-cell">
                    <div class="reasoning-preview">${esc(preview)}</div>
                    <details>
                        <summary>查看完整分析</summary>
                        <div class="reasoning-full">${esc(s.reasoning)}</div>
                    </details>
                </td>
            `;
            tbody.appendChild(row);
        });
    }

    // -----------------------------------------------------------------------
    // Risk rendering
    // -----------------------------------------------------------------------

    function renderRiskAssessments(risks) {
        const section = document.getElementById("risk-section");
        const tbody   = document.getElementById("risk-body");
        section.classList.remove("hidden");

        risks.forEach((r) => {
            const pct = Math.round(r.risk_score * 100);
            const level = r.risk_score < 0.3 ? "low" : r.risk_score < 0.6 ? "medium" : "high";
            const maxPos = Math.round(r.max_position_size * 100);

            const factorPills = (r.risk_factors || [])
                .slice(0, 4)
                .map((f) => `<span class="risk-pill">${esc(f)}</span>`)
                .join("");

            const preview = r.reasoning.length > 120
                ? r.reasoning.substring(0, 120) + "..."
                : r.reasoning;

            const row = document.createElement("tr");
            row.classList.add("fade-in");
            row.innerHTML = `
                <td><span class="ticker-badge">${esc(r.ticker)}</span></td>
                <td>
                    <div class="risk-bar">
                        <div class="risk-track">
                            <div class="risk-fill risk-${level}" style="width:${pct}%"></div>
                        </div>
                        <span class="risk-value risk-${level}">${pct}%</span>
                    </div>
                </td>
                <td><strong>${maxPos}%</strong></td>
                <td><div class="risk-factors">${factorPills}</div></td>
                <td class="reasoning-cell">
                    <div class="reasoning-preview">${esc(preview)}</div>
                    <details>
                        <summary>查看完整分析</summary>
                        <div class="reasoning-full">${esc(r.reasoning)}</div>
                    </details>
                </td>
            `;
            tbody.appendChild(row);
        });
    }

    // -----------------------------------------------------------------------
    // Decision rendering
    // -----------------------------------------------------------------------

    function renderDecisions(decisions) {
        const section   = document.getElementById("decisions-section");
        const container = document.getElementById("decisions-cards");
        section.classList.remove("hidden");

        decisions.forEach((d) => {
            const actionClass = {
                buy:  "action-buy",
                sell: "action-sell",
                hold: "action-hold",
            }[d.action] || "";

            const pct = Math.round(d.confidence * 100);

            const card = document.createElement("div");
            card.className = "decision-card";
            card.innerHTML = `
                <div class="decision-card-header">
                    <span class="ticker-badge">${esc(d.ticker)}</span>
                    <span class="action-badge ${actionClass}">${esc(d.action)}</span>
                </div>
                <div class="decision-meta">
                    <div><span>数量 </span><strong>${d.quantity} 股</strong></div>
                    <div><span>置信度 </span><strong>${pct}%</strong></div>
                </div>
                <div class="decision-reasoning">${esc(d.reasoning)}</div>
            `;
            container.appendChild(card);
        });
    }

    // -----------------------------------------------------------------------
    // Chart rendering (Lightweight Charts by TradingView)
    // -----------------------------------------------------------------------

    const chartInstances = [];  // track for cleanup / resize

    function computeEMA(closes, span) {
        if (closes.length < span) return [];
        const k = 2 / (span + 1);
        const ema = [closes[0]];
        for (let i = 1; i < closes.length; i++) {
            ema.push(closes[i] * k + ema[i - 1] * (1 - k));
        }
        return ema;
    }

    function computeBollinger(closes, window, numStd) {
        if (closes.length < window) return { upper: [], lower: [] };
        const upper = [];
        const lower = [];
        for (let i = 0; i < closes.length; i++) {
            if (i < window - 1) {
                upper.push(null);
                lower.push(null);
                continue;
            }
            const slice = closes.slice(i - window + 1, i + 1);
            const mean = slice.reduce((a, b) => a + b, 0) / window;
            const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / window;
            const std = Math.sqrt(variance);
            upper.push(mean + numStd * std);
            lower.push(mean - numStd * std);
        }
        return { upper, lower };
    }

    function renderCharts(priceData) {
        const section = document.getElementById("charts-section");
        const grid = document.getElementById("charts-grid");
        if (!section || !grid) return;
        section.classList.remove("hidden");

        Object.entries(priceData).forEach(([ticker, prices]) => {
            if (!prices || prices.length === 0) return;

            // --- Chart card container ---
            const card = document.createElement("div");
            card.className = "chart-card fade-in";

            // Header with ticker, price, change
            const latest = prices[prices.length - 1];
            const first = prices[0];
            const changePct = ((latest.close - first.close) / first.close * 100).toFixed(2);
            const changeClass = changePct >= 0 ? "positive" : "negative";
            const changeSign = changePct >= 0 ? "+" : "";

            card.innerHTML = `
                <div class="chart-header">
                    <span class="ticker-badge">${esc(ticker)}</span>
                    <span class="chart-price">$${latest.close.toFixed(2)}</span>
                    <span class="chart-change ${changeClass}">${changeSign}${changePct}%</span>
                    <span class="chart-period">${prices.length} 天</span>
                </div>
                <div class="chart-legend">
                    <span class="legend-item"><span class="legend-swatch" style="background:#3b82f6"></span>EMA-9</span>
                    <span class="legend-item"><span class="legend-swatch" style="background:#f59e0b"></span>EMA-21</span>
                    <span class="legend-item"><span class="legend-swatch" style="background:rgba(139,143,163,0.3);border:1px solid #8b8fa3"></span>Bollinger</span>
                </div>
                <div class="chart-body" id="chart-body-${ticker}"></div>
            `;
            grid.appendChild(card);

            // --- Initialize Lightweight Chart ---
            const chartContainer = card.querySelector(`#chart-body-${ticker}`);

            const chart = LightweightCharts.createChart(chartContainer, {
                layout: {
                    background: { color: "#141720" },
                    textColor: "#8b8fa3",
                    fontSize: 11,
                },
                grid: {
                    vertLines: { color: "#1e2230" },
                    horzLines: { color: "#1e2230" },
                },
                crosshair: {
                    mode: LightweightCharts.CrosshairMode.Normal,
                    vertLine: { color: "#3b82f680", width: 1, style: 2 },
                    horzLine: { color: "#3b82f680", width: 1, style: 2 },
                },
                rightPriceScale: {
                    borderColor: "#262b3a",
                },
                timeScale: {
                    borderColor: "#262b3a",
                    timeVisible: false,
                },
            });

            // Compute indicator data
            const closes = prices.map(p => p.close);
            const ema9 = computeEMA(closes, 9);
            const ema21 = computeEMA(closes, 21);
            const bb = computeBollinger(closes, 20, 2);

            // Map dates for all series
            const dates = prices.map(p => p.date);

            // --- Bollinger Bands (render first so they appear behind) ---
            if (bb.upper.length > 0) {
                const bbUpperSeries = chart.addLineSeries({
                    color: "rgba(139,143,163,0.4)",
                    lineWidth: 1,
                    lineStyle: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false,
                });
                bbUpperSeries.setData(
                    bb.upper.map((v, i) => v !== null ? { time: dates[i], value: v } : null).filter(Boolean)
                );

                const bbLowerSeries = chart.addLineSeries({
                    color: "rgba(139,143,163,0.4)",
                    lineWidth: 1,
                    lineStyle: 2,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false,
                });
                bbLowerSeries.setData(
                    bb.lower.map((v, i) => v !== null ? { time: dates[i], value: v } : null).filter(Boolean)
                );
            }

            // --- Candlestick series ---
            const candleSeries = chart.addCandlestickSeries({
                upColor: "#22c55e",
                downColor: "#ef4444",
                borderDownColor: "#ef4444",
                borderUpColor: "#22c55e",
                wickDownColor: "#ef4444",
                wickUpColor: "#22c55e",
            });
            candleSeries.setData(
                prices.map(p => ({
                    time: p.date,
                    open: p.open,
                    high: p.high,
                    low: p.low,
                    close: p.close,
                }))
            );

            // --- EMA-9 ---
            if (ema9.length > 0) {
                const ema9Series = chart.addLineSeries({
                    color: "#3b82f6",
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false,
                });
                ema9Series.setData(
                    ema9.map((v, i) => ({ time: dates[i], value: v }))
                );
            }

            // --- EMA-21 ---
            if (ema21.length > 0) {
                const ema21Series = chart.addLineSeries({
                    color: "#f59e0b",
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false,
                });
                ema21Series.setData(
                    ema21.map((v, i) => ({ time: dates[i], value: v }))
                );
            }

            // --- Volume histogram ---
            const volumeSeries = chart.addHistogramSeries({
                priceFormat: { type: "volume" },
                priceScaleId: "volume",
            });
            chart.priceScale("volume").applyOptions({
                scaleMargins: { top: 0.85, bottom: 0 },
            });
            volumeSeries.setData(
                prices.map((p, i) => ({
                    time: p.date,
                    value: p.volume,
                    color: i === 0
                        ? "rgba(139,143,163,0.3)"
                        : p.close >= prices[i - 1].close
                            ? "rgba(34,197,94,0.3)"
                            : "rgba(239,68,68,0.3)",
                }))
            );

            // Fit content
            chart.timeScale().fitContent();

            // Responsive resize
            chartInstances.push({ chart, container: chartContainer });
        });

        // Global resize observer
        if (chartInstances.length > 0 && typeof ResizeObserver !== "undefined") {
            chartInstances.forEach(({ chart, container }) => {
                const ro = new ResizeObserver(() => {
                    chart.applyOptions({
                        width: container.clientWidth,
                        height: container.clientHeight,
                    });
                });
                ro.observe(container);
            });
        }
    }

    // -----------------------------------------------------------------------
    // Chat functionality
    // -----------------------------------------------------------------------

    const conversationHistory = [];  // {role, content}

    function initChat() {
        const chatSection = document.getElementById("chat-section");
        if (!chatSection) return;
        chatSection.classList.remove("hidden");

        setTimeout(() => {
            chatSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 400);

        const chatForm  = document.getElementById("chat-form");
        const chatInput = document.getElementById("chat-input");

        chatForm.addEventListener("submit", (e) => {
            e.preventDefault();
            const message = chatInput.value.trim();
            if (!message) return;
            chatInput.value = "";
            sendChatMessage(message);
        });
    }

    function sendChatMessage(message) {
        const messagesEl = document.getElementById("chat-messages");
        const sendBtn    = document.getElementById("chat-send-btn");
        const chatInput  = document.getElementById("chat-input");

        // Render user bubble
        appendChatMessage("user", message);
        conversationHistory.push({ role: "user", content: message });

        // Disable input while streaming
        sendBtn.disabled   = true;
        chatInput.disabled = true;

        // Create assistant message with typing indicator
        const assistantDiv = document.createElement("div");
        assistantDiv.className = "chat-message assistant-message fade-in";
        const contentDiv = document.createElement("div");
        contentDiv.className = "chat-message-content";
        contentDiv.innerHTML =
            '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        assistantDiv.appendChild(contentDiv);
        messagesEl.appendChild(assistantDiv);
        messagesEl.scrollTop = messagesEl.scrollHeight;

        fetchChatStream(message, contentDiv, sendBtn, chatInput, messagesEl);
    }

    async function fetchChatStream(message, contentDiv, sendBtn, chatInput, messagesEl) {
        let fullResponse = "";

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: message,
                    history: conversationHistory.slice(0, -1),
                    analysis_context: analysisContext,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const reader  = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer    = "";
            let firstToken = true;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Parse SSE events from buffer
                const lines = buffer.split("\n");
                buffer = lines.pop();  // keep incomplete line

                let eventType = "";
                for (const line of lines) {
                    if (line.startsWith("event:")) {
                        eventType = line.slice(6).trim();
                    } else if (line.startsWith("data:")) {
                        const dataStr = line.slice(5).trim();
                        if (!dataStr) continue;

                        try {
                            const data = JSON.parse(dataStr);

                            if (eventType === "token" && data.token) {
                                if (firstToken) {
                                    contentDiv.innerHTML = "";
                                    firstToken = false;
                                }
                                fullResponse += data.token;
                                contentDiv.innerHTML = renderMarkdown(fullResponse);
                                messagesEl.scrollTop = messagesEl.scrollHeight;
                            } else if (eventType === "error") {
                                contentDiv.textContent =
                                    "抱歉，发生了错误，请重试。";
                            }
                        } catch {
                            // ignore malformed JSON
                        }
                    }
                }
            }

            if (fullResponse) {
                conversationHistory.push({ role: "assistant", content: fullResponse });
            }
        } catch {
            contentDiv.textContent = "连接错误，请重试。";
        } finally {
            sendBtn.disabled   = false;
            chatInput.disabled = false;
            chatInput.focus();
        }
    }

    function appendChatMessage(role, content) {
        const messagesEl = document.getElementById("chat-messages");
        const div = document.createElement("div");
        div.className = `chat-message ${role}-message fade-in`;
        const contentDiv = document.createElement("div");
        contentDiv.className = "chat-message-content";
        contentDiv.textContent = content;
        div.appendChild(contentDiv);
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    function esc(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // -----------------------------------------------------------------------
    // Lightweight Markdown → HTML
    // -----------------------------------------------------------------------

    function renderMarkdown(text) {
        // Escape HTML first to prevent XSS
        let html = esc(text);

        // Code blocks: ```...```
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_m, _lang, code) => {
            return `<pre><code>${code.trim()}</code></pre>`;
        });

        // Inline code: `...`
        html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

        // Bold: **...**
        html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

        // Italic: *...*
        html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");

        // Headers: ### ... , ## ... , # ...
        html = html.replace(/^### (.+)$/gm, '<h4 class="md-h">$1</h4>');
        html = html.replace(/^## (.+)$/gm,  '<h3 class="md-h">$1</h3>');
        html = html.replace(/^# (.+)$/gm,   '<h3 class="md-h">$1</h3>');

        // Unordered lists: - item or * item
        html = html.replace(/^(?:- |\* )(.+)$/gm, '<li>$1</li>');
        html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

        // Ordered lists: 1. item
        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
        // Wrap consecutive <li> that aren't inside <ul> into <ol>
        html = html.replace(
            /(<li>.*<\/li>(?:\n<li>.*<\/li>)*)/g,
            (match) => {
                // Only wrap if not already inside a <ul>
                if (match.startsWith("<li>") && !html.includes("<ul>" + match)) {
                    return `<ol>${match}</ol>`;
                }
                return match;
            }
        );

        // Line breaks: double newline → paragraph break, single → <br>
        html = html.replace(/\n{2,}/g, "</p><p>");
        html = html.replace(/\n/g, "<br>");

        // Wrap in paragraph tags (avoid wrapping block elements)
        html = "<p>" + html + "</p>";

        // Clean up empty paragraphs and paragraphs wrapping block elements
        html = html.replace(/<p>\s*<\/p>/g, "");
        html = html.replace(/<p>(<(?:pre|ul|ol|h[34]|li))/g, "$1");
        html = html.replace(/(<\/(?:pre|ul|ol|h[34]|li)>)<\/p>/g, "$1");

        return html;
    }
});
