/* ═══════════════════════════════════════════════════════
   Chemelex Inventory Intelligence — Frontend Logic
   ═══════════════════════════════════════════════════════ */

// ── State ────────────────────────────────────────────────

let currentConvoId = null;
let isQuerying = false;
let cachedKpis = null;

// ── Init ─────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    loadKPIs();
    loadConversations();
    setupInput();
});

// ── Tab Switching ────────────────────────────────────────

function switchTab(tab) {
    document.getElementById('tabDashboard').classList.toggle('active', tab === 'dashboard');
    document.getElementById('tabChat').classList.toggle('active', tab === 'chat');
    document.getElementById('viewDashboard').classList.toggle('hidden', tab !== 'dashboard');
    document.getElementById('viewChat').classList.toggle('hidden', tab !== 'chat');
    document.getElementById('sidebar').classList.toggle('hidden', tab !== 'chat');

    if (tab === 'dashboard') {
        // Re-render charts when switching back (Plotly needs visible container)
        setTimeout(() => {
            if (cachedKpis) renderKPICharts(cachedKpis);
        }, 50);
    }

    if (tab === 'chat') {
        setTimeout(() => {
            document.getElementById('chatInput').focus();
            scrollToBottom();
        }, 50);
    }
}

// ── Input Setup ──────────────────────────────────────────

function setupInput() {
    const input = document.getElementById('chatInput');
    const btn = document.getElementById('sendBtn');

    input.addEventListener('input', () => {
        btn.disabled = !input.value.trim() || isQuerying;
    });

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (input.value.trim() && !isQuerying) {
                document.getElementById('chatForm').dispatchEvent(new Event('submit'));
            }
        }
    });
}

// ── KPIs ─────────────────────────────────────────────────

async function loadKPIs() {
    try {
        const resp = await fetch('/api/kpis');
        if (!resp.ok) throw new Error('Failed to load KPIs');
        const kpis = await resp.json();
        cachedKpis = kpis;
        renderKPIs(kpis);
        renderKPIStrip(kpis);
        renderKPICharts(kpis);
    } catch (err) {
        console.error('KPI load error:', err);
    }
}

function formatCurrency(val) {
    if (val >= 1e9) return '$' + (val / 1e9).toFixed(2) + 'B';
    if (val >= 1e6) return '$' + (val / 1e6).toFixed(1) + 'M';
    if (val >= 1e3) return '$' + (val / 1e3).toFixed(0) + 'K';
    return '$' + val.toFixed(0);
}

function formatNumber(val) {
    return new Intl.NumberFormat().format(val);
}

function renderKPIs(kpis) {
    const grid = document.getElementById('kpiGrid');
    grid.innerHTML = `
        <div class="kpi-card navy">
            <div class="kpi-icon">&#x1F4B0;</div>
            <div class="kpi-value">${formatCurrency(kpis.total_inventory_usd)}</div>
            <div class="kpi-label">Total Inventory Value</div>
            <div class="kpi-sub">Shelf + GIT + WIP</div>
        </div>
        <div class="kpi-card blue">
            <div class="kpi-icon">&#x1F3ED;</div>
            <div class="kpi-value">${kpis.active_plants} <span style="font-size:14px;font-weight:400;color:#9ca3af">/ ${kpis.total_plants}</span></div>
            <div class="kpi-label">Active Plants</div>
            <div class="kpi-sub">Plants with stock &gt; 0</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-icon">&#x1F4E6;</div>
            <div class="kpi-value">${formatNumber(kpis.plant_materials_with_stock)}</div>
            <div class="kpi-label">Stocked (Plant x SKU)</div>
            <div class="kpi-sub">${formatNumber(kpis.unique_materials_with_stock)} unique SKUs across plants</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-icon">&#x1F4C5;</div>
            <div class="kpi-value">${kpis.avg_doh} <span style="font-size:14px;font-weight:400;color:#9ca3af">days</span></div>
            <div class="kpi-label">Avg Days on Hand</div>
            <div class="kpi-sub">For stocked items</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-icon">&#x26A0;&#xFE0F;</div>
            <div class="kpi-value">${formatNumber(kpis.below_safety_stock)}</div>
            <div class="kpi-label">Below Safety Stock</div>
            <div class="kpi-sub">Across ${kpis.plants_with_shortages} plants</div>
        </div>
    `;
}

// Compact KPI strip for chat view
function renderKPIStrip(kpis) {
    const strip = document.getElementById('kpiStrip');
    strip.innerHTML = `
        <div class="kpi-chip">
            <span class="chip-value">${formatCurrency(kpis.total_inventory_usd)}</span>
            <span class="chip-label">Total Inventory</span>
        </div>
        <div class="kpi-chip">
            <span class="chip-value">${kpis.active_plants}/${kpis.total_plants}</span>
            <span class="chip-label">Active Plants</span>
        </div>
        <div class="kpi-chip">
            <span class="chip-value">${formatNumber(kpis.plant_materials_with_stock)}</span>
            <span class="chip-label">Stocked Items</span>
        </div>
        <div class="kpi-chip">
            <span class="chip-value">${kpis.avg_doh}d</span>
            <span class="chip-label">Avg DOH</span>
        </div>
        <div class="kpi-chip" style="border-color:#fecaca;">
            <span class="chip-value" style="color:#dc2626;">${formatNumber(kpis.below_safety_stock)}</span>
            <span class="chip-label">Below Safety Stock</span>
        </div>
    `;
}

function renderKPICharts(kpis) {
    const chartLayout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { family: 'Inter' },
    };

    // Top Plants Chart
    if (kpis.top_plants && kpis.top_plants.length > 0) {
        const plants = kpis.top_plants;
        Plotly.newPlot('chartTopPlants', [{
            x: plants.map(p => 'Plant ' + p.plant),
            y: plants.map(p => p.value),
            type: 'bar',
            marker: { color: ['#004976', '#0a5c8f', '#1470a8', '#3b82f6', '#60a5fa'] },
            hovertemplate: '%{x}<br>$%{y:,.0f}<extra></extra>',
        }], {
            ...chartLayout,
            margin: { t: 8, r: 16, b: 36, l: 56 },
            xaxis: { tickfont: { size: 10, family: 'Inter' } },
            yaxis: { tickfont: { size: 10, family: 'Inter' }, tickformat: '$,.0s' },
        }, { responsive: true, displayModeBar: false });
    }

    // Material Type Chart
    if (kpis.by_material_type && kpis.by_material_type.length > 0) {
        const mt = kpis.by_material_type;
        Plotly.newPlot('chartMaterialType', [{
            labels: mt.map(m => m.type),
            values: mt.map(m => m.value),
            type: 'pie',
            hole: 0.45,
            marker: { colors: ['#004976', '#0a5c8f', '#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe'] },
            textinfo: 'percent',
            textfont: { size: 11, family: 'Inter' },
            hovertemplate: '%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>',
        }], {
            ...chartLayout,
            margin: { t: 8, r: 8, b: 8, l: 8 },
            showlegend: true,
            legend: { font: { size: 9, family: 'Inter' }, orientation: 'h', y: -0.15 },
        }, { responsive: true, displayModeBar: false });
    }

    // Below Safety Stock by Plant
    if (kpis.below_ss_by_plant && kpis.below_ss_by_plant.length > 0) {
        const ss = kpis.below_ss_by_plant;
        Plotly.newPlot('chartBelowSS', [{
            x: ss.map(p => 'Plant ' + p.plant),
            y: ss.map(p => p.count),
            type: 'bar',
            marker: {
                color: ss.map((_, i) => {
                    const c = ['#dc2626', '#ef4444', '#f87171', '#fca5a5', '#fecaca', '#fee2e2', '#fef2f2', '#fff5f5'];
                    return c[i] || '#fecaca';
                }),
            },
            hovertemplate: '%{x}<br>%{y} items below safety stock<extra></extra>',
        }], {
            ...chartLayout,
            margin: { t: 8, r: 16, b: 36, l: 46 },
            xaxis: { tickfont: { size: 10, family: 'Inter' }, tickangle: -45 },
            yaxis: { tickfont: { size: 10, family: 'Inter' }, title: { text: 'SKUs at risk', font: { size: 10 } } },
        }, { responsive: true, displayModeBar: false });
    }

    // Inventory Composition
    if (kpis.shelf_stock_usd || kpis.git_usd || kpis.wip_usd) {
        Plotly.newPlot('chartComposition', [{
            labels: ['Shelf Stock', 'Goods In Transit', 'Work In Progress'],
            values: [kpis.shelf_stock_usd, kpis.git_usd, kpis.wip_usd],
            type: 'pie',
            hole: 0.5,
            marker: { colors: ['#004976', '#3b82f6', '#60a5fa'] },
            textinfo: 'label+percent',
            textfont: { size: 11, family: 'Inter' },
            hovertemplate: '%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>',
        }], {
            ...chartLayout,
            margin: { t: 8, r: 8, b: 8, l: 8 },
            showlegend: false,
        }, { responsive: true, displayModeBar: false });
    }
}

// ── Conversations ────────────────────────────────────────

async function loadConversations() {
    try {
        const resp = await fetch('/api/conversations');
        if (!resp.ok) throw new Error('Failed');
        const convos = await resp.json();
        renderConversations(convos);
    } catch (err) {
        console.error('Conversation load error:', err);
    }
}

function renderConversations(convos) {
    const list = document.getElementById('conversationList');
    if (convos.length === 0) {
        list.innerHTML = '<div style="padding:20px;text-align:center;color:#9ca3af;font-size:13px;">No conversations yet.<br>Start asking questions!</div>';
        return;
    }

    list.innerHTML = convos.map(c => `
        <div class="convo-item ${c._id === currentConvoId ? 'active' : ''}"
             onclick="loadConversation('${c._id}')">
            <span class="convo-title">${escapeHtml(c.title)}</span>
            <button class="convo-delete" onclick="event.stopPropagation();deleteConversation('${c._id}')" title="Delete">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
            </button>
        </div>
    `).join('');
}

async function newConversation() {
    try {
        const resp = await fetch('/api/conversations', { method: 'POST' });
        if (!resp.ok) throw new Error('Failed');
        const convo = await resp.json();
        currentConvoId = convo._id;
        clearChat();
        loadConversations();
        document.getElementById('quickQuestions').style.display = 'block';
    } catch (err) {
        console.error('New conversation error:', err);
    }
}

async function loadConversation(id) {
    // Switch to chat tab
    switchTab('chat');

    try {
        currentConvoId = id;
        const resp = await fetch(`/api/conversations/${id}`);
        if (!resp.ok) throw new Error('Failed');
        const convo = await resp.json();

        clearChat();
        const msgs = convo.messages || [];
        for (const msg of msgs) {
            appendMessage(msg.role, msg.content, msg.sql, null, false);
        }

        document.getElementById('quickQuestions').style.display = msgs.length > 0 ? 'none' : 'block';
        loadConversations();
        scrollToBottom();
    } catch (err) {
        console.error('Load conversation error:', err);
    }
}

async function deleteConversation(id) {
    try {
        await fetch(`/api/conversations/${id}`, { method: 'DELETE' });
        if (currentConvoId === id) {
            currentConvoId = null;
            clearChat();
            document.getElementById('quickQuestions').style.display = 'block';
        }
        loadConversations();
    } catch (err) {
        console.error('Delete error:', err);
    }
}

// ── Chat ─────────────────────────────────────────────────

function clearChat() {
    document.getElementById('chatMessages').innerHTML = '';
}

function scrollToBottom() {
    const el = document.getElementById('chatMessages');
    el.scrollTop = el.scrollHeight;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function appendMessage(role, content, sql, tableData, animate = true) {
    const container = document.getElementById('chatMessages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `msg ${role}`;

    if (role === 'user') {
        msgDiv.innerHTML = `
            <div class="msg-label">You</div>
            <div class="msg-bubble">${escapeHtml(content)}</div>
        `;
    } else {
        let htmlContent;
        try {
            htmlContent = marked.parse(content);
        } catch {
            htmlContent = content.replace(/\n/g, '<br>');
        }

        let sqlHtml = '';
        if (sql) {
            const sqlId = 'sql-' + Date.now() + Math.random().toString(36).slice(2, 6);
            sqlHtml = `
                <div class="sql-toggle">
                    <button class="sql-toggle-btn" onclick="document.getElementById('${sqlId}').classList.toggle('show')">
                        &#x1F50D; View SQL Query
                    </button>
                    <div class="sql-code" id="${sqlId}">
                        <pre><code>${escapeHtml(sql)}</code></pre>
                    </div>
                </div>
            `;
        }

        let chartHtml = '';
        if (tableData && tableData.columns && tableData.rows && tableData.rows.length > 0) {
            const chartId = 'chart-' + Date.now() + Math.random().toString(36).slice(2, 6);
            chartHtml = `<div class="msg-chart" id="${chartId}"></div>`;
            setTimeout(() => renderResultChart(chartId, tableData), 100);
        }

        msgDiv.innerHTML = `
            <div class="msg-label">Chemelex AI</div>
            <div class="msg-bubble">${htmlContent}</div>
            ${sqlHtml}
            ${chartHtml}
        `;
    }

    if (!animate) msgDiv.style.animation = 'none';
    container.appendChild(msgDiv);
    scrollToBottom();
}

function showLoading() {
    const container = document.getElementById('chatMessages');
    const loadDiv = document.createElement('div');
    loadDiv.className = 'msg assistant';
    loadDiv.id = 'loadingMsg';
    loadDiv.innerHTML = `
        <div class="msg-label">Chemelex AI</div>
        <div class="msg-bubble msg-loading">
            <div class="typing-dots">
                <span></span><span></span><span></span>
            </div>
            <span class="typing-text">Analyzing your question...</span>
        </div>
    `;
    container.appendChild(loadDiv);
    scrollToBottom();
}

function removeLoading() {
    const el = document.getElementById('loadingMsg');
    if (el) el.remove();
}

function renderResultChart(containerId, tableData) {
    const el = document.getElementById(containerId);
    if (!el) return;

    const cols = tableData.columns;
    const rows = tableData.rows;

    const numCols = [];
    const catCols = [];
    for (let i = 0; i < cols.length; i++) {
        if (typeof rows[0][i] === 'number') numCols.push(i);
        else catCols.push(i);
    }

    if (numCols.length < 1 || catCols.length < 1 || rows.length < 2) {
        el.remove();
        return;
    }

    Plotly.newPlot(el, [{
        x: rows.map(r => r[catCols[0]]),
        y: rows.map(r => r[numCols[0]]),
        type: rows.length <= 20 ? 'bar' : 'scatter',
        mode: 'lines+markers',
        marker: { color: '#004976' },
        hovertemplate: '%{x}<br>%{y:,.2f}<extra></extra>',
    }], {
        margin: { t: 10, r: 16, b: 40, l: 60 },
        xaxis: { tickfont: { size: 10, family: 'Inter' }, tickangle: -45 },
        yaxis: { tickfont: { size: 10, family: 'Inter' } },
        paper_bgcolor: 'transparent',
        plot_bgcolor: '#f9fafb',
        font: { family: 'Inter' },
        height: 250,
    }, { responsive: true, displayModeBar: false });
}

// ── Submit Query ─────────────────────────────────────────

async function handleSubmit(e) {
    e.preventDefault();
    const input = document.getElementById('chatInput');
    const question = input.value.trim();
    if (!question || isQuerying) return;
    await askQuestion(question);
}

async function askQuestion(question) {
    if (isQuerying) return;

    // Switch to chat tab
    switchTab('chat');

    const input = document.getElementById('chatInput');
    input.value = '';
    document.getElementById('sendBtn').disabled = true;

    // Create conversation if none active
    if (!currentConvoId) {
        try {
            const resp = await fetch('/api/conversations', { method: 'POST' });
            if (resp.ok) {
                const convo = await resp.json();
                currentConvoId = convo._id;
            }
        } catch (err) {
            console.error('Create conversation error:', err);
        }
    }

    document.getElementById('quickQuestions').style.display = 'none';
    appendMessage('user', question);
    showLoading();

    isQuerying = true;

    try {
        const resp = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                conversation_id: currentConvoId,
            }),
        });

        removeLoading();

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.error || 'Query failed');
        }

        const data = await resp.json();
        appendMessage('assistant', data.answer, data.sql, data.table_data);
        loadConversations();

    } catch (err) {
        removeLoading();
        appendMessage('assistant', 'Sorry, something went wrong: ' + err.message);
    } finally {
        isQuerying = false;
        input.focus();
    }
}

// ── Logout ───────────────────────────────────────────────

async function logout() {
    try {
        await fetch('/api/logout', { method: 'POST' });
    } catch {}
    window.location.href = '/login';
}
