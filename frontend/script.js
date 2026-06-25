/* ============================================================
   script.js  —  ForestML Frontend Logic
   Charts: Bar, Radar, Donut, Prediction History (Chart.js)
   Console: Full prediction logging for browser DevTools
   ============================================================ */

const API = "http://localhost:8000";
let selectedAlgo   = "knn";
let metricsData    = {};
let predictionLog  = [];   // session history for history chart
let historyChart   = null;

/* ── Chart.js Global Defaults ────────────────────────────── */
const CHART_COLORS = {
  knn:           { border: "#5ba3e8", bg: "rgba(91,163,232,0.25)"  },
  naive_bayes:   { border: "#e8b44a", bg: "rgba(232,180,74,0.25)"  },
  random_forest: { border: "#4abc5a", bg: "rgba(74,188,90,0.25)"   },
};
const FONT_COLOR  = "#8dba97";
const GRID_COLOR  = "rgba(74,188,90,0.08)";
const TICK_FONT   = { family: "DM Mono, monospace", size: 11, color: FONT_COLOR };

function chartDefaults() {
  Chart.defaults.color          = FONT_COLOR;
  Chart.defaults.borderColor    = GRID_COLOR;
  Chart.defaults.font.family    = "DM Mono, monospace";
  Chart.defaults.font.size      = 11;
  Chart.defaults.plugins.legend.labels.color = FONT_COLOR;
  Chart.defaults.plugins.legend.labels.font  = { family: "DM Mono, monospace", size: 11 };
}

/* ── Algorithm Selector ──────────────────────────────────── */
function selectAlgo(algo, btn) {
  selectedAlgo = algo;
  document.querySelectorAll(".algo-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  console.log(`%c[ForestML] Algorithm selected: ${algo}`, "color:#4abc5a;font-weight:bold;");
}

/* ── API Health Check ────────────────────────────────────── */
async function checkAPI() {
  console.log("%c[ForestML] Checking API health...", "color:#5ba3e8;");
  try {
    const r = await fetch(API + "/health", { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      const health = await r.json();
      console.log("%c[ForestML] API online ✓", "color:#4abc5a;font-weight:bold;", health);
      setStatus("green", "API online");
      loadMetrics();
    } else throw new Error("Non-OK response");
  } catch (e) {
    console.warn("[ForestML] API offline:", e.message);
    setStatus("red", "API offline");
    showOfflineMetrics();
  }
}

function setStatus(color, text) {
  const dot  = document.getElementById("apiDot");
  const span = document.getElementById("apiStatus");
  dot.style.background = color === "green" ? "#4abc5a" : "#e05c5c";
  dot.style.boxShadow  = color === "green" ? "0 0 6px #4abc5a" : "none";
  span.textContent     = text;
}

/* ── Load Metrics from API ───────────────────────────────── */
async function loadMetrics() {
  try {
    const r    = await fetch(API + "/metrics");
    const data = await r.json();
    console.log("%c[ForestML] Metrics loaded:", "color:#5ba3e8;", data);
    metricsData = data;
    renderMetricsTable(data);
    renderBarChart(data);
    renderRadarChart(data);
  } catch {
    showOfflineMetrics();
  }
  renderDonutChart();
}

function showOfflineMetrics() {
  const data = {
    knn:           { accuracy: 68.97, f1: 68.34, precision: 68.46, recall: 68.97 },
    naive_bayes:   { accuracy: 12.53, f1: 11.76, precision: 64.16, recall: 12.53 },
    random_forest: { accuracy: 71.53, f1: 70.95, precision: 71.21, recall: 71.53 },
  };
  console.log("%c[ForestML] Using offline fallback metrics", "color:#e8b44a;", data);
  metricsData = data;
  renderMetricsTable(data);
  renderBarChart(data);
  renderRadarChart(data);
  renderDonutChart();
}

/* ── Metrics Table ───────────────────────────────────────── */
function metricClass(raw) {
  if (raw === "—") return "na";
  const n = parseFloat(raw);
  if (n >= 70) return "best";
  if (n >= 40) return "mid";
  return "low";
}

function renderMetricsTable(data) {
  const algos = [
    { key: "knn",           name: "K-Nearest Neighbors", type: "supervised",   label: "Supervised"    },
    { key: "naive_bayes",   name: "Naïve Bayes",          type: "supervised",   label: "Supervised"    },
    { key: "random_forest", name: "Random Forest",        type: "supervised",   label: "Supervised ★", chosen: true },
    { key: "kmeans",        name: "K-Means Clustering",   type: "unsupervised", label: "Unsupervised"  },
  ];
  const tbody = document.getElementById("compTableBody");
  tbody.innerHTML = algos.map(a => {
    const m    = data[a.key];
    const fmt  = v => m ? v.toFixed(1) + "%" : "—";
    const acc  = fmt(m?.accuracy);
    const f1   = fmt(m?.f1);
    const prec = fmt(m?.precision);
    const rec  = fmt(m?.recall);
    const badge = a.chosen ? `<span class="algo-type-badge badge-chosen">chosen</span>` : "";
    return `<tr>
      <td class="algo-name">${a.name}${badge}</td>
      <td><span class="algo-type-badge badge-${a.type}">${a.label}</span></td>
      <td class="metric-cell ${metricClass(acc)}">${acc}</td>
      <td class="metric-cell ${metricClass(f1)}">${f1}</td>
      <td class="metric-cell ${metricClass(prec)}">${prec}</td>
      <td class="metric-cell ${metricClass(rec)}">${rec}</td>
    </tr>`;
  }).join("");
}

/* ── Chart 1: Grouped Bar Chart ──────────────────────────── */
function renderBarChart(data) {
  const ctx    = document.getElementById("barChart").getContext("2d");
  const labels = ["Accuracy", "F1 Score", "Precision", "Recall"];
  const algos  = [
    { key: "knn",           label: "KNN"           },
    { key: "naive_bayes",   label: "Naïve Bayes"   },
    { key: "random_forest", label: "Random Forest" },
  ];
  const datasets = algos.map(a => ({
    label:           a.label,
    data:            [data[a.key]?.accuracy, data[a.key]?.f1, data[a.key]?.precision, data[a.key]?.recall],
    backgroundColor: CHART_COLORS[a.key].bg,
    borderColor:     CHART_COLORS[a.key].border,
    borderWidth:     2,
    borderRadius:    4,
  }));

  new Chart(ctx, {
    type: "bar",
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(1)}%` }
        }
      },
      scales: {
        y: {
          min: 0, max: 100,
          ticks: { callback: v => v + "%", font: TICK_FONT, color: FONT_COLOR },
          grid:  { color: GRID_COLOR },
        },
        x: {
          ticks: { font: TICK_FONT, color: FONT_COLOR },
          grid:  { color: GRID_COLOR },
        }
      }
    }
  });
  console.log("%c[ForestML] Bar chart rendered", "color:#4abc5a;");
}

/* ── Chart 2: Radar Chart ────────────────────────────────── */
function renderRadarChart(data) {
  const ctx    = document.getElementById("radarChart").getContext("2d");
  const labels = ["Accuracy", "F1 Score", "Precision", "Recall"];
  const algos  = [
    { key: "knn",           label: "KNN"           },
    { key: "naive_bayes",   label: "Naïve Bayes"   },
    { key: "random_forest", label: "Random Forest" },
  ];
  const datasets = algos.map(a => ({
    label:           a.label,
    data:            [data[a.key]?.accuracy, data[a.key]?.f1, data[a.key]?.precision, data[a.key]?.recall],
    backgroundColor: CHART_COLORS[a.key].bg,
    borderColor:     CHART_COLORS[a.key].border,
    borderWidth:     2,
    pointBackgroundColor: CHART_COLORS[a.key].border,
    pointRadius:     4,
  }));

  new Chart(ctx, {
    type: "radar",
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { position: "top" } },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { callback: v => v + "%", font: TICK_FONT, color: FONT_COLOR, backdropColor: "transparent" },
          grid:  { color: GRID_COLOR },
          angleLines: { color: GRID_COLOR },
          pointLabels: { font: { family: "DM Mono, monospace", size: 11 }, color: FONT_COLOR },
        }
      }
    }
  });
  console.log("%c[ForestML] Radar chart rendered", "color:#4abc5a;");
}

/* ── Chart 3: Donut Chart ────────────────────────────────── */
function renderDonutChart() {
  const ctx = document.getElementById("donutChart").getContext("2d");
  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Lodgepole Pine", "Spruce/Fir", "Ponderosa Pine", "Krummholz", "Douglas Fir", "Aspen", "Cottonwood/Willow"],
      datasets: [{
        data: [283301, 211840, 35754, 20510, 17367, 9493, 2747],
        backgroundColor: [
          "rgba(74,188,90,0.8)",
          "rgba(91,163,232,0.8)",
          "rgba(232,180,74,0.8)",
          "rgba(109,212,127,0.8)",
          "rgba(224,92,92,0.8)",
          "rgba(46,139,62,0.8)",
          "rgba(141,186,151,0.8)",
        ],
        borderColor: "#0b1a0e",
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      cutout: "60%",
      plugins: {
        legend: { position: "right", labels: { font: { size: 10 }, padding: 10 } },
        tooltip: {
          callbacks: {
            label: ctx => {
              const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
              const pct   = ((ctx.parsed / total) * 100).toFixed(1);
              return ` ${ctx.label}: ${ctx.parsed.toLocaleString()} (${pct}%)`;
            }
          }
        }
      }
    }
  });
  console.log("%c[ForestML] Donut chart rendered", "color:#4abc5a;");
}

/* ── Chart 4: Prediction History Chart ──────────────────── */
function updateHistoryChart() {
  const canvas = document.getElementById("historyChart");
  const empty  = document.getElementById("predHistoryEmpty");

  if (predictionLog.length === 0) {
    canvas.style.display = "none";
    empty.style.display  = "block";
    return;
  }

  canvas.style.display = "block";
  empty.style.display  = "none";

  const labels   = predictionLog.map((_, i) => `#${i + 1}`);
  const confData = predictionLog.map(p => p.confidence);
  const colors   = predictionLog.map(p => CHART_COLORS[p.algo]?.border || "#4abc5a");

  if (historyChart) {
    historyChart.data.labels              = labels;
    historyChart.data.datasets[0].data   = confData;
    historyChart.data.datasets[0].borderColor     = colors;
    historyChart.data.datasets[0].backgroundColor = colors.map(c => c + "44");
    historyChart.data.datasets[0].pointBackgroundColor = colors;
    historyChart.update();
  } else {
    const ctx = canvas.getContext("2d");
    historyChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: "Top Confidence %",
          data:  confData,
          borderColor:     colors,
          backgroundColor: colors.map(c => c + "22"),
          borderWidth: 2,
          pointBackgroundColor: colors,
          pointRadius: 5,
          tension: 0.3,
          fill: true,
          segment: {
            borderColor: ctx => colors[ctx.p0DataIndex] || "#4abc5a",
          }
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => {
                const i = items[0].dataIndex;
                return `Prediction #${i + 1}: ${predictionLog[i].cover_type}`;
              },
              label: (item) => {
                const i   = item.dataIndex;
                const log = predictionLog[i];
                return [
                  ` Algorithm: ${log.algo.replace(/_/g, " ").toUpperCase()}`,
                  ` Confidence: ${log.confidence.toFixed(1)}%`,
                ];
              }
            }
          }
        },
        scales: {
          y: {
            min: 0, max: 100,
            ticks: { callback: v => v + "%", font: TICK_FONT, color: FONT_COLOR },
            grid:  { color: GRID_COLOR },
          },
          x: {
            ticks: { font: TICK_FONT, color: FONT_COLOR },
            grid:  { color: GRID_COLOR },
          }
        }
      }
    });
  }
  console.log(`%c[ForestML] History chart updated (${predictionLog.length} predictions)`, "color:#4abc5a;");
}

/* ── Run Prediction ──────────────────────────────────────── */
async function runPrediction() {
  const btn       = document.getElementById("predictBtn");
  btn.disabled    = true;
  btn.textContent = "Running…";
  showResultState("loading");

  const payload = {
    elevation:          +document.getElementById("elevation").value,
    aspect:             +document.getElementById("aspect").value,
    slope:              +document.getElementById("slope").value,
    h_dist_hydrology:   +document.getElementById("h_dist_hydrology").value,
    v_dist_hydrology:   +document.getElementById("v_dist_hydrology").value,
    h_dist_roadways:    +document.getElementById("h_dist_roadways").value,
    hillshade_9am:      +document.getElementById("hillshade_9am").value,
    hillshade_noon:     +document.getElementById("hillshade_noon").value,
    hillshade_3pm:      +document.getElementById("hillshade_3pm").value,
    h_dist_fire_points: +document.getElementById("h_dist_fire_points").value,
    wilderness_area:    +document.getElementById("wilderness_area").value,
    soil_type:          +document.getElementById("soil_type").value,
    algorithm:          selectedAlgo,
  };

  /* ── Console: Log the outgoing request ── */
  console.group("%c[ForestML] Prediction Request", "color:#5ba3e8;font-weight:bold;");
  console.log("Algorithm:", selectedAlgo.toUpperCase());
  console.table(payload);
  console.groupEnd();

  try {
    const r = await fetch(API + "/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
      signal:  AbortSignal.timeout(10000),
    });

    if (!r.ok) {
      const err = await r.json();
      throw new Error(err.detail || "Prediction failed");
    }

    const data = await r.json();

    /* ── Console: Log full response ── */
    console.group("%c[ForestML] Prediction Response ✓", "color:#4abc5a;font-weight:bold;");
    console.log("Predicted Class:", data.cover_type || `Cluster ${data.cluster}`);
    if (data.probabilities) {
      console.log("Class Probabilities:");
      console.table(
        Object.fromEntries(
          Object.entries(data.probabilities).map(([k, v]) => [k, v.toFixed(1) + "%"])
        )
      );
    }
    if (data.metrics) {
      console.log("Model Metrics:");
      console.table({
        Accuracy:  data.metrics.accuracy?.toFixed(2) + "%",
        F1:        data.metrics.f1?.toFixed(2) + "%",
        Precision: data.metrics.precision?.toFixed(2) + "%",
        Recall:    data.metrics.recall?.toFixed(2) + "%",
      });
    }
    console.log("Full Response Object:", data);
    console.groupEnd();

    /* ── Log to session history ── */
    if (data.probabilities) {
      const topConf = Math.max(...Object.values(data.probabilities));
      predictionLog.push({
        algo:       selectedAlgo,
        cover_type: data.cover_type,
        confidence: topConf,
        timestamp:  new Date().toLocaleTimeString(),
      });
      console.log(`%c[ForestML] Session predictions so far: ${predictionLog.length}`, "color:#8dba97;");
      updateHistoryChart();
    }

    renderResult(data);
  } catch (e) {
    console.error("[ForestML] Prediction error:", e.message);
    showResultState("error", e.message);
  }

  btn.disabled    = false;
  btn.textContent = "Predict Cover Type →";
}

/* ── Show/Hide Result States ─────────────────────────────── */
function showResultState(state, msg = "") {
  document.getElementById("resultEmpty").style.display   = state === "empty"   ? "flex"  : "none";
  document.getElementById("resultContent").style.display = state === "content" ? "block" : "none";
  document.getElementById("resultLoading").style.display = state === "loading" ? "block" : "none";
  document.getElementById("resultError").style.display   = state === "error"   ? "block" : "none";
  if (state === "error") {
    document.getElementById("resultError").innerHTML =
      `<div class="error-msg">⚠ ${msg}<br>
       <small style="opacity:.7">Make sure FastAPI is running:<br>
       cd backend &amp;&amp; uvicorn main:app --reload</small></div>`;
  }
}

/* ── Render Prediction Result ────────────────────────────── */
function renderResult(data) {
  if (data.cluster !== undefined) {
    document.getElementById("rBadge").textContent = "k-means cluster";
    document.getElementById("rClass").textContent = `Cluster ${data.cluster}`;
    document.getElementById("rAlgo").textContent  = "Unsupervised — no class label assigned";
    document.getElementById("probBars").innerHTML =
      `<p style="color:var(--text3);font-size:.8rem;">
         K-Means is unsupervised — it groups by feature similarity.<br>No probability output available.
       </p>`;
    document.getElementById("metricsGrid").innerHTML =
      `<p style="color:var(--text3);font-size:.8rem;grid-column:span 2;">
         K-Means has no supervised evaluation metrics.
       </p>`;
    showResultState("content");
    return;
  }

  document.getElementById("rBadge").textContent = `class ${data.prediction}`;
  document.getElementById("rClass").textContent = data.cover_type;
  document.getElementById("rAlgo").textContent  =
    `via ${data.algorithm.replace(/_/g, " ").toUpperCase()}`;

  const probs   = data.probabilities || {};
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  const maxP    = entries.length ? entries[0][1] : 1;
  document.getElementById("probBars").innerHTML = entries.map(([name, pct]) => `
    <div class="prob-row">
      <span class="prob-name">${name}</span>
      <div class="prob-track">
        <div class="prob-bar${pct === maxP ? " top" : ""}" style="width:${Math.max(pct, 0.5)}%"></div>
      </div>
      <span class="prob-val">${pct.toFixed(1)}%</span>
    </div>`).join("");

  const m = data.metrics || {};
  document.getElementById("metricsGrid").innerHTML =
    [["Accuracy", m.accuracy], ["F1 Score", m.f1], ["Precision", m.precision], ["Recall", m.recall]]
    .map(([label, val]) =>
      `<div class="m-cell">
         <div class="mv">${val !== undefined ? val.toFixed(1) + "%" : "—"}</div>
         <div class="ml">${label}</div>
       </div>`
    ).join("");

  showResultState("content");
}

/* ── Init ────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  chartDefaults();
  checkAPI();
  console.log(
    "%c ForestML Dashboard Loaded ",
    "background:#2e8b3e;color:#fff;font-size:14px;font-weight:bold;padding:4px 8px;border-radius:4px;"
  );
  console.log("%cOpen DevTools Console to see live prediction data as you predict.", "color:#8dba97;");
});
/* ── Theme Toggle ─────────────────────────────────────────── */
function initThemeToggle() {
    const toggle = document.createElement('div');
    toggle.className = 'theme-toggle';
    toggle.innerHTML = '🌙';
    toggle.onclick = () => {
        document.body.classList.toggle('light-theme');
        toggle.innerHTML = document.body.classList.contains('light-theme') ? '🌞' : '🌙';
        localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
        showToast(`Theme changed to ${document.body.classList.contains('light-theme') ? 'light' : 'dark'} mode`);
    };
    
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
        toggle.innerHTML = '🌞';
    }
    
    document.body.appendChild(toggle);
}

/* ── Toast Notifications ──────────────────────────────────── */
function showToast(message, type = 'success') {
    const container = document.getElementById('toastContainer') || (() => {
        const div = document.createElement('div');
        div.id = 'toastContainer';
        div.className = 'toast-container';
        document.body.appendChild(div);
        return div;
    })();
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

/* ── Batch CSV Upload ─────────────────────────────────────── */
function initBatchUpload() {
    const uploadZone = document.createElement('div');
    uploadZone.className = 'upload-zone';
    uploadZone.innerHTML = `
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M12 16v-6m0 0-2 2m2-2 2 2M5 12a3 3 0 0 1 3-3h8a3 3 0 0 1 3 3"/>
            <path d="M4 20h16"/>
        </svg>
        <p>Drop CSV file here or click to upload</p>
        <small>Format: elevation,aspect,slope,... (same as single prediction)</small>
    `;
    
    uploadZone.onclick = () => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.csv';
        input.onchange = (e) => handleBatchFile(e.target.files[0]);
        input.click();
    };
    
    uploadZone.ondragover = (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    };
    
    uploadZone.ondragleave = () => {
        uploadZone.classList.remove('drag-over');
    };
    
    uploadZone.ondrop = (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        handleBatchFile(e.dataTransfer.files[0]);
    };
    
    // Add to compare section
    const compareSection = document.getElementById('compare');
    const batchCard = document.createElement('div');
    batchCard.className = 'form-card';
    batchCard.innerHTML = '<h3>📁 Batch Predictions (CSV Upload)</h3>';
    batchCard.appendChild(uploadZone);
    const resultsDiv = document.createElement('div');
    resultsDiv.id = 'batchResults';
    resultsDiv.className = 'batch-results';
    batchCard.appendChild(resultsDiv);
    compareSection.insertBefore(batchCard, compareSection.querySelector('.two-col'));
}

async function handleBatchFile(file) {
    if (!file) return;
    
    const text = await file.text();
    const lines = text.split('\n');
    const headers = lines[0].split(',');
    
    const requests = [];
    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        const values = lines[i].split(',');
        const req = {};
        headers.forEach((h, idx) => {
            req[h.trim()] = parseFloat(values[idx]);
        });
        req.algorithm = selectedAlgo;
        requests.push(req);
    }
    
    showToast(`Processing ${requests.length} predictions...`);
    
    try {
        const response = await fetch(API + "/predict/batch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ requests })
        });
        
        const data = await response.json();
        displayBatchResults(data);
        showToast(`Completed ${data.successful}/${data.total} predictions`);
    } catch (error) {
        showToast(`Batch prediction failed: ${error.message}`, 'error');
    }
}

function displayBatchResults(data) {
    const container = document.getElementById('batchResults');
    if (!container) return;
    
    const table = document.createElement('table');
    table.innerHTML = `
        <thead>
            <tr>
                <th>#</th>
                <th>Cover Type</th>
                <th>Confidence</th>
                <th>Algorithm</th>
            </tr>
        </thead>
        <tbody>
            ${data.results.map((r, idx) => `
                <tr>
                    <td>${idx + 1}</td>
                    <td>${r.cover_type || r.cluster || 'Error'}</td>
                    <td>${r.top_confidence ? r.top_confidence.toFixed(1) + '%' : 'N/A'}</td>
                    <td>${r.algorithm || 'N/A'}</td>
                </tr>
            `).join('')}
        </tbody>
    `;
    
    container.innerHTML = '';
    container.appendChild(table);
    
    // Add export button
    const exportBtn = document.createElement('button');
    exportBtn.className = 'export-btn';
    exportBtn.innerHTML = '📥 Export Results to CSV';
    exportBtn.onclick = () => exportBatchResults(data);
    container.appendChild(exportBtn);
}

function exportBatchResults(data) {
    const csv = ['Index,Cover Type,Confidence,Algorithm'];
    data.results.forEach((r, idx) => {
        csv.push(`${idx + 1},${r.cover_type || r.cluster || 'Error'},${r.top_confidence || 'N/A'},${r.algorithm || 'N/A'}`);
    });
    
    const blob = new Blob([csv.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `predictions_${new Date().toISOString().slice(0, 19)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Results exported to CSV');
}

/* ── Keyboard Shortcuts ───────────────────────────────────── */
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter to predict
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            runPrediction();
            showToast('⌨️ Running prediction...');
        }
        
        // Alt + 1-4 to switch algorithms
        if (e.altKey && !e.ctrlKey) {
            const algos = ['knn', 'naive_bayes', 'random_forest', 'kmeans'];
            const idx = parseInt(e.key) - 1;
            if (idx >= 0 && idx < algos.length) {
                e.preventDefault();
                const algo = algos[idx];
                const btn = document.querySelector(`.algo-btn[onclick*="${algo}"]`);
                if (btn) btn.click();
                showToast(`⌨️ Switched to ${algo.replace('_', ' ')}`);
            }
        }
    });
    
    // Add hint
    const hint = document.createElement('div');
    hint.className = 'keyboard-hint';
    hint.innerHTML = '⌨️ Ctrl+Enter to predict | Alt+1/2/3/4 to switch algorithms';
    document.body.appendChild(hint);
}

/* ── Confidence Gauge ─────────────────────────────────────── */
function addConfidenceGauge(container, confidence) {
    const gaugeContainer = document.createElement('div');
    gaugeContainer.className = 'gauge-container';
    gaugeContainer.innerHTML = `
        <div class="gauge">
            <svg width="120" height="120" viewBox="0 0 120 120">
                <circle cx="60" cy="60" r="50" fill="none" stroke="var(--border)" stroke-width="8"/>
                <circle cx="60" cy="60" r="50" fill="none" stroke="var(--green)" stroke-width="8" 
                        stroke-dasharray="${2 * Math.PI * 50}" stroke-dashoffset="${2 * Math.PI * 50 * (1 - confidence / 100)}"
                        stroke-linecap="round"/>
            </svg>
            <div class="gauge-value">${confidence.toFixed(0)}%</div>
        </div>
        <small>Top Prediction Confidence</small>
    `;
    container.appendChild(gaugeContainer);
}

// Override renderResult to include confidence gauge
const originalRenderResult = renderResult;
renderResult = function(data) {
    originalRenderResult(data);
    
    if (data.top_confidence) {
        const resultContent = document.getElementById('resultContent');
        const existingGauge = resultContent.querySelector('.gauge-container');
        if (existingGauge) existingGauge.remove();
        
        // Insert after first div
        const firstDiv = resultContent.querySelector('div:first-child');
        addConfidenceGauge(resultContent, data.top_confidence);
        resultContent.insertBefore(resultContent.querySelector('.gauge-container'), firstDiv.nextSibling);
    }
};

/* ── Loading Skeletons ────────────────────────────────────── */
function showSkeleton() {
    const resultPanel = document.getElementById('resultPanel');
    if (!resultPanel) return;
    
    const skeleton = document.createElement('div');
    skeleton.className = 'skeleton';
    skeleton.style.height = '200px';
    skeleton.style.margin = '1rem';
    resultPanel.appendChild(skeleton);
    return skeleton;
}

function hideSkeleton(skeleton) {
    if (skeleton) skeleton.remove();
}

// Override runPrediction to show skeleton
const originalRunPrediction = runPrediction;
runPrediction = async function() {
    const skeleton = showSkeleton();
    try {
        await originalRunPrediction();
    } finally {
        hideSkeleton(skeleton);
    }
};

/* ── Initialize All New Features ─────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    // Don't override existing initialization
    initThemeToggle();
    initBatchUpload();
    initKeyboardShortcuts();
    console.log('%c✨ Enhanced features loaded (Theme, Batch Upload, Keyboard Shortcuts)', 'color:#4abc5a;');
});