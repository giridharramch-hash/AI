/**
 * Extractive Question Answering Capstone Web App
 * Front-end Controller: Demo, Heatmap, Chart.js, Course Browser, and Dark Mode
 */

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initHeroSimulator();
  initDemo();
  initPipelineAccordion();
  initAttentionExplainer();
  initResultsAndCharts();
  initCourseBrowser();
  initFooterAndConfig();
  checkSystemStatus();
});

/* -------------------------------------------------------------------------- */
/* 1. Theme (Dark / Light Mode)                                               */
/* -------------------------------------------------------------------------- */
function initTheme() {
  const themeToggle = document.getElementById("themeToggle");
  const savedTheme = localStorage.getItem("capstone_theme") || "light";
  document.documentElement.setAttribute("data-theme", savedTheme);

  updateThemeIcon(savedTheme);

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("capstone_theme", next);
      updateThemeIcon(next);
      window.dispatchEvent(new Event("themeChanged"));
    });
  }
}

function updateThemeIcon(theme) {
  const icon = document.getElementById("themeIcon");
  if (!icon) return;
  if (theme === "dark") {
    icon.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>`;
  } else {
    icon.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
  }
}

/* -------------------------------------------------------------------------- */
/* 2. System Status & Graceful Offline Banner                                */
/* -------------------------------------------------------------------------- */
async function checkSystemStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();
    if (!data.model_available) {
      const banner = document.getElementById("fallbackBanner");
      if (banner) {
        banner.style.display = "flex";
        banner.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          <span>Running in <strong>Sentence Baseline Fallback Mode</strong> (Transformer weights unavailable offline).</span>
        `;
      }
    }
  } catch (err) {
    console.warn("Could not check backend status:", err);
  }
}

/* -------------------------------------------------------------------------- */
/* 3. Hero Interactive Simulator                                              */
/* -------------------------------------------------------------------------- */
function initHeroSimulator() {
  const qElem = document.getElementById("heroTypingQuestion");
  const pElem = document.getElementById("heroPassageText");
  if (!qElem || !pElem) return;

  const simulations = [
    {
      q: "Which company released BERT?",
      passage: "BERT, which stands for Bidirectional Encoder Representations from Transformers, is an encoder-only model pretrained with masked language modeling and next sentence prediction. It was released by Google in 2018.",
      ans: "Google"
    },
    {
      q: "What property must the heuristic have for A* to be optimal?",
      passage: "A* search evaluates nodes using f(n) = g(n) + h(n), where g(n) is the cost so far and h(n) is the estimated cost to the goal. A* is optimal when the heuristic is admissible, meaning it never overestimates the true cost to reach the goal.",
      ans: "admissible"
    },
    {
      q: "What does resolution try to derive to complete a proof?",
      passage: "Resolution is an inference rule that is refutation-complete for propositional logic. To prove that a sentence follows from the KB, it adds the negation of the sentence to the KB in conjunctive normal form and tries to derive the empty clause.",
      ans: "the empty clause"
    }
  ];

  let simIndex = 0;

  function runSim() {
    const sim = simulations[simIndex];
    let charIdx = 0;
    qElem.textContent = "";
    
    // Normal passage without highlight initially
    pElem.innerHTML = escapeHtml(sim.passage);

    const typeInterval = setInterval(() => {
      if (charIdx < sim.q.length) {
        qElem.textContent += sim.q.charAt(charIdx);
        charIdx++;
      } else {
        clearInterval(typeInterval);
        // Highlight answer span after brief pause
        setTimeout(() => {
          const parts = sim.passage.split(sim.ans);
          if (parts.length === 2) {
            pElem.innerHTML = `${escapeHtml(parts[0])}<span class="hero-highlight-pill">${escapeHtml(sim.ans)}</span>${escapeHtml(parts[1])}`;
          }
          // Move to next simulation after 3.5s
          setTimeout(() => {
            simIndex = (simIndex + 1) % simulations.length;
            runSim();
          }, 3500);
        }, 500);
      }
    }, 45);
  }

  runSim();
}

/* -------------------------------------------------------------------------- */
/* 4. Live Demo (Centerpiece)                                                 */
/* -------------------------------------------------------------------------- */
function initDemo() {
  const input = document.getElementById("demoQuestionInput");
  const searchBtn = document.getElementById("demoSearchBtn");
  const clearBtn = document.getElementById("demoClearBtn");
  const chipsContainer = document.getElementById("exampleChips");
  const topKButtons = document.querySelectorAll(".top-k-btn");
  const compareToggle = document.getElementById("compareBaselineToggle");
  const resultCard = document.getElementById("demoResultCard");
  const skeleton = document.getElementById("demoLoadingSkeleton");

  let currentTopK = 3;

  // Render example chips from config
  if (chipsContainer && window.CAPSTONE_CONFIG) {
    chipsContainer.innerHTML = "";
    window.CAPSTONE_CONFIG.exampleQuestions.forEach(q => {
      const btn = document.createElement("button");
      btn.className = "chip-btn";
      btn.textContent = q;
      btn.addEventListener("click", () => {
        if (input) {
          input.value = q;
          executeAsk(q, currentTopK);
        }
      });
      chipsContainer.appendChild(btn);
    });
  }

  // Top-k buttons
  topKButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      topKButtons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentTopK = parseInt(btn.dataset.k, 10);
      if (input && input.value.trim()) {
        executeAsk(input.value.trim(), currentTopK);
      }
    });
  });

  // Search trigger
  if (searchBtn && input) {
    searchBtn.addEventListener("click", () => {
      const q = input.value.trim();
      if (q) executeAsk(q, currentTopK);
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const q = input.value.trim();
        if (q) executeAsk(q, currentTopK);
      }
    });
  }

  if (clearBtn && input) {
    clearBtn.addEventListener("click", () => {
      input.value = "";
      input.focus();
    });
  }

  // Compare toggle
  if (compareToggle) {
    compareToggle.addEventListener("change", () => {
      const compareBox = document.getElementById("comparisonContainer");
      if (compareBox) {
        compareBox.style.display = compareToggle.checked ? "block" : "none";
      }
    });
  }

  // Collapsible Under the Hood
  const hoodToggle = document.getElementById("underTheHoodToggle");
  const hoodContent = document.getElementById("underTheHoodContent");
  if (hoodToggle && hoodContent) {
    hoodToggle.addEventListener("click", () => {
      hoodContent.classList.toggle("open");
      const icon = hoodToggle.querySelector(".accordion-icon");
      if (icon) {
        icon.style.transform = hoodContent.classList.contains("open") ? "rotate(180deg)" : "rotate(0deg)";
      }
    });
  }

  async function executeAsk(question, topK) {
    if (!question) return;
    if (skeleton) skeleton.style.display = "block";
    if (resultCard) resultCard.style.display = "none";

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Error querying QA pipeline.");
      }

      const data = await res.json();
      renderDemoResult(data);
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      if (skeleton) skeleton.style.display = "none";
    }
  }

  function renderDemoResult(data) {
    if (!resultCard) return;
    resultCard.style.display = "block";

    // Answer text
    const ansText = document.getElementById("resAnswerText");
    if (ansText) ansText.textContent = data.answer || "No span found.";

    // Source Label
    const srcTag = document.getElementById("resSourceTag");
    if (srcTag) {
      srcTag.textContent = `${data.source.label || data.source.unit} (Passage #${data.source.id})`;
    }

    // Confidence meter
    const confVal = document.getElementById("resConfidenceVal");
    const confFill = document.getElementById("resConfidenceFill");
    const confWarn = document.getElementById("resConfidenceWarning");

    const confPct = Math.round((data.confidence || 0) * 100);
    if (confVal) confVal.textContent = `${(data.confidence || 0).toFixed(3)} (${confPct}%)`;
    if (confFill) {
      confFill.style.width = `${Math.min(100, Math.max(5, confPct))}%`;
      if (data.confidence < 0.2) {
        confFill.style.backgroundColor = "#ef4444";
      } else if (data.confidence < 0.6) {
        confFill.style.backgroundColor = "#f59e0b";
      } else {
        confFill.style.backgroundColor = "#2b6cb0";
      }
    }

    if (confWarn) {
      confWarn.style.display = data.confidence < 0.2 ? "flex" : "none";
    }

    // Highlighted Passage
    const passageBox = document.getElementById("resPassageText");
    if (passageBox && data.passage) {
      const fullText = data.passage.text;
      const [start, end] = data.char_span || [0, 0];

      if (start >= 0 && end > start && end <= fullText.length) {
        const before = fullText.slice(0, start);
        const match = fullText.slice(start, end);
        const after = fullText.slice(end);
        passageBox.innerHTML = `${escapeHtml(before)}<mark class="answer-span-highlight">${escapeHtml(match)}</mark>${escapeHtml(after)}`;
      } else {
        passageBox.textContent = fullText;
      }
    }

    // Baseline Comparison
    if (data.baseline) {
      const transAns = document.getElementById("compareTransformerAns");
      const transConf = document.getElementById("compareTransformerConf");
      const baseAns = document.getElementById("compareBaselineAns");
      const baseConf = document.getElementById("compareBaselineConf");

      if (transAns) transAns.textContent = `"${data.answer}"`;
      if (transConf) transConf.textContent = `Confidence: ${(data.confidence || 0).toFixed(3)}`;
      if (baseAns) baseAns.textContent = `"${data.baseline.text}"`;
      if (baseConf) baseConf.textContent = `Cosine Sim: ${(data.baseline.confidence || 0).toFixed(3)}`;
    }

    // Under the hood: Candidate Passages
    const candTbody = document.getElementById("candidatesTableBody");
    if (candTbody && data.candidates) {
      candTbody.innerHTML = "";
      data.candidates.forEach((c, idx) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>#${idx + 1}</strong></td>
          <td><span class="badge badge-blue">[${escapeHtml(c.passage.unit)}] ${escapeHtml(c.passage.title)}</span></td>
          <td><code>${c.retrieval_score.toFixed(3)}</code></td>
          <td><span class="answer-span-highlight">${escapeHtml(c.span)}</span></td>
          <td><code>${c.confidence.toFixed(3)}</code></td>
        `;
        candTbody.appendChild(tr);
      });
    }

    // Under the hood: Start / End Token Logits
    const inspectionBox = document.getElementById("inspectionTokensBox");
    if (inspectionBox) {
      if (data.inspection && data.inspection.start && data.inspection.end) {
        inspectionBox.style.display = "block";
        const startList = data.inspection.start.map(([tok, p]) => `<li><code>${escapeHtml(tok)}</code>: ${(p * 100).toFixed(1)}%</li>`).join("");
        const endList = data.inspection.end.map(([tok, p]) => `<li><code>${escapeHtml(tok)}</code>: ${(p * 100).toFixed(1)}%</li>`).join("");
        inspectionBox.innerHTML = `
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px;">
            <div style="background: var(--bg-card-subtle); padding: 12px; border-radius: 8px;">
              <strong style="font-size: 0.8rem; text-transform: uppercase; color: var(--color-blue);">Top Start Boundary Tokens</strong>
              <ul style="list-style: none; padding-left: 0; margin-top: 8px; font-size: 0.82rem;">${startList}</ul>
            </div>
            <div style="background: var(--bg-card-subtle); padding: 12px; border-radius: 8px;">
              <strong style="font-size: 0.8rem; text-transform: uppercase; color: var(--color-blue);">Top End Boundary Tokens</strong>
              <ul style="list-style: none; padding-left: 0; margin-top: 8px; font-size: 0.82rem;">${endList}</ul>
            </div>
          </div>
        `;
      } else {
        inspectionBox.style.display = "none";
      }
    }

    // Smooth scroll to result
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

/* -------------------------------------------------------------------------- */
/* 5. Pipeline Steps Accordion (How It Works)                                */
/* -------------------------------------------------------------------------- */
function initPipelineAccordion() {
  const cards = document.querySelectorAll(".step-card");
  cards.forEach(card => {
    card.addEventListener("click", () => {
      card.classList.toggle("expanded");
    });
  });
}

/* -------------------------------------------------------------------------- */
/* 6. Self-Attention Explainer & Heatmap                                      */
/* -------------------------------------------------------------------------- */
function initAttentionExplainer() {
  // Render KaTeX Math
  const mathElem = document.getElementById("katexAttentionFormula");
  if (mathElem && window.katex) {
    try {
      window.katex.render(
        "\\text{Attention}(Q, K, V) = \\text{softmax}\\left( \\frac{Q K^T}{\\sqrt{d_k}} \\right) V",
        mathElem,
        { displayMode: true, throwOnError: false }
      );
    } catch (e) {
      console.warn("KaTeX render error:", e);
    }
  }

  const slider = document.getElementById("dkSlider");
  const dkVal = document.getElementById("dkValue");
  const scaleToggle = document.getElementById("scaleToggle");
  const fallbackImgBtn = document.getElementById("toggleFallbackImgBtn");
  const fallbackImgWrap = document.getElementById("attentionFallbackWrap");

  if (fallbackImgBtn && fallbackImgWrap) {
    fallbackImgBtn.addEventListener("click", () => {
      const isVisible = fallbackImgWrap.style.display === "block";
      fallbackImgWrap.style.display = isVisible ? "none" : "block";
      fallbackImgBtn.textContent = isVisible ? "Show Python matplotlib comparison" : "Hide comparison";
    });
  }

  async function updateHeatmap() {
    const dk = slider ? parseFloat(slider.value) : 4;
    const scale = scaleToggle ? scaleToggle.checked : true;
    if (dkVal) dkVal.textContent = dk;

    try {
      const res = await fetch(`/api/attention?d_k=${dk}&scale=${scale}`);
      if (!res.ok) return;
      const data = await res.json();
      renderHeatmapGrid(data);
    } catch (err) {
      console.warn("Error loading attention matrix:", err);
    }
  }

  if (slider) {
    slider.addEventListener("input", () => {
      if (dkVal) dkVal.textContent = slider.value;
    });
    slider.addEventListener("change", updateHeatmap);
  }

  if (scaleToggle) {
    scaleToggle.addEventListener("change", updateHeatmap);
  }

  function renderHeatmapGrid(data) {
    const tableWrap = document.getElementById("heatmapTableWrap");
    if (!tableWrap) return;

    const { tokens, weights, row_sums } = data;
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";

    let html = `<table class="heatmap-table" aria-label="Interactive Self-Attention Heatmap">`;
    
    // Header Row (Key tokens)
    html += `<thead><tr><th class="heatmap-row-token" style="font-size: 0.72rem; color: var(--text-faint);">Query \\ Key</th>`;
    tokens.forEach(tok => {
      html += `<th class="heatmap-header-token"><code>${escapeHtml(tok)}</code></th>`;
    });
    html += `<th class="heatmap-header-token" style="font-size: 0.72rem;">Row Sum</th></tr></thead><tbody>`;

    // Data rows
    tokens.forEach((qTok, r) => {
      html += `<tr><td class="heatmap-row-token"><code>${escapeHtml(qTok)}</code></td>`;
      tokens.forEach((kTok, c) => {
        const w = weights[r][c];
        // Blue palette interpolation
        const bg = isDark
          ? `rgba(59, 130, 246, ${Math.max(0.08, w)})`
          : `rgba(43, 108, 176, ${Math.max(0.05, w)})`;
        const textCol = w > 0.4 ? "#ffffff" : "var(--text-main)";

        html += `
          <td class="heatmap-cell" 
              style="background-color: ${bg}; color: ${textCol};" 
              title="Query '${escapeHtml(qTok)}' -> Key '${escapeHtml(kTok)}': ${w.toFixed(4)}"
              data-query="${escapeHtml(qTok)}"
              data-key="${escapeHtml(kTok)}"
              data-val="${w.toFixed(4)}">
            ${(w * 100).toFixed(0)}%
          </td>
        `;
      });
      html += `<td class="heatmap-row-sum">${(row_sums[r] || 1).toFixed(2)}</td></tr>`;
    });

    html += `</tbody></table>`;
    tableWrap.innerHTML = html;

    // Hover tooltip / detail
    const cells = tableWrap.querySelectorAll(".heatmap-cell");
    const detailBox = document.getElementById("heatmapCellDetail");
    cells.forEach(cell => {
      cell.addEventListener("mouseenter", () => {
        if (detailBox) {
          detailBox.innerHTML = `Attention score: <strong>${cell.dataset.query}</strong> &rarr; <strong>${cell.dataset.key}</strong> is <strong>${cell.dataset.val}</strong> (${(parseFloat(cell.dataset.val) * 100).toFixed(1)}%)`;
        }
      });
    });
  }

  // Initial fetch
  updateHeatmap();
  window.addEventListener("themeChanged", updateHeatmap);
}

/* -------------------------------------------------------------------------- */
/* 7. Results & Chart.js                                                      */
/* -------------------------------------------------------------------------- */
let mainChartInstance = null;
let ablationChartInstance = null;
let unitChartInstance = null;

async function initResultsAndCharts() {
  try {
    const res = await fetch("/api/results");
    if (!res.ok) return;
    const data = await res.json();
    if (data.status === "not_generated") return;

    renderMetricCards(data);
    renderCharts(data);
    renderErrorTable(data);
  } catch (err) {
    console.warn("Could not load /api/results:", err);
  }
}

function renderMetricCards(data) {
  const mainSystemName = data.main_system || "transformer_top3";
  const mainSys = data.systems ? data.systems[mainSystemName] : null;
  if (!mainSys) return;

  const hitVal = data.retrieval_hit_at_k ? (data.retrieval_hit_at_k[data.top_k || "3"] * 100).toFixed(1) : "100.0";

  document.getElementById("metricEM").textContent = `${(mainSys.EM * 100).toFixed(1)}%`;
  document.getElementById("metricF1").textContent = `${(mainSys.F1 * 100).toFixed(1)}%`;
  document.getElementById("metricContains").textContent = `${(mainSys.Contains * 100).toFixed(1)}%`;
  document.getElementById("metricHit").textContent = `${hitVal}%`;
  document.getElementById("metricLatency").textContent = `${Math.round(mainSys.latency_ms || 264)} ms`;
}

function renderCharts(data) {
  if (!window.Chart) return;

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  const textColor = isDark ? "#cbd5e1" : "#475569";
  const gridColor = isDark ? "#1e293b" : "#e2e8f0";

  // 1. Grouped Bar: Baseline vs Transformer
  const baseSys = data.systems["baseline_sentence_tfidf"];
  const mainSys = data.systems[data.main_system];

  const ctxMain = document.getElementById("chartBaselineVsTransformer");
  if (ctxMain) {
    if (mainChartInstance) mainChartInstance.destroy();
    mainChartInstance = new Chart(ctxMain, {
      type: "bar",
      data: {
        labels: ["Exact Match (EM)", "Token F1 Score", "Contains (Lenient)"],
        datasets: [
          {
            label: "TF-IDF Sentence Baseline",
            data: [baseSys.EM * 100, baseSys.F1 * 100, baseSys.Contains * 100],
            backgroundColor: isDark ? "#475569" : "#9aa5b1",
            borderRadius: 6
          },
          {
            label: "DistilBERT Reader",
            data: [mainSys.EM * 100, mainSys.F1 * 100, mainSys.Contains * 100],
            backgroundColor: isDark ? "#3b82f6" : "#2b6cb0",
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: textColor } }
        },
        scales: {
          x: { ticks: { color: textColor }, grid: { display: false } },
          y: {
            beginAtZero: true,
            max: 100,
            ticks: { color: textColor, callback: v => `${v}%` },
            grid: { color: gridColor }
          }
        }
      }
    });
  }

  // 2. Top-k Ablation Chart
  const ctxAblation = document.getElementById("chartTopKAblation");
  if (ctxAblation) {
    const k1 = data.systems["transformer_top1"] || { EM: 0.75, F1: 0.903 };
    const k3 = data.systems["transformer_top3"] || { EM: 0.536, F1: 0.622 };
    const k5 = data.systems["transformer_top5"] || { EM: 0.429, F1: 0.521 };

    if (ablationChartInstance) ablationChartInstance.destroy();
    ablationChartInstance = new Chart(ctxAblation, {
      type: "line",
      data: {
        labels: ["k = 1", "k = 3", "k = 5"],
        datasets: [
          {
            label: "Exact Match (EM)",
            data: [k1.EM * 100, k3.EM * 100, k5.EM * 100],
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59, 130, 246, 0.15)",
            tension: 0.3,
            fill: true,
            pointRadius: 5
          },
          {
            label: "F1 Score",
            data: [k1.F1 * 100, k3.F1 * 100, k5.F1 * 100],
            borderColor: "#10b981",
            backgroundColor: "transparent",
            borderDash: [5, 5],
            tension: 0.3,
            pointRadius: 5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: textColor } }
        },
        scales: {
          x: { ticks: { color: textColor }, grid: { display: false } },
          y: {
            beginAtZero: true,
            max: 100,
            ticks: { color: textColor, callback: v => `${v}%` },
            grid: { color: gridColor }
          }
        }
      }
    });
  }

  // 3. Per-Unit Breakdown
  const ctxUnit = document.getElementById("chartPerUnitBreakdown");
  if (ctxUnit && data.per_unit) {
    const units = Object.keys(data.per_unit);
    const emScores = units.map(u => (data.per_unit[u].EM * 100));
    const f1Scores = units.map(u => (data.per_unit[u].F1 * 100));

    if (unitChartInstance) unitChartInstance.destroy();
    unitChartInstance = new Chart(ctxUnit, {
      type: "bar",
      data: {
        labels: units,
        datasets: [
          {
            label: "EM (%)",
            data: emScores,
            backgroundColor: isDark ? "#3b82f6" : "#2b6cb0",
            borderRadius: 4
          },
          {
            label: "F1 (%)",
            data: f1Scores,
            backgroundColor: isDark ? "#60a5fa" : "#63b3ed",
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: textColor } } },
        scales: {
          x: { ticks: { color: textColor }, grid: { display: false } },
          y: { beginAtZero: true, max: 100, ticks: { color: textColor }, grid: { color: gridColor } }
        }
      }
    });
  }
}

window.addEventListener("themeChanged", () => {
  fetch("/api/results")
    .then(r => r.json())
    .then(data => {
      if (data && data.status !== "not_generated") renderCharts(data);
    })
    .catch(() => {});
});

/* -------------------------------------------------------------------------- */
/* 8. Error Analysis Table                                                    */
/* -------------------------------------------------------------------------- */
function renderErrorTable(data) {
  const errors = data.errors || [];
  const tbody = document.getElementById("errorTableBody");
  const unitFilter = document.getElementById("errorUnitFilter");
  const typeFilter = document.getElementById("errorTypeFilter");
  if (!tbody) return;

  function updateTable() {
    const uVal = unitFilter ? unitFilter.value : "ALL";
    const tVal = typeFilter ? typeFilter.value : "ALL";

    const filtered = errors.filter(e => {
      const matchUnit = uVal === "ALL" || e.unit === uVal;
      const matchType = tVal === "ALL" || e.error_type.toLowerCase().includes(tVal.toLowerCase());
      return matchUnit && matchType;
    });

    tbody.innerHTML = "";
    if (filtered.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 20px;">No error cases matching current filter.</td></tr>`;
      return;
    }

    filtered.forEach(err => {
      const isRetMiss = err.error_type.includes("retrieval miss");
      const badgeClass = isRetMiss ? "badge-warning" : "badge-blue";

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="badge badge-muted">${escapeHtml(err.unit)}</span></td>
        <td style="font-weight: 500;">${escapeHtml(err.question)}</td>
        <td><strong style="color: #059669;">${escapeHtml(err.gold)}</strong></td>
        <td><span style="color: #dc2626;">${escapeHtml(err.prediction)}</span></td>
        <td><span class="badge ${badgeClass}">${escapeHtml(err.error_type)}</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  if (unitFilter) unitFilter.addEventListener("change", updateTable);
  if (typeFilter) typeFilter.addEventListener("change", updateTable);

  updateTable();
}

/* -------------------------------------------------------------------------- */
/* 9. Course Material Browser                                                 */
/* -------------------------------------------------------------------------- */
async function initCourseBrowser() {
  const container = document.getElementById("passagesGrid");
  const searchInput = document.getElementById("passageSearchInput");
  const filterPills = document.querySelectorAll(".unit-filter-pill");
  if (!container) return;

  let allPassages = [];
  let currentUnit = "ALL";

  try {
    const res = await fetch("/api/passages");
    if (!res.ok) return;
    allPassages = await res.json();
    renderPassages();
  } catch (err) {
    console.warn("Could not load passages:", err);
  }

  filterPills.forEach(pill => {
    pill.addEventListener("click", () => {
      filterPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      currentUnit = pill.dataset.unit;
      renderPassages();
    });
  });

  if (searchInput) {
    searchInput.addEventListener("input", renderPassages);
  }

  function renderPassages() {
    const query = (searchInput ? searchInput.value : "").toLowerCase().trim();

    const filtered = allPassages.filter(p => {
      const matchUnit = currentUnit === "ALL" || p.unit === currentUnit;
      const matchQuery = !query || p.title.toLowerCase().includes(query) || p.text.toLowerCase().includes(query);
      return matchUnit && matchQuery;
    });

    container.innerHTML = "";
    if (filtered.length === 0) {
      container.innerHTML = `<p style="grid-column: 1/-1; text-align: center; padding: 32px; color: var(--text-muted);">No passages found matching criteria.</p>`;
      return;
    }

    filtered.forEach(p => {
      const card = document.createElement("div");
      card.className = "passage-card";
      card.innerHTML = `
        <div class="passage-card-header">
          <span class="badge badge-blue">${escapeHtml(p.unit)}</span>
          <span style="font-size: 0.75rem; color: var(--text-faint); font-family: var(--font-mono);">${p.word_count || p.text.split(" ").length} words</span>
        </div>
        <div class="passage-card-title">${escapeHtml(p.title || "Lecture Note")}</div>
        <div class="passage-card-body">${escapeHtml(p.text)}</div>
      `;
      container.appendChild(card);
    });
  }
}

/* -------------------------------------------------------------------------- */
/* 10. Footer, Metadata & Team Config                                         */
/* -------------------------------------------------------------------------- */
function initFooterAndConfig() {
  if (!window.CAPSTONE_CONFIG) return;
  const cfg = window.CAPSTONE_CONFIG;

  const teamList = document.getElementById("footerTeamList");
  if (teamList && cfg.teamMembers) {
    teamList.innerHTML = "";
    cfg.teamMembers.forEach(member => {
      const li = document.createElement("li");
      li.textContent = member;
      teamList.appendChild(li);
    });
  }

  const githubLink = document.getElementById("footerGithubLink");
  if (githubLink && cfg.githubUrl) {
    githubLink.href = cfg.githubUrl;
  }

  const posterBtn = document.getElementById("downloadPosterBtn");
  if (posterBtn && cfg.posterUrl) {
    posterBtn.href = cfg.posterUrl;
  }
}

/* -------------------------------------------------------------------------- */
/* Utility: HTML Escaping                                                     */
/* -------------------------------------------------------------------------- */
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
