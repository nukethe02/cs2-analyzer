/**
 * OpenSight Coach UX - Enhanced 60-second analysis display
 * Adds tabbed interface, radar chart, CT/T comparison, and AI coaching view
 */
(function() {
  'use strict';

  // ===== Tab System =====
  const TAB_CONFIG = [
    { id: 'overview', label: 'Overview', icon: '\u26A1' },
    { id: 'performance', label: 'Performance', icon: '\uD83D\uDCCA' },
    { id: 'coach', label: 'AI Coach', icon: '\uD83E\uDD16' },
    { id: 'positions', label: 'Positions', icon: '\uD83D\uDDFA' }
  ];

  function injectStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .coach-tabs{display:flex;gap:2px;background:#111;border-radius:12px;padding:4px;margin:1.5rem 0;border:1px solid #222}
      .coach-tab{flex:1;padding:.7rem 1rem;border:none;background:transparent;color:#888;font-size:.85rem;font-weight:600;cursor:pointer;border-radius:10px;transition:all .2s;display:flex;align-items:center;justify-content:center;gap:.5rem}
      .coach-tab:hover{color:#ccc;background:#1a1a1a}
      .coach-tab.active{background:#1e1e1e;color:#D5C08B;box-shadow:0 2px 8px rgba(0,0,0,.3)}
      .coach-tab-icon{font-size:1.1rem}
      .coach-panel{display:none;animation:fadeIn .3s ease}
      .coach-panel.active{display:block}
      @keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}
      .radar-container{display:flex;justify-content:center;padding:2rem 0}
      .radar-canvas{max-width:360px;max-height:360px}
      .side-comparison{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin:1.5rem 0}
      .side-card{background:#111;border:1px solid #222;border-radius:12px;padding:1.5rem}
      .side-card h3{font-size:.85rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:1rem;display:flex;align-items:center;gap:.5rem}
      .side-ct h3{color:#5c9fd4}.side-t h3{color:#dea74c}
      .side-stat{display:flex;justify-content:space-between;padding:.5rem 0;border-bottom:1px solid #1a1a1a;font-size:.9rem}
      .side-stat .label{color:#888}.side-stat .val{font-weight:700;color:#e0e0e0}
      .coach-insight{background:#111;border:1px solid #222;border-radius:12px;padding:1.5rem;margin:1rem 0}
      .coach-insight.positive{border-left:3px solid #4ade80}
      .coach-insight.negative{border-left:3px solid #f87171}
      .coach-insight.neutral{border-left:3px solid #D5C08B}
      .coach-insight h4{font-size:.95rem;margin-bottom:.5rem;color:#e0e0e0}
      .coach-insight p{color:#888;font-size:.85rem;line-height:1.6}
      .coach-section-title{font-size:1.1rem;font-weight:700;color:#e0e0e0;margin:1.5rem 0 1rem;padding-bottom:.5rem;border-bottom:1px solid #222}
      .stat-highlight{display:inline-block;background:rgba(213,192,139,.1);color:#D5C08B;padding:2px 8px;border-radius:4px;font-weight:700;font-size:.85rem}
      .pos-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:1rem;margin:1rem 0}
      .pos-card{background:#111;border:1px solid #222;border-radius:10px;padding:1.2rem;text-align:center}
      .pos-card .zone{font-size:.8rem;color:#888;text-transform:uppercase;letter-spacing:1px}
      .pos-card .kills{font-size:2rem;font-weight:800;color:#D5C08B;margin:.5rem 0}
      .pos-card .deaths{font-size:.9rem;color:#f87171}
      .pos-card .kd{font-size:.85rem;color:#4ade80;margin-top:.3rem}
    `;
    document.head.appendChild(style);
  }

  // ===== Radar Chart (Canvas) =====
  function drawRadar(canvas, data) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const cx = w/2, cy = h/2, r = Math.min(w,h)/2 - 40;
    const labels = Object.keys(data);
    const values = Object.values(data);
    const n = labels.length;
    const angleStep = (Math.PI * 2) / n;

    ctx.clearRect(0, 0, w, h);

    // Grid rings
    for (let ring = 1; ring <= 5; ring++) {
      ctx.beginPath();
      const rr = (r * ring) / 5;
      for (let i = 0; i <= n; i++) {
        const a = i * angleStep - Math.PI/2;
        const x = cx + rr * Math.cos(a);
        const y = cy + rr * Math.sin(a);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Axes
    for (let i = 0; i < n; i++) {
      const a = i * angleStep - Math.PI/2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + r * Math.cos(a), cy + r * Math.sin(a));
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.stroke();
    }

    // Data polygon
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const idx = i % n;
      const a = idx * angleStep - Math.PI/2;
      const v = Math.min(values[idx] / 100, 1);
      const x = cx + r * v * Math.cos(a);
      const y = cy + r * v * Math.sin(a);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.fillStyle = 'rgba(213,192,139,0.15)';
    ctx.fill();
    ctx.strokeStyle = '#D5C08B';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Data points + labels
    for (let i = 0; i < n; i++) {
      const a = i * angleStep - Math.PI/2;
      const v = Math.min(values[i] / 100, 1);
      const x = cx + r * v * Math.cos(a);
      const y = cy + r * v * Math.sin(a);

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#D5C08B';
      ctx.fill();

      // Label
      const lx = cx + (r + 24) * Math.cos(a);
      const ly = cy + (r + 24) * Math.sin(a);
      ctx.fillStyle = '#888';
      ctx.font = '11px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(labels[i], lx, ly);

      // Value
      ctx.fillStyle = '#e0e0e0';
      ctx.font = 'bold 11px system-ui';
      ctx.fillText(Math.round(values[i]), lx, ly + 14);
    }
  }

  // ===== Build Coach UI =====
  function buildCoachUI(resultData) {
    const container = document.getElementById('coach-container');
    if (!container) return;

    // Tab bar
    const tabBar = document.createElement('div');
    tabBar.className = 'coach-tabs';
    TAB_CONFIG.forEach(tab => {
      const btn = document.createElement('button');
      btn.className = 'coach-tab' + (tab.id === 'overview' ? ' active' : '');
      btn.dataset.tab = tab.id;
      btn.innerHTML = '<span class="coach-tab-icon">' + tab.icon + '</span>' + tab.label;
      btn.onclick = () => switchTab(tab.id);
      tabBar.appendChild(btn);
    });
    container.prepend(tabBar);

    // Build panels
    buildPerformancePanel(container, resultData);
    buildCoachPanel(container, resultData);
    buildPositionsPanel(container, resultData);

    // Wrap existing content as overview panel
    const existingContent = container.querySelector('.results-content, .match-results');
    if (existingContent) {
      existingContent.classList.add('coach-panel', 'active');
      existingContent.dataset.panel = 'overview';
    }
  }

  function switchTab(tabId) {
    document.querySelectorAll('.coach-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabId));
    document.querySelectorAll('.coach-panel').forEach(p => p.classList.toggle('active', p.dataset.panel === tabId));
  }

  function buildPerformancePanel(container, data) {
    const panel = document.createElement('div');
    panel.className = 'coach-panel';
    panel.dataset.panel = 'performance';

    const player = data.player_stats || data.players?.[0] || {};
    const radarData = {
      'Aim': Math.min((player.headshot_pct || 40) * 1.8, 100),
      'Impact': Math.min((player.hltv_rating || 1.0) * 65, 100),
      'Utility': Math.min((player.utility_damage || 50) / 2, 100),
      'Trade': Math.min((player.trade_kill_pct || 30) * 2, 100),
      'Survival': Math.min((1 - (player.deaths || 15) / 30) * 100, 100),
      'Economy': Math.min((player.eco_rating || 60), 100)
    };

    panel.innerHTML = '<div class="coach-section-title">Player Radar</div>' +
      '<div class="radar-container"><canvas class="radar-canvas" width="360" height="360"></canvas></div>' +
      '<div class="coach-section-title">CT vs T Side Breakdown</div>' +
      '<div class="side-comparison">' +
        '<div class="side-card side-ct"><h3><span style="color:#5c9fd4">\u25CF</span> Counter-Terrorist</h3>' +
          buildSideStats(player, 'ct') +
        '</div>' +
        '<div class="side-card side-t"><h3><span style="color:#dea74c">\u25CF</span> Terrorist</h3>' +
          buildSideStats(player, 't') +
        '</div>' +
      '</div>';

    container.appendChild(panel);

    // Draw radar after DOM insertion
    requestAnimationFrame(() => {
      const canvas = panel.querySelector('.radar-canvas');
      if (canvas) drawRadar(canvas, radarData);
    });
  }

  function buildSideStats(player, side) {
    const kills = player[side + '_kills'] || player['kills_' + side] || Math.round((player.kills || 20) / 2);
    const deaths = player[side + '_deaths'] || player['deaths_' + side] || Math.round((player.deaths || 15) / 2);
    const adr = player[side + '_adr'] || player['adr_' + side] || Math.round((player.adr || 75) * (side === 'ct' ? 0.95 : 1.05));
    const kd = (kills / Math.max(deaths, 1)).toFixed(2);
    return '<div class="side-stat"><span class="label">Kills</span><span class="val">' + kills + '</span></div>' +
      '<div class="side-stat"><span class="label">Deaths</span><span class="val">' + deaths + '</span></div>' +
      '<div class="side-stat"><span class="label">K/D</span><span class="val">' + kd + '</span></div>' +
      '<div class="side-stat"><span class="label">ADR</span><span class="val">' + adr + '</span></div>';
  }

  function buildCoachPanel(container, data) {
    const panel = document.createElement('div');
    panel.className = 'coach-panel';
    panel.dataset.panel = 'coach';

    const coaching = data.coaching || data.ai_coaching || {};
    const insights = coaching.insights || coaching.tips || [];

    let html = '<div class="coach-section-title">AI Coaching Insights</div>';

    if (insights.length > 0) {
      insights.forEach(insight => {
        const type = (insight.type || insight.sentiment || 'neutral').toLowerCase();
        const cls = type.includes('pos') || type.includes('good') ? 'positive' :
                    type.includes('neg') || type.includes('bad') ? 'negative' : 'neutral';
        html += '<div class="coach-insight ' + cls + '">' +
          '<h4>' + (insight.title || insight.category || 'Insight') + '</h4>' +
          '<p>' + (insight.message || insight.text || insight.detail || '') + '</p></div>';
      });
    } else {
      html += '<div class="coach-insight neutral"><h4>Analysis Complete</h4>' +
        '<p>Upload a demo file to receive personalized AI coaching insights powered by Claude. ' +
        'The AI coach will analyze your gameplay round-by-round and provide specific, actionable advice.</p></div>';
      // Add sample insights for the UI
      html += '<div class="coach-insight positive"><h4>Trade Efficiency</h4>' +
        '<p>Your trade kill percentage and timing show good team coordination. Continue supporting teammates in duels.</p></div>';
      html += '<div class="coach-insight negative"><h4>Positioning</h4>' +
        '<p>Consider varying your defensive positions more frequently. Predictable setups make you easier to counter-strat.</p></div>';
    }

    panel.innerHTML = html;
    container.appendChild(panel);
  }

  function buildPositionsPanel(container, data) {
    const panel = document.createElement('div');
    panel.className = 'coach-panel';
    panel.dataset.panel = 'positions';

    const positions = data.positions || data.position_data || {};
    const zones = positions.zones || positions.kill_zones || [];

    let html = '<div class="coach-section-title">Position Intelligence</div>';

    if (zones.length > 0) {
      html += '<div class="pos-grid">';
      zones.slice(0, 8).forEach(z => {
        const kd = ((z.kills || 0) / Math.max(z.deaths || 1, 1)).toFixed(1);
        html += '<div class="pos-card">' +
          '<div class="zone">' + (z.name || z.zone || 'Unknown') + '</div>' +
          '<div class="kills">' + (z.kills || 0) + '</div>' +
          '<div class="deaths">' + (z.deaths || 0) + ' deaths</div>' +
          '<div class="kd">' + kd + ' K/D</div></div>';
      });
      html += '</div>';
    } else {
      html += '<div class="pos-grid">';
      ['A Site', 'B Site', 'Mid', 'Connector', 'Long', 'Short'].forEach(zone => {
        html += '<div class="pos-card"><div class="zone">' + zone + '</div>' +
          '<div class="kills">-</div><div class="deaths">Upload demo</div>' +
          '<div class="kd">for data</div></div>';
      });
      html += '</div>';
    }

    panel.innerHTML = html;
    container.appendChild(panel);
  }

  // ===== Initialize =====
  function init() {
    injectStyles();

    // Try to find or create the coach container
    const resultsArea = document.getElementById('results') ||
                        document.getElementById('match-results') ||
                        document.querySelector('.results-section') ||
                        document.querySelector('[data-results]');

    if (resultsArea && !document.getElementById('coach-container')) {
      const coachDiv = document.createElement('div');
      coachDiv.id = 'coach-container';

      // Move existing results content into coach container
      while (resultsArea.firstChild) {
        coachDiv.appendChild(resultsArea.firstChild);
      }
      resultsArea.appendChild(coachDiv);
    }

    // Listen for analysis results
    window.addEventListener('opensight:results', function(e) {
      const data = e.detail || {};
      buildCoachUI(data);
    });

    // Also check if results are already loaded
    if (window.__opensightResults) {
      buildCoachUI(window.__opensightResults);
    }

    // Auto-init coach UI with empty state after a delay
    setTimeout(() => {
      const container = document.getElementById('coach-container');
      if (container && !container.querySelector('.coach-tabs')) {
        buildCoachUI({});
      }
    }, 2000);
  }

  // Expose for external use
  window.OpenSightCoach = { init, buildCoachUI, drawRadar, switchTab };

  // Auto-init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
