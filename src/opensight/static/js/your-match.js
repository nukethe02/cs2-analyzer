/**
 * OpenSight - Your Match Dashboard
 *
 * Implements Leetify-style personal performance tracking with:
 * - Match Identity persona card
 * - Top 5 Stats with progress bars
 * - This Match vs Your 30 Match Average comparison
 */

class YourMatchDashboard {
  constructor(containerId = "your-match-container") {
    this.container = document.getElementById(containerId);
    this.data = null;
    this.demoId = null;
    this.steamId = null;
  }

  /**
   * Load Your Match data from the API
   * @param {string} demoId - Demo hash or job ID
   * @param {string} steamId - Player's Steam ID (17 digits)
   */
  async load(demoId, steamId) {
    this.demoId = demoId;
    this.steamId = steamId;

    try {
      const res = await fetch(`/api/your-match/${demoId}/${steamId}`);

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || "Failed to load Your Match data");
      }

      this.data = await res.json();
      this.render();
    } catch (err) {
      console.error("Your Match load error:", err);
      this.renderError(err.message);
    }
  }

  /**
   * Render the complete Your Match dashboard
   */
  render() {
    if (!this.container || !this.data) return;

    const { persona, top_5, comparison, match_count } = this.data;

    this.container.innerHTML = `
      <div class="your-match-dashboard">
        <div class="your-match-header">
          <h2>Your Match</h2>
          <span class="match-count">${match_count > 0 ? `Based on ${match_count} matches` : "First match tracked"}</span>
        </div>

        <div class="your-match-content">
          <!-- Left Panel: Persona Card -->
          <div class="persona-card" style="--persona-color: ${this.esc(persona.color)}">
            <div class="persona-ring">
              <svg viewBox="0 0 100 100" class="confidence-ring">
                <circle cx="50" cy="50" r="45" fill="none" stroke="#2a2a2a" stroke-width="6"/>
                <circle cx="50" cy="50" r="45" fill="none" stroke="${this.esc(persona.color)}" stroke-width="6"
                  stroke-dasharray="${Math.round(persona.confidence * 283)} 283"
                  stroke-linecap="round" transform="rotate(-90 50 50)"/>
              </svg>
              <div class="persona-icon">${this.getPersonaIcon(persona.icon)}</div>
            </div>

            <div class="persona-label">MATCH IDENTITY | YOUR TOP 5 STATS</div>
            <h3 class="persona-name">${this.esc(persona.name)}</h3>
            <p class="persona-desc">${this.esc(persona.description)}</p>

            <div class="top-5-stats">
              ${top_5.map((stat, idx) => this.renderTopStat(stat, idx + 1)).join("")}
            </div>
          </div>

          <!-- Right Panel: Comparison Table -->
          <div class="comparison-panel">
            <div class="comparison-header">
              <span class="col-label"></span>
              <span class="col-this">This Match</span>
              <span class="col-avg">YOUR ${match_count || 30} MATCH AVERAGE</span>
            </div>

            <div class="comparison-rows">
              ${comparison.map((row) => this.renderComparisonRow(row)).join("")}
            </div>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * Render a single top stat row
   */
  renderTopStat(stat, rank) {
    const categoryClass = stat.category.toLowerCase().replace(/\s+/g, "-");

    return `
      <div class="stat-row">
        <span class="stat-rank">#${rank}</span>
        <div class="stat-info">
          <span class="stat-name">${this.esc(stat.name)}</span>
          <span class="stat-category ${categoryClass}">${this.esc(stat.category)}</span>
        </div>
        <div class="stat-bar-container">
          <div class="stat-bar" style="width: ${Math.min(100, stat.percentile)}%"></div>
        </div>
        <span class="stat-value">${this.esc(stat.formatted_value)}</span>
      </div>
    `;
  }

  /**
   * Render a comparison table row
   */
  renderComparisonRow(row) {
    const diffClass = row.is_better ? "better" : "worse";
    const diffSign = row.diff >= 0 ? "+" : "";
    const diffText =
      row.diff_percent !== null
        ? `${diffSign}${row.diff.toFixed(1)} (${diffSign}${row.diff_percent.toFixed(0)}%)`
        : `${diffSign}${row.diff.toFixed(1)}`;

    return `
      <div class="comparison-row ${diffClass}">
        <span class="row-label">${this.esc(row.label)}</span>
        <span class="row-this">${this.esc(row.formatted_this)}</span>
        <span class="row-avg">${this.esc(row.formatted_avg)}</span>
        <span class="row-diff">${diffText}</span>
        <button class="row-expand" aria-label="Expand details">
          <svg width="12" height="12" viewBox="0 0 12 12">
            <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" fill="none" stroke-width="1.5"/>
          </svg>
        </button>
      </div>
    `;
  }

  /**
   * Get icon for persona type
   */
  getPersonaIcon(iconType) {
    const icons = {
      trade:
        '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7.5 21L3 16.5L7.5 12L9 13.5L6.5 16H13V18H6.5L9 20.5L7.5 21ZM16.5 12L21 7.5L16.5 3L15 4.5L17.5 7H11V9H17.5L15 11.5L16.5 12Z"/></svg>',
      entry:
        '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M4 12L12 4L20 12H15V20H9V12H4Z"/></svg>',
      anchor:
        '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C13.1 2 14 2.9 14 4C14 5.1 13.1 6 12 6C10.9 6 10 5.1 10 4C10 2.9 10.9 2 12 2ZM11 7H13V9H18V11H13V22H11V11H6V9H11V7Z"/></svg>',
      utility:
        '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L2 7L12 12L22 7L12 2ZM2 17L12 22L22 17L12 12L2 17Z"/></svg>',
      headshot:
        '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6" fill="#1a1a1a"/><circle cx="12" cy="12" r="2"/></svg>',
      survival:
        '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 21.35L10.55 20.03C5.4 15.36 2 12.27 2 8.5C2 5.41 4.42 3 7.5 3C9.24 3 10.91 3.81 12 5.08C13.09 3.81 14.76 3 16.5 3C19.58 3 22 5.41 22 8.5C22 12.27 18.6 15.36 13.45 20.03L12 21.35Z"/></svg>',
      damage:
        '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L4 5V11.09C4 16.14 7.41 20.85 12 22C16.59 20.85 20 16.14 20 11.09V5L12 2ZM18 11.09C18 15.09 15.45 18.79 12 19.92C8.55 18.79 6 15.1 6 11.09V6.39L12 4.14L18 6.39V11.09Z"/></svg>',
      flash:
        '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 2V13H10V22L17 10H13L17 2H7Z"/></svg>',
      kills:
        '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L14.4 8.4L21 9.6L16.5 14.4L17.6 21L12 17.6L6.4 21L7.5 14.4L3 9.6L9.6 8.4L12 2Z"/></svg>',
      default:
        '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"/></svg>',
    };

    return icons[iconType] || icons.default;
  }

  /**
   * Render error state
   */
  renderError(message) {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="your-match-error">
        <h3>Unable to load Your Match data</h3>
        <p>${this.esc(message)}</p>
        <button onclick="yourMatchDashboard.load('${this.demoId}', '${this.steamId}')">
          Retry
        </button>
      </div>
    `;
  }

  /**
   * HTML escape utility
   */
  esc(text) {
    if (text === null || text === undefined) return "";
    const div = document.createElement("div");
    div.textContent = String(text);
    return div.innerHTML;
  }

  /**
   * Store a match result for tracking
   */
  async storeMatch(steamId, demoHash, playerStats, mapName = null, result = null) {
    try {
      const res = await fetch("/api/your-match/store", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          steam_id: steamId,
          demo_hash: demoHash,
          player_stats: playerStats,
          map_name: mapName,
          result: result,
        }),
      });

      if (!res.ok) {
        const error = await res.json();
        console.error("Failed to store match:", error);
        return false;
      }

      return true;
    } catch (err) {
      console.error("Store match error:", err);
      return false;
    }
  }

  /**
   * Get player baselines
   */
  async getBaselines(steamId) {
    try {
      const res = await fetch(`/api/your-match/baselines/${steamId}`);
      if (!res.ok) return null;
      return await res.json();
    } catch (err) {
      console.error("Get baselines error:", err);
      return null;
    }
  }

  /**
   * Get player's match history
   */
  async getHistory(steamId, limit = 30) {
    try {
      const res = await fetch(`/api/your-match/history/${steamId}?limit=${limit}`);
      if (!res.ok) return null;
      return await res.json();
    } catch (err) {
      console.error("Get history error:", err);
      return null;
    }
  }

  /**
   * Get player's persona
   */
  async getPersona(steamId) {
    try {
      const res = await fetch(`/api/your-match/persona/${steamId}`);
      if (!res.ok) return null;
      return await res.json();
    } catch (err) {
      console.error("Get persona error:", err);
      return null;
    }
  }
}

// Global instance
let yourMatchDashboard = null;

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("your-match-container");
  if (container) {
    yourMatchDashboard = new YourMatchDashboard("your-match-container");
  }
});

// Export for module usage
if (typeof module !== "undefined" && module.exports) {
  module.exports = { YourMatchDashboard };
}
