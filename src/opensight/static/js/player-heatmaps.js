/**
 * Per-Player Positioning Heatmaps
 *
 * Renders individual player heatmaps showing:
 * - Presence (where they spend time)
 * - Kills (where they get kills)
 * - Deaths (where they die)
 * - Early/Late round positioning
 * - Favorite/Danger zones
 *
 * This is more useful than Leetify's team-aggregate heatmaps
 * for scouting specific opponents.
 */

class PlayerHeatmaps {
    // Heatmap configuration
    static HEATMAP_SIZE = 256; // Canvas size in pixels (rendered from 64x64 grid)
    static GRID_SIZE = 64;

    // Color gradients
    static COLORS = {
        presence: ['#0d47a1', '#2196f3', '#64b5f6', '#90caf9', '#e3f2fd'], // Blue
        kills: ['#1b5e20', '#4caf50', '#81c784', '#a5d6a7', '#c8e6c9'], // Green
        deaths: ['#b71c1c', '#f44336', '#e57373', '#ef9a9a', '#ffcdd2'], // Red
        early: ['#e65100', '#ff9800', '#ffb74d', '#ffcc80', '#ffe0b2'], // Orange
        late: ['#4a148c', '#9c27b0', '#ba68c8', '#ce93d8', '#e1bee7'], // Purple
    };

    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.players = {};
        this.selectedPlayer = null;
        this.selectedType = 'presence';
        this.canvas = null;
        this.ctx = null;
        this.mapName = '';
        this.radarImage = null;
    }

    /**
     * Initialize with heatmap data from API
     * @param {Object} data - Response from /api/positioning/{job_id}/all
     */
    init(data) {
        this.players = data.players || {};
        this.mapName = (data.map_name || '').toLowerCase();

        this.render();
        this.loadRadarImage();

        // Select first player by default
        const steamIds = Object.keys(this.players);
        if (steamIds.length > 0) {
            this.selectPlayer(steamIds[0]);
        }
    }

    /**
     * Load radar background image
     */
    loadRadarImage() {
        const _RADAR = 'https://raw.githubusercontent.com/2mlml/cs2-radar-images/master';
        const radarUrls = {
            'de_dust2': _RADAR + '/de_dust2.png',
            'de_mirage': _RADAR + '/de_mirage.png',
            'de_inferno': _RADAR + '/de_inferno.png',
            'de_nuke': _RADAR + '/de_nuke.png',
            'de_overpass': _RADAR + '/de_overpass.png',
            'de_vertigo': _RADAR + '/de_vertigo.png',
            'de_ancient': _RADAR + '/de_ancient.png',
            'de_anubis': _RADAR + '/de_anubis.png',
        };

        const url = radarUrls[this.mapName];
        if (url) {
            this.radarImage = new Image();
            this.radarImage.crossOrigin = 'anonymous';
            this.radarImage.onload = () => this.drawHeatmap();
            this.radarImage.onerror = () => { this.radarImage = null; this.drawHeatmap(); };
            this.radarImage.src = url;
        }
    }

    /**
     * Render the full UI
     */
    render() {
        if (!this.container) return;

        this.container.innerHTML = `
            <div class="player-heatmaps-container">
                <div class="heatmap-header">
                    <h3>📍 Player Positioning Heatmaps</h3>
                    <p class="heatmap-subtitle">Individual player tendencies - better than Leetify's team aggregate</p>
                </div>

                <div class="heatmap-controls">
                    <div class="control-group">
                        <label>Player</label>
                        <select id="player-heatmap-select" class="heatmap-select">
                            ${this.renderPlayerOptions()}
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Heatmap Type</label>
                        <div class="heatmap-type-buttons">
                            <button class="heatmap-type-btn active" data-type="presence" title="Where player spends time">Presence</button>
                            <button class="heatmap-type-btn" data-type="kills" title="Where player gets kills">Kills</button>
                            <button class="heatmap-type-btn" data-type="deaths" title="Where player dies">Deaths</button>
                            <button class="heatmap-type-btn" data-type="early" title="First 30 seconds of round">Early</button>
                            <button class="heatmap-type-btn" data-type="late" title="After 30 seconds">Late</button>
                        </div>
                    </div>
                </div>

                <div class="heatmap-content">
                    <div class="heatmap-canvas-container">
                        <canvas id="player-heatmap-canvas" width="${PlayerHeatmaps.HEATMAP_SIZE}" height="${PlayerHeatmaps.HEATMAP_SIZE}"></canvas>
                        <div class="heatmap-legend">
                            <span class="legend-low">Low</span>
                            <div class="legend-gradient" id="heatmap-gradient"></div>
                            <span class="legend-high">High</span>
                        </div>
                    </div>

                    <div class="heatmap-zones-panel">
                        <div class="zones-section">
                            <h4>⭐ Favorite Zones</h4>
                            <ul id="favorite-zones-list" class="zone-list"></ul>
                        </div>
                        <div class="zones-section">
                            <h4>⚠️ Danger Zones</h4>
                            <ul id="danger-zones-list" class="zone-list danger"></ul>
                        </div>
                        <div class="zones-section">
                            <h4>📊 Stats</h4>
                            <div id="player-heatmap-stats" class="heatmap-stats"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Get canvas context
        this.canvas = document.getElementById('player-heatmap-canvas');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;

        // Attach event listeners
        this.attachListeners();

        // Update gradient
        this.updateGradient();
    }

    /**
     * Render player select options
     */
    renderPlayerOptions() {
        return Object.entries(this.players)
            .map(([steamId, data]) =>
                `<option value="${this.esc(steamId)}">${this.esc(data.player_name || steamId)}</option>`
            )
            .join('');
    }

    /**
     * Attach event listeners
     */
    attachListeners() {
        // Player select
        const select = document.getElementById('player-heatmap-select');
        if (select) {
            select.addEventListener('change', (e) => this.selectPlayer(e.target.value));
        }

        // Type buttons
        const buttons = document.querySelectorAll('.heatmap-type-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                buttons.forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                this.selectedType = e.target.dataset.type;
                this.updateGradient();
                this.drawHeatmap();
            });
        });
    }

    /**
     * Select a player
     * @param {string} steamId
     */
    selectPlayer(steamId) {
        this.selectedPlayer = steamId;
        this.drawHeatmap();
        this.updateZones();
        this.updateStats();
    }

    /**
     * Get heatmap data for selected player and type
     */
    getHeatmapData() {
        if (!this.selectedPlayer || !this.players[this.selectedPlayer]) {
            return null;
        }

        const player = this.players[this.selectedPlayer];
        const typeMap = {
            presence: 'presence_heatmap',
            kills: 'kills_heatmap',
            deaths: 'deaths_heatmap',
            early: 'early_round_heatmap',
            late: 'late_round_heatmap',
        };

        return player[typeMap[this.selectedType]] || null;
    }

    /**
     * Draw heatmap on canvas
     */
    drawHeatmap() {
        if (!this.ctx || !this.canvas) return;

        const data = this.getHeatmapData();
        const width = this.canvas.width;
        const height = this.canvas.height;
        const cellWidth = width / PlayerHeatmaps.GRID_SIZE;
        const cellHeight = height / PlayerHeatmaps.GRID_SIZE;

        // Clear canvas
        this.ctx.clearRect(0, 0, width, height);

        // Draw radar background if available
        if (this.radarImage) {
            this.ctx.globalAlpha = 0.5;
            this.ctx.drawImage(this.radarImage, 0, 0, width, height);
            this.ctx.globalAlpha = 1.0;
        } else {
            // Dark background
            this.ctx.fillStyle = '#1a1a1a';
            this.ctx.fillRect(0, 0, width, height);
        }

        if (!data || data.length === 0) {
            this.drawNoDataMessage();
            return;
        }

        // Get colors for current type
        const colors = PlayerHeatmaps.COLORS[this.selectedType] || PlayerHeatmaps.COLORS.presence;

        // Draw heatmap cells
        for (let y = 0; y < PlayerHeatmaps.GRID_SIZE; y++) {
            for (let x = 0; x < PlayerHeatmaps.GRID_SIZE; x++) {
                const value = data[x]?.[y] || 0;
                if (value > 0) {
                    const color = this.getHeatColor(value, colors);
                    this.ctx.fillStyle = color;
                    this.ctx.globalAlpha = Math.min(0.8, value * 0.8 + 0.2);
                    this.ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth + 1, cellHeight + 1);
                }
            }
        }

        this.ctx.globalAlpha = 1.0;
    }

    /**
     * Get color for heatmap value
     * @param {number} value - 0 to 1
     * @param {string[]} colors - Color gradient array
     */
    getHeatColor(value, colors) {
        const idx = Math.min(Math.floor(value * colors.length), colors.length - 1);
        return colors[colors.length - 1 - idx]; // Reverse for high = dark
    }

    /**
     * Draw "no data" message on canvas
     */
    drawNoDataMessage() {
        this.ctx.fillStyle = '#666';
        this.ctx.font = '14px system-ui, sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('No position data available', this.canvas.width / 2, this.canvas.height / 2);
    }

    /**
     * Update zone lists
     */
    updateZones() {
        const player = this.players[this.selectedPlayer];
        if (!player) return;

        // Favorite zones
        const favList = document.getElementById('favorite-zones-list');
        if (favList) {
            favList.innerHTML = (player.favorite_zones || [])
                .map((zone, i) => `<li><span class="zone-rank">${i + 1}</span> ${this.esc(zone)}</li>`)
                .join('') || '<li class="no-data">No data</li>';
        }

        // Danger zones
        const dangerList = document.getElementById('danger-zones-list');
        if (dangerList) {
            dangerList.innerHTML = (player.danger_zones || [])
                .map((zone, i) => `<li><span class="zone-rank">${i + 1}</span> ${this.esc(zone)}</li>`)
                .join('') || '<li class="no-data">No data</li>';
        }
    }

    /**
     * Update stats panel
     */
    updateStats() {
        const player = this.players[this.selectedPlayer];
        if (!player) return;

        const statsEl = document.getElementById('player-heatmap-stats');
        if (!statsEl) return;

        statsEl.innerHTML = `
            <div class="stat-row">
                <span class="stat-label">Positions tracked</span>
                <span class="stat-value">${player.total_positions || 0}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total kills</span>
                <span class="stat-value">${player.total_kills || 0}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total deaths</span>
                <span class="stat-value">${player.total_deaths || 0}</span>
            </div>
        `;
    }

    /**
     * Update legend gradient
     */
    updateGradient() {
        const gradientEl = document.getElementById('heatmap-gradient');
        if (!gradientEl) return;

        const colors = PlayerHeatmaps.COLORS[this.selectedType] || PlayerHeatmaps.COLORS.presence;
        gradientEl.style.background = `linear-gradient(to right, ${colors.join(', ')})`;
    }

    /**
     * Escape HTML to prevent XSS
     * @param {string} str
     */
    esc(str) {
        if (typeof str !== 'string') return str;
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }
}

// Export for use
window.PlayerHeatmaps = PlayerHeatmaps;
