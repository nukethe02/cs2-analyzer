/**
 * Map Zones Module for CS2 Demo Analyzer
 *
 * Provides Leetify-style zone-based K/D visualization with:
 * - Colored zone overlays based on K/D ratio
 * - Multi-filter support (side, phase, economy, player)
 * - Click-to-show kill/death positions
 * - Zone performance sidebar
 */

// ============================================================================
// MAP ZONE DEFINITIONS (Fallback if not provided by backend)
// ============================================================================

const DEFAULT_MAP_ZONES = {
    "de_ancient": {
        // === SPAWN AREAS ===
        "T Spawn": { bounds: [[-2400, -900], [-1400, -900], [-1400, 200], [-2400, 200]], type: "spawn" },
        "CT Spawn": { bounds: [[600, 200], [1500, 200], [1500, 1000], [600, 1000]], type: "spawn" },
        // === T SIDE APPROACH (Outside / Alley) ===
        "Outside": { bounds: [[-2400, 200], [-1400, 200], [-1400, 800], [-2400, 800]], type: "route" },
        "T Alley": { bounds: [[-1400, -400], [-1200, -400], [-1200, 400], [-1400, 400]], type: "route" },
        // === A SITE AREA ===
        "A Site": { bounds: [[-600, 650], [350, 650], [350, 1450], [-600, 1450]], type: "bombsite" },
        "A Main": { bounds: [[-1300, 150], [-400, 150], [-400, 700], [-1300, 700]], type: "route" },
        "Elbow": { bounds: [[-400, 400], [100, 400], [100, 700], [-400, 700]], type: "choke" },
        "A Default": { bounds: [[-200, 1000], [300, 1000], [300, 1400], [-200, 1400]], type: "position" },
        // === MID AREA ===
        "Donut": { bounds: [[-600, -300], [300, -300], [300, 200], [-600, 200]], type: "route" },
        "Mid": { bounds: [[-300, -800], [700, -800], [700, -250], [-300, -250]], type: "mid" },
        "Temple": { bounds: [[-1300, -350], [-500, -350], [-500, 200], [-1300, 200]], type: "route" },
        // === B SITE AREA ===
        "B Site": { bounds: [[800, -500], [1700, -500], [1700, 400], [800, 400]], type: "bombsite" },
        "B Main": { bounds: [[500, -1200], [1300, -1200], [1300, -450], [500, -450]], type: "route" },
        "B Ramp": { bounds: [[600, -500], [900, -500], [900, 100], [600, 100]], type: "choke" },
        "Ruins": { bounds: [[1400, -600], [1800, -600], [1800, 100], [1400, 100]], type: "position" },
        "Back of B": { bounds: [[1300, 100], [1700, 100], [1700, 500], [1300, 500]], type: "position" },
        // === CT SIDE / ROTATIONS ===
        "Cave": { bounds: [[100, 650], [800, 650], [800, 1300], [100, 1300]], type: "route" },
        "House": { bounds: [[350, 300], [700, 300], [700, 700], [350, 700]], type: "route" },
        "CT": { bounds: [[700, 650], [1100, 650], [1100, 1100], [700, 1100]], type: "route" },
        "Ramp": { bounds: [[100, -300], [700, -300], [700, 400], [100, 400]], type: "route" },
        // === WATER / SEWER AREA ===
        "Water": { bounds: [[-400, -1200], [550, -1200], [550, -750], [-400, -750]], type: "route" },
        "Tunnel": { bounds: [[400, -800], [700, -800], [700, -450], [400, -450]], type: "route" },
    },
    "de_dust2": {
        "T Spawn": { bounds: [[-710, -1780], [540, -1780], [540, -2820], [-710, -2820]], type: "spawn" },
        "CT Spawn": { bounds: [[800, 2430], [1850, 2430], [1850, 3120], [800, 3120]], type: "spawn" },
        "Long A": { bounds: [[180, -510], [1380, -510], [1380, 280], [180, 280]], type: "route" },
        "Long Doors": { bounds: [[-100, -510], [180, -510], [180, -100], [-100, -100]], type: "choke" },
        "A Site": { bounds: [[670, 1510], [1480, 1510], [1480, 2430], [670, 2430]], type: "bombsite" },
        "A Ramp": { bounds: [[470, 280], [1050, 280], [1050, 1510], [470, 1510]], type: "route" },
        "Short A": { bounds: [[-1350, 1550], [-650, 1550], [-650, 2250], [-1350, 2250]], type: "route" },
        "Catwalk": { bounds: [[-650, 1550], [470, 1550], [470, 2250], [-650, 2250]], type: "route" },
        "Mid": { bounds: [[-1480, -20], [-400, -20], [-400, 1200], [-1480, 1200]], type: "mid" },
        "Mid Doors": { bounds: [[-1480, -580], [-400, -580], [-400, -20], [-1480, -20]], type: "choke" },
        "Lower Tunnels": { bounds: [[-2040, -1180], [-1480, -1180], [-1480, -580], [-2040, -580]], type: "route" },
        "Upper Tunnels": { bounds: [[-1900, 280], [-1500, 280], [-1500, 960], [-1900, 960]], type: "route" },
        "B Site": { bounds: [[-2350, 1200], [-1500, 1200], [-1500, 2480], [-2350, 2480]], type: "bombsite" },
        "B Window": { bounds: [[-1500, 1700], [-1150, 1700], [-1150, 2150], [-1500, 2150]], type: "position" },
        "Pit": { bounds: [[1480, 750], [1900, 750], [1900, 1510], [1480, 1510]], type: "position" },
        "Goose": { bounds: [[1480, 2430], [1900, 2430], [1900, 2850], [1480, 2850]], type: "position" },
    },
    "de_mirage": {
        "T Spawn": { bounds: [[-2900, -2200], [-2100, -2200], [-2100, -1500], [-2900, -1500]], type: "spawn" },
        "CT Spawn": { bounds: [[600, 1100], [1400, 1100], [1400, 1700], [600, 1700]], type: "spawn" },
        "A Site": { bounds: [[-400, 800], [600, 800], [600, 1600], [-400, 1600]], type: "bombsite" },
        "A Ramp": { bounds: [[-1200, 400], [-400, 400], [-400, 1000], [-1200, 1000]], type: "route" },
        "Palace": { bounds: [[-1800, 600], [-1200, 600], [-1200, 1200], [-1800, 1200]], type: "route" },
        "Tetris": { bounds: [[-400, 300], [200, 300], [200, 800], [-400, 800]], type: "position" },
        "Jungle": { bounds: [[200, 300], [800, 300], [800, 800], [200, 800]], type: "position" },
        "Connector": { bounds: [[200, -400], [800, -400], [800, 300], [200, 300]], type: "route" },
        "Mid": { bounds: [[-600, -1000], [200, -1000], [200, -200], [-600, -200]], type: "mid" },
        "Top Mid": { bounds: [[-1200, -1200], [-600, -1200], [-600, -600], [-1200, -600]], type: "mid" },
        "Underpass": { bounds: [[200, -1200], [800, -1200], [800, -600], [200, -600]], type: "route" },
        "B Short": { bounds: [[-1200, -600], [-600, -600], [-600, -200], [-1200, -200]], type: "route" },
        "B Site": { bounds: [[-2200, -800], [-1400, -800], [-1400, 0], [-2200, 0]], type: "bombsite" },
        "B Apartments": { bounds: [[-2800, -1200], [-2200, -1200], [-2200, -400], [-2800, -400]], type: "route" },
        "Market": { bounds: [[800, -200], [1400, -200], [1400, 400], [800, 400]], type: "route" },
        "Window": { bounds: [[600, 400], [1000, 400], [1000, 800], [600, 800]], type: "position" },
    },
    "de_inferno": {
        "T Spawn": { bounds: [[-400, -1400], [400, -1400], [400, -800], [-400, -800]], type: "spawn" },
        "CT Spawn": { bounds: [[2000, 1400], [2700, 1400], [2700, 2000], [2000, 2000]], type: "spawn" },
        "A Site": { bounds: [[1700, 200], [2400, 200], [2400, 900], [1700, 900]], type: "bombsite" },
        "A Long": { bounds: [[1200, -200], [1800, -200], [1800, 400], [1200, 400]], type: "route" },
        "Apartments": { bounds: [[600, 0], [1200, 0], [1200, 600], [600, 600]], type: "route" },
        "Pit": { bounds: [[2400, 200], [2900, 200], [2900, 700], [2400, 700]], type: "position" },
        "Library": { bounds: [[1700, 900], [2200, 900], [2200, 1400], [1700, 1400]], type: "position" },
        "Arch": { bounds: [[1200, 900], [1700, 900], [1700, 1400], [1200, 1400]], type: "route" },
        "Mid": { bounds: [[400, -600], [1000, -600], [1000, 0], [400, 0]], type: "mid" },
        "Top Mid": { bounds: [[-200, -600], [400, -600], [400, 0], [-200, 0]], type: "mid" },
        "Banana": { bounds: [[-800, 600], [-200, 600], [-200, 1400], [-800, 1400]], type: "route" },
        "B Site": { bounds: [[-1200, 1800], [-400, 1800], [-400, 2600], [-1200, 2600]], type: "bombsite" },
        "CT": { bounds: [[1000, 1400], [1600, 1400], [1600, 2000], [1000, 2000]], type: "route" },
        "Second Mid": { bounds: [[400, 0], [1000, 0], [1000, 600], [400, 600]], type: "mid" },
    },
    "de_anubis": {
        "T Spawn": { bounds: [[-2400, -200], [-1600, -200], [-1600, 600], [-2400, 600]], type: "spawn" },
        "CT Spawn": { bounds: [[1000, 1200], [1800, 1200], [1800, 2000], [1000, 2000]], type: "spawn" },
        "A Site": { bounds: [[-400, 1400], [400, 1400], [400, 2200], [-400, 2200]], type: "bombsite" },
        "A Main": { bounds: [[-1200, 800], [-400, 800], [-400, 1400], [-1200, 1400]], type: "route" },
        "A Long": { bounds: [[-1800, 600], [-1200, 600], [-1200, 1200], [-1800, 1200]], type: "route" },
        "Mid": { bounds: [[-600, 0], [200, 0], [200, 800], [-600, 800]], type: "mid" },
        "Connector": { bounds: [[200, 400], [800, 400], [800, 1000], [200, 1000]], type: "route" },
        "B Site": { bounds: [[800, -400], [1600, -400], [1600, 400], [800, 400]], type: "bombsite" },
        "B Main": { bounds: [[-200, -800], [600, -800], [600, -200], [-200, -200]], type: "route" },
        "Canal": { bounds: [[600, -1200], [1200, -1200], [1200, -600], [600, -600]], type: "route" },
        "Palace": { bounds: [[400, 1000], [1000, 1000], [1000, 1600], [400, 1600]], type: "route" },
        "Water": { bounds: [[-1200, 0], [-600, 0], [-600, 600], [-1200, 600]], type: "route" },
    },
};

// ============================================================================
// MAP COORDINATE TRANSFORMATION PARAMETERS
// ============================================================================

const MAP_PARAMS = {
    'de_dust2': { pos_x: -2476, pos_y: 3239, scale: 4.4 },
    'de_mirage': { pos_x: -3230, pos_y: 1713, scale: 5.0 },
    'de_inferno': { pos_x: -2087, pos_y: 3870, scale: 4.9 },
    'de_nuke': { pos_x: -3453, pos_y: 2887, scale: 7.0 },
    'de_overpass': { pos_x: -4831, pos_y: 1781, scale: 5.2 },
    'de_vertigo': { pos_x: -3168, pos_y: 1762, scale: 4.0 },
    'de_ancient': { pos_x: -2953, pos_y: 2164, scale: 5.0 },
    'de_anubis': { pos_x: -2796, pos_y: 3328, scale: 5.22 },
};

const RADAR_URLS = {
    'de_dust2': 'https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_dust2_radar.png',
    'de_mirage': 'https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_mirage_radar.png',
    'de_inferno': 'https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_inferno_radar.png',
    'de_nuke': 'https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_nuke_radar.png',
    'de_overpass': 'https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_overpass_radar.png',
    'de_vertigo': 'https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_vertigo_radar.png',
    'de_ancient': 'https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_ancient_radar.png',
    'de_anubis': 'https://raw.githubusercontent.com/pnxenopoulos/csgo/master/csgo/data/maps/de_anubis_radar.png',
};

// ============================================================================
// MAPZONES CLASS
// ============================================================================

class MapZones {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`MapZones: Container #${containerId} not found`);
            return;
        }

        // Options
        this.options = {
            canvasWidth: options.canvasWidth || 800,
            canvasHeight: options.canvasHeight || 800,
            showLegend: options.showLegend !== false,
            showSidebar: options.showSidebar !== false,
            ...options
        };

        // State
        this.filters = {
            side: 'all',
            phase: 'all',
            economy: 'all',
            player: 'all',
            scope: 'all'
        };
        this.selectedZone = null;
        this.radarImage = null;
        this.data = null;
        this.mapName = '';
        this.zoneDefs = {};

        // Dry Peek Mode state
        this.dryPeekMode = false;
        this.dryPeekData = null;
        this.selectedEntry = null;
        this.hoveredEntry = null;
        this.showSupportRadius = true;
        this.dryPeekFilter = 'all'; // 'all', 'dry', 'supported'

        // Canvas refs
        this.canvas = null;
        this.ctx = null;
    }

    /**
     * Set dry peek visualization data
     * @param {Object} dryPeekData - Data from backend with events, summary, constants
     */
    setDryPeekData(dryPeekData) {
        this.dryPeekData = dryPeekData;
    }

    /**
     * Initialize the map zones visualization
     * @param {Object} heatmapData - Data from backend containing kill_positions, death_positions, zone_definitions, zone_stats
     * @param {string} playerSteamId - Optional player steam ID for "Me" filter
     */
    init(heatmapData, playerSteamId = null) {
        if (!heatmapData) {
            this._renderEmptyState();
            return;
        }

        this.data = heatmapData;
        this.mapName = (heatmapData.map_name || '').toLowerCase();
        this.playerSteamId = playerSteamId;

        // Use zone definitions from backend, fallback to defaults
        this.zoneDefs = heatmapData.zone_definitions || DEFAULT_MAP_ZONES[this.mapName] || {};

        this._render();
        this._loadRadarImage();
        this._setupEventListeners();
    }

    /**
     * Render the complete UI
     */
    _render() {
        const kills = this.data.kill_positions || [];
        const deaths = this.data.death_positions || [];
        const players = this._getUniquePlayers(kills, deaths);

        this.container.innerHTML = `
            <div class="map-zones-wrapper">
                <!-- Main Map Area -->
                <div class="map-zones-main">
                    <div class="map-zones-header">
                        <h3>Map Zones - ${this._escapeHtml(this._formatMapName(this.mapName))}</h3>
                        <div class="map-zones-stats">
                            <span class="kills-count">${kills.length} Kills</span>
                            <span class="deaths-count">${deaths.length} Deaths</span>
                        </div>
                    </div>

                    <!-- Filter Controls -->
                    ${this._renderFilters(players)}

                    <!-- Dry Peek Mode Controls -->
                    ${this.dryPeekData ? this._renderDryPeekControls() : ''}

                    <!-- Canvas Container -->
                    <div class="map-zones-canvas-container">
                        <canvas id="map-zones-canvas" width="${this.options.canvasWidth}" height="${this.options.canvasHeight}"></canvas>
                        <div id="map-zones-tooltip" class="map-zones-tooltip"></div>
                    </div>

                    <!-- Legend -->
                    ${this.options.showLegend ? this._renderLegend() : ''}
                </div>

                <!-- Zone Stats Sidebar -->
                ${this.options.showSidebar ? this._renderSidebar() : ''}
            </div>
        `;

        // Cache canvas refs
        this.canvas = document.getElementById('map-zones-canvas');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    }

    /**
     * Render filter controls
     */
    _renderFilters(players) {
        return `
            <div class="map-zones-filters">
                <!-- Side Filter -->
                <div class="filter-group">
                    <label>Side</label>
                    <div class="btn-group" data-filter="side">
                        <button class="filter-btn active" data-value="all">All</button>
                        <button class="filter-btn" data-value="CT">CT</button>
                        <button class="filter-btn" data-value="T">T</button>
                    </div>
                </div>

                <!-- Phase Filter -->
                <div class="filter-group">
                    <label>Phase</label>
                    <div class="btn-group" data-filter="phase">
                        <button class="filter-btn active" data-value="all">All</button>
                        <button class="filter-btn" data-value="pre_plant">Pre-plant</button>
                        <button class="filter-btn" data-value="post_plant">Post-plant</button>
                    </div>
                </div>

                <!-- Economy Filter -->
                <div class="filter-group">
                    <label>Economy</label>
                    <div class="btn-group" data-filter="economy">
                        <button class="filter-btn active" data-value="all">All</button>
                        <button class="filter-btn" data-value="pistol">Pistol</button>
                        <button class="filter-btn" data-value="eco">Eco</button>
                        <button class="filter-btn" data-value="force">Force</button>
                        <button class="filter-btn" data-value="full_buy">Full Buy</button>
                    </div>
                </div>

                <!-- Player Filter -->
                <div class="filter-group">
                    <label>Player</label>
                    <select id="map-zones-player-filter">
                        <option value="all">All Players</option>
                        ${players.map(p => `<option value="${this._escapeHtml(p)}">${this._escapeHtml(p)}</option>`).join('')}
                    </select>
                </div>

                <!-- Display Toggles -->
                <div class="filter-group toggles">
                    <label>Show</label>
                    <div class="toggle-group">
                        <label class="toggle-label">
                            <input type="checkbox" id="show-kills" checked> Kills
                        </label>
                        <label class="toggle-label">
                            <input type="checkbox" id="show-deaths" checked> Deaths
                        </label>
                        <label class="toggle-label">
                            <input type="checkbox" id="show-zones" checked> Zones
                        </label>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render legend
     */
    _renderLegend() {
        return `
            <div class="map-zones-legend">
                <div class="legend-section">
                    <span class="legend-title">Zone K/D:</span>
                    <div class="legend-items">
                        <div class="legend-item">
                            <span class="legend-color" style="background: rgba(239, 68, 68, 0.4);"></span>
                            <span>Bad (&lt;0.8)</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-color" style="background: rgba(234, 179, 8, 0.4);"></span>
                            <span>Neutral (0.8-1.2)</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-color" style="background: rgba(16, 185, 129, 0.4);"></span>
                            <span>Good (&gt;1.2)</span>
                        </div>
                    </div>
                </div>
                <div class="legend-section">
                    <span class="legend-title">Markers:</span>
                    <div class="legend-items">
                        <div class="legend-item">
                            <span class="legend-dot green"></span>
                            <span>Kill position</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-dot red"></span>
                            <span>Death position</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-dot green-ring"></span>
                            <span>Headshot kill</span>
                        </div>
                    </div>
                </div>
                <div class="legend-tips">
                    <p>Click a zone to highlight its events</p>
                    <p>% = percentage of total events in zone</p>
                </div>
            </div>
        `;
    }

    /**
     * Render sidebar with zone stats
     */
    _renderSidebar() {
        return `
            <div class="map-zones-sidebar">
                <div class="sidebar-header">
                    <h4>Zone Performance</h4>
                </div>
                <div id="zone-stats-list" class="zone-stats-list">
                    <p class="empty-state">Loading zone statistics...</p>
                </div>
            </div>
        `;
    }

    /**
     * Render empty state
     */
    _renderEmptyState() {
        this.container.innerHTML = `
            <div class="map-zones-empty">
                <div class="empty-icon">&#128506;</div>
                <p class="empty-title">Heatmap data not available</p>
                <p class="empty-subtitle">Position tracking requires kill events with position data</p>
            </div>
        `;
    }

    /**
     * Load radar image
     */
    _loadRadarImage() {
        const url = RADAR_URLS[this.mapName];
        if (!url) {
            this._draw();
            return;
        }

        this.radarImage = new Image();
        this.radarImage.crossOrigin = 'anonymous';
        this.radarImage.onload = () => this._draw();
        this.radarImage.onerror = () => {
            this.radarImage = null;
            this._draw();
        };
        this.radarImage.src = url;
    }

    /**
     * Setup event listeners
     */
    _setupEventListeners() {
        // Filter button clicks
        this.container.querySelectorAll('[data-filter]').forEach(group => {
            const filterType = group.dataset.filter;
            group.querySelectorAll('.filter-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    group.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    this.filters[filterType] = btn.dataset.value;
                    this._draw();
                });
            });
        });

        // Player select
        const playerSelect = document.getElementById('map-zones-player-filter');
        if (playerSelect) {
            playerSelect.addEventListener('change', (e) => {
                this.filters.player = e.target.value;
                this._draw();
            });
        }

        // Toggle checkboxes
        ['show-kills', 'show-deaths', 'show-zones'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener('change', () => this._draw());
            }
        });

        // Canvas click for zone selection
        if (this.canvas) {
            this.canvas.addEventListener('click', (e) => this._handleCanvasClick(e));
            this.canvas.addEventListener('mousemove', (e) => this._handleCanvasHover(e));
        }

        // Dry peek mode listeners
        this._setupDryPeekListeners();
    }

    /**
     * Handle canvas click for zone selection
     */
    _handleCanvasClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const canvasX = (e.clientX - rect.left) * scaleX;
        const canvasY = (e.clientY - rect.top) * scaleY;

        const params = MAP_PARAMS[this.mapName] || { pos_x: -3000, pos_y: 3000, scale: 5.0 };
        const gameX = canvasX / (this.canvas.width / 1024) * params.scale + params.pos_x;
        const gameY = params.pos_y - canvasY / (this.canvas.height / 1024) * params.scale;

        // Find clicked zone
        for (const [zoneName, zoneDef] of Object.entries(this.zoneDefs)) {
            if (zoneDef.bounds && this._pointInPolygon(gameX, gameY, zoneDef.bounds)) {
                this.selectedZone = this.selectedZone === zoneName ? null : zoneName;
                this._draw();
                return;
            }
        }

        // Clicked outside any zone - deselect
        if (this.selectedZone) {
            this.selectedZone = null;
            this._draw();
        }
    }

    /**
     * Handle canvas hover for tooltips
     */
    _handleCanvasHover(e) {
        const tooltip = document.getElementById('map-zones-tooltip');
        if (!tooltip) return;

        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        const canvasX = (e.clientX - rect.left) * scaleX;
        const canvasY = (e.clientY - rect.top) * scaleY;

        const params = MAP_PARAMS[this.mapName] || { pos_x: -3000, pos_y: 3000, scale: 5.0 };
        const gameX = canvasX / (this.canvas.width / 1024) * params.scale + params.pos_x;
        const gameY = params.pos_y - canvasY / (this.canvas.height / 1024) * params.scale;

        // Find hovered zone
        for (const [zoneName, zoneDef] of Object.entries(this.zoneDefs)) {
            if (zoneDef.bounds && this._pointInPolygon(gameX, gameY, zoneDef.bounds)) {
                const stats = this._getFilteredZoneStats();
                const zs = stats[zoneName];
                if (zs) {
                    tooltip.innerHTML = `
                        <strong>${zoneName}</strong><br>
                        K/D: ${zs.kd_ratio.toFixed(2)}<br>
                        ${zs.kills}K / ${zs.deaths}D<br>
                        ${zs.kill_pct}% of kills
                    `;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.clientX - rect.left + 10) + 'px';
                    tooltip.style.top = (e.clientY - rect.top + 10) + 'px';
                    return;
                }
            }
        }

        tooltip.style.display = 'none';
    }

    /**
     * Main draw function
     */
    _draw() {
        if (!this.ctx) return;

        const ctx = this.ctx;
        const canvas = this.canvas;

        // Clear canvas
        ctx.fillStyle = '#0d0d12';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw radar background or grid
        if (this.radarImage && this.radarImage.complete) {
            ctx.globalAlpha = 0.6;
            ctx.drawImage(this.radarImage, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        } else {
            this._drawGrid();
        }

        // Get filtered data
        const kills = this._filterPositions(this.data.kill_positions || []);
        const deaths = this._filterPositions(this.data.death_positions || []);
        const zoneStats = this._computeZoneStats(kills, deaths);

        // Draw zone overlays
        const showZones = document.getElementById('show-zones')?.checked ?? true;
        if (showZones && Object.keys(this.zoneDefs).length > 0) {
            this._drawZones(zoneStats);
        }

        // Draw deaths first (below kills) - only when not in dry peek mode
        const showDeaths = document.getElementById('show-deaths')?.checked ?? true;
        if (showDeaths && !this.dryPeekMode) {
            for (const pos of deaths) {
                if (pos.x == null || pos.y == null) continue;
                const canvasPos = this._gameToCanvas(pos.x, pos.y);
                this._drawMarker(canvasPos.x, canvasPos.y, 'death', pos);
            }
        }

        // Draw kills on top (only when not in dry peek mode)
        const showKills = document.getElementById('show-kills')?.checked ?? true;
        if (showKills && !this.dryPeekMode) {
            for (const pos of kills) {
                if (pos.x == null || pos.y == null) continue;
                const canvasPos = this._gameToCanvas(pos.x, pos.y);
                this._drawMarker(canvasPos.x, canvasPos.y, 'kill', pos);
            }
        }

        // Draw dry peek overlay (when in dry peek mode)
        if (this.dryPeekMode) {
            this._drawDryPeekOverlay();
        }

        // Update sidebar
        this._updateSidebar(zoneStats);
    }

    /**
     * Draw grid background
     */
    _drawGrid() {
        const ctx = this.ctx;
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 10; i++) {
            ctx.beginPath();
            ctx.moveTo(i * 80, 0);
            ctx.lineTo(i * 80, 800);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, i * 80);
            ctx.lineTo(800, i * 80);
            ctx.stroke();
        }
    }

    /**
     * Draw zone overlays with K/D-based coloring
     */
    _drawZones(stats) {
        const ctx = this.ctx;

        for (const [zoneName, zoneDef] of Object.entries(this.zoneDefs)) {
            const zs = stats[zoneName];
            if (!zoneDef.bounds || zoneDef.bounds.length < 3) continue;

            // Get color based on K/D ratio
            const kd = zs ? zs.kd_ratio : 1;
            let fillColor = this._getZoneColor(kd);

            // Highlight selected zone
            if (this.selectedZone === zoneName) {
                fillColor = fillColor.replace(/[\d.]+\)$/, '0.5)');
            }

            // Draw polygon
            ctx.beginPath();
            const firstPoint = this._gameToCanvas(zoneDef.bounds[0][0], zoneDef.bounds[0][1]);
            ctx.moveTo(firstPoint.x, firstPoint.y);
            for (let i = 1; i < zoneDef.bounds.length; i++) {
                const point = this._gameToCanvas(zoneDef.bounds[i][0], zoneDef.bounds[i][1]);
                ctx.lineTo(point.x, point.y);
            }
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();
            ctx.strokeStyle = this.selectedZone === zoneName ? '#00e5ff' : 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = this.selectedZone === zoneName ? 2 : 1;
            ctx.stroke();

            // Draw zone label
            if (zs && (zs.kills > 0 || zs.deaths > 0)) {
                const center = this._getPolygonCenter(zoneDef.bounds);
                const canvasCenter = this._gameToCanvas(center.x, center.y);
                ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                ctx.font = '11px Inter, system-ui, sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(`${zs.kill_pct}%`, canvasCenter.x, canvasCenter.y - 5);
                ctx.font = '9px Inter, system-ui, sans-serif';
                ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
                const displayName = zoneName.length > 12 ? zoneName.substring(0, 10) + '..' : zoneName;
                ctx.fillText(displayName, canvasCenter.x, canvasCenter.y + 8);
            }
        }
    }

    /**
     * Draw a kill/death marker
     */
    _drawMarker(x, y, type, data) {
        const ctx = this.ctx;
        const color = type === 'kill' ? '16, 185, 129' : '239, 68, 68';
        const highlight = this.selectedZone && data.zone === this.selectedZone;
        const radius = highlight ? 14 : 10;

        // Glow effect
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
        gradient.addColorStop(0, `rgba(${color}, ${highlight ? 1 : 0.8})`);
        gradient.addColorStop(1, `rgba(${color}, 0)`);
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();

        // Core dot
        ctx.fillStyle = `rgba(${color}, 1)`;
        ctx.beginPath();
        ctx.arc(x, y, highlight ? 5 : 3, 0, Math.PI * 2);
        ctx.fill();

        // Headshot indicator
        if (type === 'kill' && data.headshot) {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    /**
     * Get zone color based on K/D ratio
     */
    _getZoneColor(kd) {
        if (kd > 2.0) return 'rgba(16, 185, 129, 0.5)';   // Bright green
        if (kd > 1.5) return 'rgba(16, 185, 129, 0.35)';  // Green
        if (kd > 1.2) return 'rgba(16, 185, 129, 0.25)';  // Light green
        if (kd >= 0.8) return 'rgba(234, 179, 8, 0.25)';  // Neutral yellow
        if (kd > 0.5) return 'rgba(239, 68, 68, 0.25)';   // Light red
        return 'rgba(239, 68, 68, 0.4)';                   // Red
    }

    /**
     * Update sidebar with zone statistics (or dry peek stats in dry peek mode)
     */
    _updateSidebar(stats) {
        const panel = document.getElementById('zone-stats-list');
        if (!panel) return;

        // In dry peek mode, show dry peek sidebar instead
        if (this.dryPeekMode && this.dryPeekData) {
            panel.innerHTML = this._renderDryPeekSidebar();
            return;
        }

        const sortedZones = Object.entries(stats || {})
            .filter(([z, s]) => s.kills > 0 || s.deaths > 0)
            .sort((a, b) => b[1].kill_pct - a[1].kill_pct);

        if (sortedZones.length === 0) {
            panel.innerHTML = '<p class="empty-state">No data for current filters</p>';
            return;
        }

        panel.innerHTML = sortedZones.map(([zone, s]) => {
            const kdColor = s.kd_ratio >= 1.2 ? '#10b981' : s.kd_ratio >= 0.8 ? '#f59e0b' : '#ef4444';
            const isSelected = this.selectedZone === zone;
            return `
                <div class="zone-stat-item ${isSelected ? 'selected' : ''}" data-zone="${this._escapeHtml(zone)}">
                    <div class="zone-stat-header">
                        <span class="zone-name">${this._escapeHtml(zone)}</span>
                        <span class="zone-kd" style="color: ${kdColor};">${s.kd_ratio.toFixed(2)} K/D</span>
                    </div>
                    <div class="zone-stat-details">
                        <span>${s.kill_pct}% of kills</span>
                        <span class="kills">${s.kills}K</span>
                        <span class="deaths">${s.deaths}D</span>
                    </div>
                </div>
            `;
        }).join('');

        // Add click handlers
        panel.querySelectorAll('.zone-stat-item').forEach(item => {
            item.addEventListener('click', () => {
                const zone = item.dataset.zone;
                this.selectedZone = this.selectedZone === zone ? null : zone;
                this._draw();
            });
        });
    }

    /**
     * Filter positions based on current filters
     */
    _filterPositions(positions) {
        return positions.filter(p => {
            // Side filter: check if side matches (CT or T)
            // If filtering for a specific side, exclude events with no side or wrong side
            if (this.filters.side !== 'all') {
                const pSide = (p.side || '').toUpperCase();
                if (!pSide || !pSide.includes(this.filters.side)) return false;
            }

            // Phase filter: pre_plant or post_plant
            if (this.filters.phase !== 'all' && p.phase !== this.filters.phase) return false;

            // Economy filter: handle semi_eco grouping with eco
            if (this.filters.economy !== 'all') {
                const roundType = p.round_type || '';
                // Group semi_eco with eco for filtering purposes
                if (this.filters.economy === 'eco') {
                    if (roundType !== 'eco' && roundType !== 'semi_eco') return false;
                } else if (roundType !== this.filters.economy) {
                    return false;
                }
            }

            // Player filter
            if (this.filters.player !== 'all' && p.player_name !== this.filters.player) return false;

            return true;
        });
    }

    /**
     * Compute zone statistics from filtered data
     */
    _computeZoneStats(kills, deaths) {
        const stats = {};
        const totalKills = kills.length;

        for (const k of kills) {
            const zone = k.zone || 'Unknown';
            if (!stats[zone]) stats[zone] = { kills: 0, deaths: 0, ct_kills: 0, t_kills: 0 };
            stats[zone].kills++;
            if (k.side && k.side.toUpperCase().includes('CT')) stats[zone].ct_kills++;
            else stats[zone].t_kills++;
        }

        for (const d of deaths) {
            const zone = d.zone || 'Unknown';
            if (!stats[zone]) stats[zone] = { kills: 0, deaths: 0, ct_kills: 0, t_kills: 0 };
            stats[zone].deaths++;
        }

        for (const [zone, s] of Object.entries(stats)) {
            s.kd_ratio = s.deaths > 0 ? (s.kills / s.deaths) : s.kills;
            s.kill_pct = totalKills > 0 ? Math.round(s.kills / totalKills * 100) : 0;
        }

        return stats;
    }

    /**
     * Get filtered zone stats (for hover tooltip)
     */
    _getFilteredZoneStats() {
        const kills = this._filterPositions(this.data.kill_positions || []);
        const deaths = this._filterPositions(this.data.death_positions || []);
        return this._computeZoneStats(kills, deaths);
    }

    /**
     * Convert game coordinates to canvas coordinates
     */
    _gameToCanvas(x, y) {
        const params = MAP_PARAMS[this.mapName] || { pos_x: -3000, pos_y: 3000, scale: 5.0 };
        const pixelX = (x - params.pos_x) / params.scale;
        const pixelY = (params.pos_y - y) / params.scale;
        const scaleX = this.canvas.width / 1024;
        const scaleY = this.canvas.height / 1024;
        return {
            x: Math.max(0, Math.min(this.canvas.width, pixelX * scaleX)),
            y: Math.max(0, Math.min(this.canvas.height, pixelY * scaleY))
        };
    }

    /**
     * Get polygon center point
     */
    _getPolygonCenter(bounds) {
        let sumX = 0, sumY = 0;
        for (const [x, y] of bounds) { sumX += x; sumY += y; }
        return { x: sumX / bounds.length, y: sumY / bounds.length };
    }

    /**
     * Point in polygon check
     */
    _pointInPolygon(x, y, polygon) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i][0], yi = polygon[i][1];
            const xj = polygon[j][0], yj = polygon[j][1];
            if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
                inside = !inside;
            }
        }
        return inside;
    }

    /**
     * Get unique players from kill/death positions
     */
    _getUniquePlayers(kills, deaths) {
        return [...new Set([
            ...kills.map(k => k.player_name),
            ...deaths.map(d => d.player_name)
        ])].filter(Boolean).sort();
    }

    /**
     * Format map name for display
     */
    _formatMapName(name) {
        if (!name) return 'Unknown Map';
        return name.replace(/^de_/, '').replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    }

    /**
     * Escape HTML to prevent XSS
     */
    _escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    /**
     * Update filters programmatically
     */
    setFilter(filterType, value) {
        if (this.filters.hasOwnProperty(filterType)) {
            this.filters[filterType] = value;
            this._draw();
        }
    }

    /**
     * Select a zone programmatically
     */
    selectZone(zoneName) {
        this.selectedZone = zoneName;
        this._draw();
    }

    /**
     * Clear zone selection
     */
    clearSelection() {
        this.selectedZone = null;
        this._draw();
    }

    // ============================================================================
    // DRY PEEK VISUALIZATION METHODS
    // ============================================================================

    /**
     * Convert game units to canvas pixels (for radius drawing)
     */
    _gameUnitsToPixels(units) {
        const params = MAP_PARAMS[this.mapName] || { pos_x: -3000, pos_y: 3000, scale: 5.0 };
        // units / scale gives radar pixels (1024x1024), then scale to canvas
        return (units / params.scale) * (this.canvas.width / 1024);
    }

    /**
     * Filter dry peek events based on current filter setting
     */
    _filterDryPeekEvents(events) {
        if (!events) return [];

        let filtered = events;

        // Apply player filter if set
        if (this.filters.player !== 'all') {
            const steamId = this.filters.player;
            filtered = filtered.filter(e => String(e.player_steamid) === String(steamId));
        }

        // Apply side filter
        if (this.filters.side !== 'all') {
            const side = this.filters.side.toUpperCase();
            filtered = filtered.filter(e => e.side === side);
        }

        // Apply dry peek type filter
        if (this.dryPeekFilter === 'dry') {
            filtered = filtered.filter(e => !e.is_supported);
        } else if (this.dryPeekFilter === 'supported') {
            filtered = filtered.filter(e => e.is_supported);
        }

        return filtered;
    }

    /**
     * Draw dry peek overlay (main entry point)
     */
    _drawDryPeekOverlay() {
        if (!this.dryPeekMode || !this.dryPeekData?.events) return;

        const events = this._filterDryPeekEvents(this.dryPeekData.events);

        // Draw support radius for selected/hovered entry first (below markers)
        for (const event of events) {
            if (this.showSupportRadius && (this.selectedEntry === event.id || this.hoveredEntry === event.id)) {
                this._drawSupportRadius(event);
            }
        }

        // Draw all entry markers
        for (const event of events) {
            if (event.x == null || event.y == null) continue;
            this._drawDryPeekMarker(event);
        }
    }

    /**
     * Draw support radius circle around an engagement
     */
    _drawSupportRadius(event) {
        const ctx = this.ctx;
        const pos = this._gameToCanvas(event.x, event.y);
        const radiusPixels = this._gameUnitsToPixels(this.dryPeekData?.constants?.support_radius_units || 2000);

        // Semi-transparent fill
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radiusPixels, 0, Math.PI * 2);
        ctx.fillStyle = event.is_supported
            ? 'rgba(16, 185, 129, 0.12)'  // Green for supported
            : 'rgba(239, 68, 68, 0.12)';   // Red for dry peek
        ctx.fill();

        // Dashed border
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = event.is_supported ? '#10b981' : '#ef4444';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw supporting utilities (flashes/smokes)
        if (event.support_utilities?.length > 0) {
            this._drawSupportUtilities(event.support_utilities, pos);
        }
    }

    /**
     * Draw flash/smoke positions that supported an engagement
     */
    _drawSupportUtilities(utilities, engagementPos) {
        const ctx = this.ctx;

        for (const util of utilities) {
            if (util.x == null || util.y == null) continue;
            const pos = this._gameToCanvas(util.x, util.y);

            // Line connecting utility to engagement
            ctx.beginPath();
            ctx.strokeStyle = util.type === 'flashbang' ? 'rgba(254, 240, 138, 0.5)' : 'rgba(148, 163, 184, 0.5)';
            ctx.setLineDash([3, 3]);
            ctx.lineWidth = 1.5;
            ctx.moveTo(pos.x, pos.y);
            ctx.lineTo(engagementPos.x, engagementPos.y);
            ctx.stroke();
            ctx.setLineDash([]);

            // Utility marker
            const color = util.type === 'flashbang' ? '#fef08a' : '#94a3b8';
            const size = 7;

            // Glow
            const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, size * 2);
            gradient.addColorStop(0, color);
            gradient.addColorStop(1, 'transparent');
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, size * 2, 0, Math.PI * 2);
            ctx.fill();

            // Core
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, size, 0, Math.PI * 2);
            ctx.fill();

            // Icon text
            ctx.fillStyle = '#000';
            ctx.font = 'bold 8px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(util.type === 'flashbang' ? 'F' : 'S', pos.x, pos.y);
        }
    }

    /**
     * Draw a dry peek entry marker
     */
    _drawDryPeekMarker(event) {
        const ctx = this.ctx;
        const pos = this._gameToCanvas(event.x, event.y);

        const isKill = event.event_type === 'entry_kill';
        const isDryPeek = !event.is_supported;
        const isSelected = this.selectedEntry === event.id;
        const isHovered = this.hoveredEntry === event.id;

        // Color coding:
        // - Green (#10b981): Supported entry kill (good play)
        // - Orange (#f59e0b): Dry peek kill (risky but worked)
        // - Red (#ef4444): Dry peek death (punished)
        // - Blue (#3b82f6): Supported entry death (unlucky)
        let color;
        if (isKill && event.is_supported) color = '#10b981';      // Green
        else if (isKill && isDryPeek) color = '#f59e0b';           // Orange
        else if (!isKill && isDryPeek) color = '#ef4444';          // Red
        else color = '#3b82f6';                                     // Blue

        const radius = (isSelected || isHovered) ? 12 : 8;

        // Glow effect
        const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, radius * 2);
        gradient.addColorStop(0, color);
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius * 2, 0, Math.PI * 2);
        ctx.fill();

        // Core marker
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fill();

        // White border for kills
        if (isKill) {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Selection ring
        if (isSelected || isHovered) {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2);
            ctx.stroke();
        }

        // Store position for hit testing
        event._canvasX = pos.x;
        event._canvasY = pos.y;
        event._radius = radius;
    }

    /**
     * Render dry peek mode controls
     */
    _renderDryPeekControls() {
        const summary = this.dryPeekData?.summary || {};
        const dryPeekRate = summary.dry_peek_rate || 0;
        const rateClass = dryPeekRate > 50 ? 'danger' : dryPeekRate > 30 ? 'warning' : 'success';

        return `
            <div class="dry-peek-controls">
                <label class="toggle-label">
                    <input type="checkbox" id="dry-peek-mode" ${this.dryPeekMode ? 'checked' : ''}>
                    <span>Dry Peek Mode</span>
                    ${this.dryPeekData ? `<span class="dry-peek-rate ${rateClass}">${dryPeekRate.toFixed(1)}%</span>` : ''}
                </label>

                <div class="dry-peek-filters" style="display: ${this.dryPeekMode ? 'block' : 'none'};">
                    <div class="filter-group">
                        <label>Show</label>
                        <div class="btn-group" id="dry-peek-type-filter">
                            <button class="filter-btn ${this.dryPeekFilter === 'all' ? 'active' : ''}" data-value="all">All Entries</button>
                            <button class="filter-btn ${this.dryPeekFilter === 'dry' ? 'active' : ''}" data-value="dry">Dry Peeks</button>
                            <button class="filter-btn ${this.dryPeekFilter === 'supported' ? 'active' : ''}" data-value="supported">Supported</button>
                        </div>
                    </div>
                    <label class="toggle-label">
                        <input type="checkbox" id="show-support-radius" ${this.showSupportRadius ? 'checked' : ''}>
                        <span>Show Support Radius on Hover</span>
                    </label>
                </div>
            </div>
        `;
    }

    /**
     * Render dry peek statistics sidebar section
     */
    _renderDryPeekSidebar() {
        if (!this.dryPeekData?.summary || !this.dryPeekMode) return '';

        const summary = this.dryPeekData.summary;
        const byPlayer = Object.entries(summary.by_player || {})
            .sort((a, b) => b[1].dry_peek_rate - a[1].dry_peek_rate);

        const rateClass = summary.dry_peek_rate > 50 ? 'danger' : summary.dry_peek_rate > 30 ? 'warning' : 'success';

        return `
            <div class="dry-peek-sidebar">
                <h4>Dry Peek Analysis</h4>
                <div class="dry-peek-summary">
                    <div class="stat-row">
                        <span>Total Entries</span>
                        <span>${summary.total_entries}</span>
                    </div>
                    <div class="stat-row">
                        <span>Supported</span>
                        <span class="success">${summary.supported_entries}</span>
                    </div>
                    <div class="stat-row">
                        <span>Dry Peeks</span>
                        <span class="danger">${summary.dry_peek_entries}</span>
                    </div>
                    <div class="stat-row highlight">
                        <span>Dry Peek Rate</span>
                        <span class="${rateClass}">${summary.dry_peek_rate.toFixed(1)}%</span>
                    </div>
                </div>

                <h5>By Player</h5>
                <div class="player-dry-peek-list">
                    ${byPlayer.map(([steamId, data]) => `
                        <div class="player-dry-peek" data-steamid="${steamId}">
                            <span class="player-name">${this._escapeHtml(data.name)}</span>
                            <span class="dry-peek-rate ${data.dry_peek_rate > 50 ? 'danger' : ''}">
                                ${data.dry_peek_rate.toFixed(1)}%
                            </span>
                            <span class="entry-counts">
                                ${data.supported}/${data.supported + data.unsupported}
                            </span>
                        </div>
                    `).join('')}
                </div>

                <div class="dry-peek-legend">
                    <div class="legend-item">
                        <span class="legend-marker supported-kill"></span>
                        <span>Supported Kill</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-marker dry-peek-kill"></span>
                        <span>Dry Peek Kill</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-marker dry-peek-death"></span>
                        <span>Dry Peek Death</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-marker supported-death"></span>
                        <span>Supported Death</span>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Setup dry peek event listeners
     */
    _setupDryPeekListeners() {
        // Dry peek mode toggle
        const dryPeekToggle = document.getElementById('dry-peek-mode');
        if (dryPeekToggle) {
            dryPeekToggle.addEventListener('change', (e) => {
                this.dryPeekMode = e.target.checked;
                const filters = document.querySelector('.dry-peek-filters');
                if (filters) filters.style.display = this.dryPeekMode ? 'block' : 'none';
                this._draw();
                this._updateSidebar();
            });
        }

        // Support radius toggle
        const radiusToggle = document.getElementById('show-support-radius');
        if (radiusToggle) {
            radiusToggle.addEventListener('change', (e) => {
                this.showSupportRadius = e.target.checked;
                this._draw();
            });
        }

        // Dry peek type filter buttons
        const typeFilter = document.getElementById('dry-peek-type-filter');
        if (typeFilter) {
            typeFilter.addEventListener('click', (e) => {
                const btn = e.target.closest('.filter-btn');
                if (!btn) return;
                typeFilter.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.dryPeekFilter = btn.dataset.value;
                this._draw();
            });
        }

        // Canvas hover for dry peek markers
        if (this.canvas) {
            this.canvas.addEventListener('mousemove', (e) => {
                if (!this.dryPeekMode || !this.dryPeekData?.events) return;

                const rect = this.canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                // Find hovered entry
                let foundEntry = null;
                for (const event of this.dryPeekData.events) {
                    if (event._canvasX == null) continue;
                    const dx = x - event._canvasX;
                    const dy = y - event._canvasY;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist <= (event._radius || 8) + 5) {
                        foundEntry = event.id;
                        break;
                    }
                }

                if (this.hoveredEntry !== foundEntry) {
                    this.hoveredEntry = foundEntry;
                    this._draw();

                    // Show tooltip
                    if (foundEntry) {
                        const event = this.dryPeekData.events.find(e => e.id === foundEntry);
                        if (event) {
                            this._showDryPeekTooltip(e, event);
                        }
                    } else {
                        this._hideTooltip();
                    }
                }
            });

            // Canvas click to select entry
            this.canvas.addEventListener('click', (e) => {
                if (!this.dryPeekMode || !this.dryPeekData?.events) return;

                const rect = this.canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                // Find clicked entry
                for (const event of this.dryPeekData.events) {
                    if (event._canvasX == null) continue;
                    const dx = x - event._canvasX;
                    const dy = y - event._canvasY;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist <= (event._radius || 8) + 5) {
                        this.selectedEntry = this.selectedEntry === event.id ? null : event.id;
                        this._draw();
                        return;
                    }
                }

                // Click on empty space clears selection
                this.selectedEntry = null;
                this._draw();
            });
        }
    }

    /**
     * Show tooltip for a dry peek event
     */
    _showDryPeekTooltip(mouseEvent, event) {
        const tooltip = document.getElementById('map-zones-tooltip');
        if (!tooltip) return;

        const statusIcon = event.is_supported ? '✓' : '✗';
        const statusText = event.is_supported ? 'Supported' : 'Dry Peek';
        const statusClass = event.is_supported ? 'success' : 'danger';
        const eventType = event.event_type === 'entry_kill' ? 'Entry Kill' : 'Entry Death';

        let supportInfo = '';
        if (event.is_supported && event.support_utilities?.length > 0) {
            const utils = event.support_utilities.map(u =>
                `${u.type === 'flashbang' ? 'Flash' : 'Smoke'} by ${u.thrower_name} (${u.time_before_ms}ms before)`
            ).join('<br>');
            supportInfo = `<div class="tooltip-support"><strong>Support:</strong><br>${utils}</div>`;
        }

        tooltip.innerHTML = `
            <div class="dry-peek-tooltip">
                <div class="tooltip-header">
                    <span class="tooltip-player">${this._escapeHtml(event.player_name)}</span>
                    <span class="tooltip-round">Round ${event.round_num}</span>
                </div>
                <div class="tooltip-type">${eventType} (${event.weapon})</div>
                <div class="tooltip-status ${statusClass}">${statusIcon} ${statusText}</div>
                ${supportInfo}
            </div>
        `;

        tooltip.style.display = 'block';
        tooltip.style.left = `${mouseEvent.clientX - this.canvas.getBoundingClientRect().left + 15}px`;
        tooltip.style.top = `${mouseEvent.clientY - this.canvas.getBoundingClientRect().top + 15}px`;
    }

    /**
     * Hide tooltip
     */
    _hideTooltip() {
        const tooltip = document.getElementById('map-zones-tooltip');
        if (tooltip) tooltip.style.display = 'none';
    }
}

// ============================================================================
// STANDALONE RENDER FUNCTION (for backwards compatibility)
// ============================================================================

/**
 * Render map zones into a container
 * @param {string} containerId - ID of container element
 * @param {Object} heatmapData - Heatmap data from backend
 * @param {string} playerSteamId - Optional player steam ID
 * @param {Object} options - Optional configuration
 * @returns {MapZones} The MapZones instance
 */
function renderMapZones(containerId, heatmapData, playerSteamId = null, options = {}) {
    const mapZones = new MapZones(containerId, options);
    mapZones.init(heatmapData, playerSteamId);
    return mapZones;
}


