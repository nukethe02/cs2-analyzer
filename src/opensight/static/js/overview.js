/**
 * Enhanced Overview Tab for OpenSight
 *
 * Provides Leetify-style overview with:
 * - Top 3 player podium with ratings
 * - Sortable columns with stars for best values
 * - Team sections with win/loss badges
 * - Color-coded performance metrics
 *
 * Usage:
 *   const overview = new OverviewTab('container-id');
 *   overview.setData(players, myTeamIds, myTeamScore, enemyTeamScore);
 */

class OverviewTab {
    /**
     * Create a new overview tab
     * @param {string} containerId - ID of the container element
     */
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.sortCol = 'hltv_rating';
        this.sortDir = 'desc';
        this.data = [];
        this.myTeam = new Set();
        this.myTeamScore = 0;
        this.enemyTeamScore = 0;
        this.best = {};
    }

    /**
     * Set the data for the overview
     * @param {Array} players - Array of player objects
     * @param {Array} myTeamIds - Array of steam IDs for the user's team
     * @param {number} myTeamScore - Score of user's team
     * @param {number} enemyTeamScore - Score of enemy team
     */
    setData(players, myTeamIds, myTeamScore = 0, enemyTeamScore = 0) {
        this.data = players;
        this.myTeam = new Set(myTeamIds.map(id => String(id)));
        this.myTeamScore = myTeamScore;
        this.enemyTeamScore = enemyTeamScore;
        this.calcBestValues();
        this.render();
    }

    /**
     * Calculate best values for highlighting
     */
    calcBestValues() {
        const columns = ['hltv_rating', 'kd_ratio', 'adr', 'aim_rating', 'utility_rating', 'personal_performance'];
        this.best = {};
        columns.forEach(col => {
            const values = this.data.map(p => this.getPlayerValue(p, col) || 0);
            this.best[col] = Math.max(...values);
        });
    }

    /**
     * Get a value from a player object
     * @param {Object} player - Player data
     * @param {string} key - Property key
     * @returns {*} Value
     */
    getPlayerValue(player, key) {
        // Check direct property
        if (player[key] !== undefined) return player[key];

        // Check nested properties
        if (player.stats && player.stats[key] !== undefined) return player.stats[key];
        if (player.rating && player.rating[key] !== undefined) return player.rating[key];

        // Calculate derived values
        if (key === 'kd_ratio') {
            const kills = player.kills || player.stats?.kills || 0;
            const deaths = player.deaths || player.stats?.deaths || 1;
            return deaths > 0 ? kills / deaths : kills;
        }
        if (key === 'personal_performance') {
            return this.calculatePersonalPerformance(player);
        }

        return 0;
    }

    /**
     * Calculate K-D difference for a player
     * @param {Object} p - Player data
     * @returns {number} Kill-Death difference
     */
    calculatePersonalPerformance(p) {
        const kills = p.kills || p.stats?.kills || 0;
        const deaths = p.deaths || p.stats?.deaths || 0;

        // Return simple K-D difference (e.g., +5, -2)
        return kills - deaths;
    }

    /**
     * Format rating with +/- sign relative to baseline
     * @param {number} rating - The rating value
     * @returns {string} Formatted rating string
     */
    formatRating(rating) {
        const diff = (rating || 0) - 1.0; // Baseline is 1.0
        const sign = diff >= 0 ? '+' : '';
        return `${sign}${(diff * 100).toFixed(0)}`;
    }

    /**
     * Get CSS class for rating color
     * @param {number} rating - The rating value
     * @returns {string} CSS class name
     */
    getRatingClass(rating) {
        if (rating >= 1.2) return 'rating-excellent';
        if (rating >= 1.0) return 'rating-good';
        if (rating >= 0.8) return 'rating-average';
        return 'rating-poor';
    }

    /**
     * Get CSS class for performance value
     * @param {number} perf - Performance value
     * @returns {string} CSS class
     */
    getPerformanceClass(perf) {
        if (perf >= 2) return 'positive';
        if (perf <= -2) return 'negative';
        return '';
    }

    /**
     * Get CSS class for HLTV rating
     * @param {number} rating - HLTV rating
     * @returns {string} CSS class
     */
    getHLTVClass(rating) {
        if (rating >= 1.3) return 'rating-excellent';
        if (rating >= 1.0) return 'rating-good';
        if (rating >= 0.8) return 'rating-average';
        return 'rating-poor';
    }

    /**
     * Get CSS class for K/D ratio
     * @param {number} kd - K/D ratio
     * @returns {string} CSS class
     */
    getKDClass(kd) {
        if (kd >= 1.5) return 'positive';
        if (kd >= 1.0) return 'neutral';
        return 'negative';
    }

    /**
     * Get CSS class for ADR
     * @param {number} adr - ADR value
     * @returns {string} CSS class
     */
    getADRClass(adr) {
        if (adr >= 90) return 'rating-excellent';
        if (adr >= 70) return 'rating-good';
        if (adr >= 50) return 'rating-average';
        return 'rating-poor';
    }

    /**
     * Get CSS class for aim rating
     * @param {number} rating - Aim rating
     * @returns {string} CSS class
     */
    getAimClass(rating) {
        if (rating >= 70) return 'rating-excellent';
        if (rating >= 50) return 'rating-good';
        if (rating >= 30) return 'rating-average';
        return 'rating-poor';
    }

    /**
     * Get CSS class for utility rating
     * @param {number} rating - Utility rating
     * @returns {string} CSS class
     */
    getUtilityClass(rating) {
        if (rating >= 60) return 'rating-excellent';
        if (rating >= 40) return 'rating-good';
        if (rating >= 20) return 'rating-average';
        return 'rating-poor';
    }

    /**
     * Check if player has the best value for a column
     * @param {string} column - Column name
     * @param {Object} player - Player data
     * @returns {boolean} True if player has best value
     */
    isBest(column, player) {
        const val = this.getPlayerValue(player, column);
        return val === this.best[column] && val > 0;
    }

    /**
     * Escape HTML to prevent XSS
     * @param {string} str - String to escape
     * @returns {string} Escaped string
     */
    escapeHtml(str) {
        if (typeof str !== 'string') return str;
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    /**
     * Render the complete overview
     */
    render() {
        if (!this.container) return;

        // Sort all players by current sort column for podium
        const allSorted = this.sortPlayers([...this.data]);

        // Split data by team
        const myTeamPlayers = this.data.filter(p => this.myTeam.has(String(p.steamid || p.steam_id)));
        const enemyPlayers = this.data.filter(p => !this.myTeam.has(String(p.steamid || p.steam_id)));

        // Determine results
        const myResult = this.myTeamScore > this.enemyTeamScore ? 'WIN' :
                         this.myTeamScore < this.enemyTeamScore ? 'LOSS' : 'TIE';
        const enemyResult = this.enemyTeamScore > this.myTeamScore ? 'WIN' :
                            this.enemyTeamScore < this.myTeamScore ? 'LOSS' : 'TIE';

        // Sort data within teams
        const sortedMy = this.sortPlayers([...myTeamPlayers]);
        const sortedEnemy = this.sortPlayers([...enemyPlayers]);

        this.container.innerHTML = `
            ${this.renderPodium(allSorted)}
            ${this.renderSortableTable(sortedMy, sortedEnemy, myResult, enemyResult)}
        `;

        this.attachSortListeners();
    }

    /**
     * Render the top 3 podium section
     * @param {Array} sortedPlayers - Players sorted by rating
     * @returns {string} HTML string
     */
    renderPodium(sortedPlayers) {
        if (sortedPlayers.length < 3) return '';

        const top3 = sortedPlayers.slice(0, 3);
        const [first, second, third] = top3;

        return `
            <div class="top-players-podium">
                <!-- 2nd place (left) -->
                <div class="podium-player second">
                    <span class="podium-rank">2ND</span>
                    <div class="podium-avatar">${this.escapeHtml((second.name || 'U')[0].toUpperCase())}</div>
                    <span class="podium-name">${this.escapeHtml(second.name || 'Unknown')}</span>
                    <div class="podium-rating ${this.getRatingClass(this.getPlayerValue(second, 'hltv_rating'))}">
                        ${this.formatRating(this.getPlayerValue(second, 'hltv_rating'))}
                    </div>
                </div>

                <!-- 1st place (center, elevated) -->
                <div class="podium-player first">
                    <span class="podium-rank">1ST</span>
                    <div class="podium-avatar">${this.escapeHtml((first.name || 'U')[0].toUpperCase())}</div>
                    <span class="podium-name">${this.escapeHtml(first.name || 'Unknown')}</span>
                    <div class="podium-rating ${this.getRatingClass(this.getPlayerValue(first, 'hltv_rating'))}">
                        ${this.formatRating(this.getPlayerValue(first, 'hltv_rating'))}
                    </div>
                </div>

                <!-- 3rd place (right) -->
                <div class="podium-player third">
                    <span class="podium-rank">3RD</span>
                    <div class="podium-avatar">${this.escapeHtml((third.name || 'U')[0].toUpperCase())}</div>
                    <span class="podium-name">${this.escapeHtml(third.name || 'Unknown')}</span>
                    <div class="podium-rating ${this.getRatingClass(this.getPlayerValue(third, 'hltv_rating'))}">
                        ${this.formatRating(this.getPlayerValue(third, 'hltv_rating'))}
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Sort players by current sort column
     * @param {Array} players - Players to sort
     * @returns {Array} Sorted players
     */
    sortPlayers(players) {
        return players.sort((a, b) => {
            const av = this.getPlayerValue(a, this.sortCol) || 0;
            const bv = this.getPlayerValue(b, this.sortCol) || 0;
            return this.sortDir === 'asc' ? av - bv : bv - av;
        });
    }

    /**
     * Render the sortable table with team sections
     * @param {Array} myTeam - My team players
     * @param {Array} enemyTeam - Enemy team players
     * @param {string} myResult - My team result
     * @param {string} enemyResult - Enemy team result
     * @returns {string} HTML string
     */
    renderSortableTable(myTeam, enemyTeam, myResult, enemyResult) {
        const columns = [
            { key: 'name', label: 'Player', sortable: false },
            { key: 'hltv_rating', label: 'Rating', sortable: true },
            { key: 'personal_performance', label: '+/-', sortable: true },
            { key: 'kd_ratio', label: 'K/D', sortable: true },
            { key: 'adr', label: 'ADR', sortable: true },
            { key: 'aim_rating', label: 'Aim', sortable: true },
            { key: 'utility_rating', label: 'Utility', sortable: true }
        ];

        return `
            <div class="overview-table-wrapper">
                <table class="overview-table sortable">
                    <thead>
                        <tr>
                            ${columns.map(col => this.renderHeader(col)).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        <!-- My Team section -->
                        <tr class="team-header-row">
                            <td colspan="${columns.length}">
                                <span class="team-label">My Team</span>
                                <span class="team-score">${this.myTeamScore}</span>
                                <span class="team-badge ${myResult.toLowerCase()}">${myResult}</span>
                            </td>
                        </tr>
                        ${myTeam.map(p => this.renderPlayerRow(p)).join('')}

                        <!-- Enemy Team section -->
                        <tr class="team-header-row">
                            <td colspan="${columns.length}">
                                <span class="team-label">Enemy Team</span>
                                <span class="team-score">${this.enemyTeamScore}</span>
                                <span class="team-badge ${enemyResult.toLowerCase()}">${enemyResult}</span>
                            </td>
                        </tr>
                        ${enemyTeam.map(p => this.renderPlayerRow(p)).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    /**
     * Render a table header cell
     * @param {Object} col - Column definition
     * @returns {string} HTML string
     */
    renderHeader(col) {
        const sortClass = col.sortable ? 'sortable-header' : '';
        const activeClass = this.sortCol === col.key ? 'active' : '';
        const dirIndicator = this.sortCol === col.key
            ? (this.sortDir === 'asc' ? ' &#9650;' : ' &#9660;')
            : '';

        return `<th class="${sortClass} ${activeClass}" data-col="${col.key}">
            ${col.sortable ? '<span class="sort-icon">&#9881;</span> ' : ''}${this.escapeHtml(col.label)}${dirIndicator}
        </th>`;
    }

    /**
     * Render a player row
     * @param {Object} player - Player data
     * @returns {string} HTML string
     */
    renderPlayerRow(player) {
        const rating = this.getPlayerValue(player, 'hltv_rating');
        const perf = this.getPlayerValue(player, 'personal_performance');
        const kd = this.getPlayerValue(player, 'kd_ratio');
        const adr = this.getPlayerValue(player, 'adr');
        const aim = this.getPlayerValue(player, 'aim_rating');
        const util = this.getPlayerValue(player, 'utility_rating');

        return `
            <tr class="player-row" data-steamid="${player.steam_id || player.steamid}">
                <td class="player-cell">
                    <div class="player-avatar-small">${this.escapeHtml((player.name || 'U')[0].toUpperCase())}</div>
                    <span class="player-name">${this.escapeHtml(player.name || 'Unknown')}</span>
                </td>
                <td class="${this.getHLTVClass(rating)}">
                    ${this.isBest('hltv_rating', player) ? '<span class="best-star">&#9733;</span>' : ''}
                    ${this.formatRating(rating)}
                </td>
                <td class="${this.getPerformanceClass(perf)}">
                    ${perf > 0 ? '+' : ''}${perf.toFixed(2)}
                </td>
                <td class="${this.getKDClass(kd)}">
                    ${this.isBest('kd_ratio', player) ? '<span class="best-star">&#9733;</span>' : ''}
                    ${kd.toFixed(2)}
                </td>
                <td class="${this.getADRClass(adr)}">
                    ${this.isBest('adr', player) ? '<span class="best-star">&#9733;</span>' : ''}
                    ${adr.toFixed(0)}
                </td>
                <td class="${this.getAimClass(aim)}">
                    ${this.isBest('aim_rating', player) ? '<span class="best-star">&#9733;</span>' : ''}
                    ${aim || 0}
                </td>
                <td class="${this.getUtilityClass(util)}">
                    ${this.isBest('utility_rating', player) ? '<span class="best-star">&#9733;</span>' : ''}
                    ${util || 0}
                </td>
            </tr>
        `;
    }

    /**
     * Attach click listeners for sorting
     */
    attachSortListeners() {
        if (!this.container) return;

        this.container.querySelectorAll('th.sortable-header').forEach(th => {
            th.addEventListener('click', () => {
                const col = th.dataset.col;
                if (this.sortCol === col) {
                    this.sortDir = this.sortDir === 'desc' ? 'asc' : 'desc';
                } else {
                    this.sortCol = col;
                    this.sortDir = 'desc';
                }
                this.render();
            });
        });
    }
}

/**
 * Render the overview tab with podium and sortable table
 * This function can be called from index.html
 * @param {Array} players - All players from analysis
 * @param {Array} myTeamIds - Steam IDs for user's team
 * @param {number} myTeamScore - My team score
 * @param {number} enemyTeamScore - Enemy team score
 * @param {string} containerId - Container element ID
 */
function renderOverviewTab(players, myTeamIds, myTeamScore, enemyTeamScore, containerId = 'overview-podium') {
    const container = document.getElementById(containerId);
    if (!container || !players || players.length === 0) return;

    const overview = new OverviewTab(containerId);
    overview.setData(players, myTeamIds, myTeamScore, enemyTeamScore);
    return overview;
}


