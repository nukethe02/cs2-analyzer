/**
 * Enhanced Stats Tables for OpenSight
 *
 * Provides Leetify-style stats tables with:
 * - Sortable columns (click headers)
 * - Star indicators for best-in-match stats
 * - Color-coded K/D and performance values
 * - Team groupings with win/loss badges
 * - Player avatars with initials
 *
 * Usage:
 *   const table = new StatsTable('container-id', columns);
 *   table.setData(players, myTeamIds, myTeamScore, enemyTeamScore);
 */

class StatsTable {
    /**
     * Create a new stats table
     * @param {string} containerId - ID of the container element
     * @param {Array} columns - Column definitions
     */
    constructor(containerId, columns = null) {
        this.container = document.getElementById(containerId);
        this.columns = columns || this.getDefaultColumns();
        this.sortCol = 'hltv_rating';
        this.sortDir = 'desc';
        this.data = [];
        this.myTeam = new Set();
        this.myTeamScore = 0;
        this.enemyTeamScore = 0;
        this.best = {};
    }

    /**
     * Get default column definitions
     * @returns {Array} Column configuration
     */
    getDefaultColumns() {
        return [
            { key: 'name', label: 'Player', numeric: false, sortable: false, highlight: false },
            { key: 'kills', label: 'K', numeric: true, sortable: true, highlight: true },
            { key: 'assists', label: 'A', numeric: true, sortable: true, highlight: false },
            { key: 'deaths', label: 'D', numeric: true, sortable: true, highlight: false, invert: true },
            { key: 'kd', label: 'K/D', numeric: true, sortable: true, highlight: true, format: 'decimal', colorCode: true },
            { key: 'adr', label: 'ADR', numeric: true, sortable: true, highlight: true, format: 'decimal' },
            { key: 'multi_2k', label: '2K', numeric: true, sortable: true, highlight: false },
            { key: 'multi_3k', label: '3K', numeric: true, sortable: true, highlight: true },
            { key: 'multi_4k', label: '4K', numeric: true, sortable: true, highlight: true },
            { key: 'multi_5k', label: '5K', numeric: true, sortable: true, highlight: true },
            { key: 'hltv_rating', label: 'Rating', numeric: true, sortable: true, highlight: true, format: 'decimal' },
            { key: 'performance', label: '+/-', numeric: true, sortable: true, highlight: false, format: 'plusminus', colorCode: true }
        ];
    }

    /**
     * Set the data for the table
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
     * Calculate best values for each column
     */
    calcBestValues() {
        this.best = {};
        this.columns.forEach(col => {
            if (col.numeric && col.highlight) {
                const values = this.data.map(p => this.getPlayerValue(p, col.key) || 0);
                if (col.invert) {
                    // Lower is better
                    this.best[col.key] = Math.min(...values);
                } else {
                    this.best[col.key] = Math.max(...values);
                }
            }
        });
    }

    /**
     * Get a value from a player object
     * @param {Object} player - Player data
     * @param {string} key - Property key
     * @returns {*} Value
     */
    getPlayerValue(player, key) {
        // Handle nested properties like multi_kills.2k
        if (key.includes('.')) {
            const parts = key.split('.');
            let val = player;
            for (const part of parts) {
                val = val?.[part];
            }
            return val;
        }
        return player[key];
    }

    /**
     * Render the complete table
     */
    render() {
        if (!this.container) return;

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
            <div class="stats-table-wrapper">
                ${this.renderTeamSection(sortedMy, 'My Team', myResult, this.myTeamScore)}
                ${this.renderTeamSection(sortedEnemy, 'Enemy Team', enemyResult, this.enemyTeamScore)}
            </div>
        `;

        this.attachSortListeners();
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
     * Render a team section with header and table
     * @param {Array} players - Team players
     * @param {string} label - Team label
     * @param {string} result - WIN, LOSS, or TIE
     * @param {number} score - Team score
     * @returns {string} HTML string
     */
    renderTeamSection(players, label, result, score) {
        if (!players || players.length === 0) {
            return '';
        }

        return `
            <div class="team-section">
                <div class="team-header">
                    <span class="team-name">${this.escapeHtml(label)}</span>
                    <div class="team-result-group">
                        <span class="team-score">${score}</span>
                        <span class="result-badge ${result.toLowerCase()}">${result}</span>
                    </div>
                </div>
                <div class="table-container">
                    <table class="enhanced-stats-table">
                        <thead>
                            <tr>
                                ${this.columns.map(col => this.renderHeader(col)).join('')}
                            </tr>
                        </thead>
                        <tbody>
                            ${players.map(p => this.renderRow(p)).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }

    /**
     * Render a table header cell
     * @param {Object} col - Column definition
     * @returns {string} HTML string
     */
    renderHeader(col) {
        const sortClass = col.sortable ? 'sortable' : '';
        const activeClass = this.sortCol === col.key ? 'active' : '';
        const dirIndicator = this.sortCol === col.key
            ? (this.sortDir === 'asc' ? ' &#9650;' : ' &#9660;')
            : '';

        return `<th class="${sortClass} ${activeClass}" data-col="${col.key}">
            ${this.escapeHtml(col.label)}${dirIndicator}
        </th>`;
    }

    /**
     * Render a table row for a player
     * @param {Object} player - Player data
     * @returns {string} HTML string
     */
    renderRow(player) {
        return `<tr>
            ${this.columns.map(col => this.renderCell(player, col)).join('')}
        </tr>`;
    }

    /**
     * Render a table cell
     * @param {Object} player - Player data
     * @param {Object} col - Column definition
     * @returns {string} HTML string
     */
    renderCell(player, col) {
        const val = this.getPlayerValue(player, col.key);

        // Special handling for player name column
        if (col.key === 'name') {
            return this.renderPlayerCell(player);
        }

        const isBest = col.highlight && this.best[col.key] === val && val > 0;
        const colorClass = this.getColorClass(col, val);
        const formattedVal = this.formatValue(col, val);

        return `<td class="${colorClass}">
            ${isBest ? '<span class="best-indicator">&#9733;</span>' : ''}
            <span class="cell-value">${formattedVal}</span>
        </td>`;
    }

    /**
     * Render the player name cell with avatar
     * @param {Object} player - Player data
     * @returns {string} HTML string
     */
    renderPlayerCell(player) {
        const name = player.name || 'Unknown';
        const initial = (player.avatar_initial || name[0] || '?').toUpperCase();

        return `<td class="player-cell">
            <div class="player-info">
                <div class="player-avatar">${this.escapeHtml(initial)}</div>
                <span class="player-name">${this.escapeHtml(name)}</span>
            </div>
        </td>`;
    }

    /**
     * Get CSS class for color-coded values
     * @param {Object} col - Column definition
     * @param {*} val - Cell value
     * @returns {string} CSS class name
     */
    getColorClass(col, val) {
        if (!col.colorCode) return '';

        if (col.key === 'kd') {
            return val >= 1.0 ? 'positive' : 'negative';
        }
        if (col.key === 'performance') {
            return val >= 0 ? 'positive' : 'negative';
        }
        return '';
    }

    /**
     * Format a value for display
     * @param {Object} col - Column definition
     * @param {*} val - Raw value
     * @returns {string} Formatted value
     */
    formatValue(col, val) {
        if (val == null || val === undefined) return 'N/A';

        switch (col.format) {
            case 'decimal':
                return typeof val === 'number' ? val.toFixed(2) : val;
            case 'plusminus':
                const num = typeof val === 'number' ? val : parseFloat(val);
                if (isNaN(num)) return val;
                return (num >= 0 ? '+' : '') + num.toFixed(2);
            case 'percent':
                return typeof val === 'number' ? val.toFixed(1) + '%' : val;
            default:
                return String(val);
        }
    }

    /**
     * Attach click listeners for sorting
     */
    attachSortListeners() {
        if (!this.container) return;

        this.container.querySelectorAll('th.sortable').forEach(th => {
            th.addEventListener('click', () => {
                const col = th.dataset.col;
                if (this.sortCol === col) {
                    // Toggle direction
                    this.sortDir = this.sortDir === 'desc' ? 'asc' : 'desc';
                } else {
                    // New column, default to desc
                    this.sortCol = col;
                    this.sortDir = 'desc';
                }
                this.render();
            });
        });
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
}

/**
 * Create stats table from analysis data
 * @param {string} containerId - Container element ID
 * @param {Object} analysisData - Full analysis response from API
 * @param {Array} myTeamSteamIds - Steam IDs for user's team
 * @returns {StatsTable} The created table instance
 */
function createStatsTableFromAnalysis(containerId, analysisData, myTeamSteamIds = []) {
    const table = new StatsTable(containerId);

    // Get players from analysis data
    const players = analysisData.players || [];

    // Transform to expected format
    const transformedPlayers = players.map(p => ({
        steamid: p.steam_id || p.steamid,
        name: p.name,
        kills: p.kills || 0,
        assists: p.assists || 0,
        deaths: p.deaths || 0,
        kd: p.kd_ratio || (p.deaths > 0 ? p.kills / p.deaths : p.kills),
        adr: p.adr || 0,
        multi_2k: p.multi_kills?.['2k'] || p.rounds_with_2k || 0,
        multi_3k: p.multi_kills?.['3k'] || p.rounds_with_3k || 0,
        multi_4k: p.multi_kills?.['4k'] || p.rounds_with_4k || 0,
        multi_5k: p.multi_kills?.['5k'] || p.rounds_with_5k || 0,
        hltv_rating: p.hltv_rating || 0,
        performance: calculatePersonalPerformance(p),
        avatar_initial: (p.name || 'U')[0]
    }));

    // Get scores
    const team1Score = analysisData.team1_score || 0;
    const team2Score = analysisData.team2_score || 0;

    // Determine my team score
    const myTeamSet = new Set(myTeamSteamIds.map(id => String(id)));
    const isTeam1 = transformedPlayers.some(p =>
        myTeamSet.has(String(p.steamid)) &&
        players.find(op => String(op.steam_id) === String(p.steamid))?.team === analysisData.team1
    );

    const myScore = isTeam1 ? team1Score : team2Score;
    const enemyScore = isTeam1 ? team2Score : team1Score;

    table.setData(transformedPlayers, myTeamSteamIds, myScore, enemyScore);

    return table;
}

/**
 * Calculate personal performance for a player
 * @param {Object} p - Player data
 * @returns {number} Performance value
 */
function calculatePersonalPerformance(p) {
    const kills = p.kills || 0;
    const deaths = p.deaths || 0;
    const assists = p.assists || 0;
    const rounds = p.rounds_played || 1;

    const expectedKills = rounds * 0.7;
    const expectedDeaths = rounds * 0.7;
    const assistValue = assists * 0.3;

    const killDiff = kills - expectedKills;
    const deathDiff = expectedDeaths - deaths;

    return ((killDiff + deathDiff * 0.5 + assistValue) / rounds * 10);
}

/**
 * Render the Clutches tab with team summaries and player clutch cards
 * @param {Array} players - All players from analysis
 * @param {Array} myTeam - Players on user's team
 * @param {Array} enemyTeam - Players on enemy team
 * @param {string} containerId - Container element ID
 */
function renderClutchesTab(players, myTeam, enemyTeam, containerId = 'clutches-content') {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Helper to escape HTML
    const escapeHtml = (str) => {
        if (typeof str !== 'string') return str;
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    };

    // Calculate team clutch summary
    const calculateTeamSummary = (teamPlayers) => {
        let totalWins = 0;
        let totalLosses = 0;
        let totalSaves = 0;
        let totalKills = 0;

        teamPlayers.forEach(player => {
            const clutches = player.clutches || {};
            const details = clutches.details || [];

            details.forEach(c => {
                totalKills += c.enemies_killed || 0;
                if (c.outcome === 'won') totalWins++;
                else if (c.outcome === 'lost') totalLosses++;
                else if (c.outcome === 'saved') totalSaves++;
            });
        });

        const total = totalWins + totalLosses + totalSaves;
        const wonPct = total > 0 ? Math.round((totalWins / total) * 100) : 0;
        const lostPct = total > 0 ? Math.round((totalLosses / total) * 100) : 0;

        // Rating thresholds
        let killsRating = 'Poor';
        if (totalKills > 10) killsRating = 'Great';
        else if (totalKills >= 5) killsRating = 'Average';

        return { totalWins, totalLosses, totalSaves, totalKills, wonPct, lostPct, killsRating };
    };

    // Render team summary bar
    const renderTeamSummary = (summary) => {
        return `
            <div class="clutch-summary">
                <div class="clutch-percentages">
                    <span class="clutch-won-pct">${summary.wonPct}%</span>
                    <div class="clutch-bar">
                        <div class="clutch-bar-won" style="width: ${summary.wonPct}%"></div>
                        <div class="clutch-bar-lost" style="width: ${summary.lostPct}%"></div>
                    </div>
                    <span class="clutch-lost-pct">${summary.lostPct}%</span>
                </div>
                <div class="clutch-total-kills">
                    <div class="clutch-kills-circle">
                        <span class="kills-count">${summary.totalKills}</span>
                        <span class="kills-label">TOTAL KILLS</span>
                    </div>
                    <span class="kills-rating kills-rating-${summary.killsRating.toLowerCase()}">${summary.killsRating}</span>
                </div>
                <div class="clutch-saves">
                    ${summary.totalWins} CLUTCHES WON | ${summary.totalLosses} CLUTCHES LOST | ${summary.totalSaves} SAVES
                </div>
            </div>
        `;
    };

    // Render individual clutch card
    const renderClutchCard = (clutch) => {
        const outcome = clutch.outcome || 'lost';
        const type = clutch.type || '1v1';
        const kills = clutch.enemies_killed || 0;
        const round = clutch.round_number || 0;

        return `
            <div class="clutch-card clutch-${outcome}">
                <div class="clutch-type">${escapeHtml(type)}</div>
                <div class="clutch-kills">
                    <span class="skull-icon">💀</span> ${kills}
                </div>
                <div class="clutch-round">Round ${round}</div>
                <div class="clutch-outcome ${outcome}">${outcome.toUpperCase()}</div>
            </div>
        `;
    };

    // Render player clutch section
    const renderPlayerClutchSection = (player) => {
        const name = player.name || 'Unknown';
        const initial = (name[0] || '?').toUpperCase();
        const clutches = player.clutches || {};
        const details = clutches.details || [];

        // Sort by round number
        const sortedDetails = [...details].sort((a, b) => (a.round_number || 0) - (b.round_number || 0));

        const cardsHtml = sortedDetails.length > 0
            ? sortedDetails.map(c => renderClutchCard(c)).join('')
            : '<div class="no-clutches">No clutch situations</div>';

        return `
            <div class="player-clutch-section">
                <div class="player-header">
                    <div class="player-avatar">${escapeHtml(initial)}</div>
                    <span class="player-name">${escapeHtml(name)}</span>
                </div>
                <div class="clutch-cards-grid">
                    ${cardsHtml}
                </div>
            </div>
        `;
    };

    // Render team section
    const renderTeamSection = (teamPlayers, teamLabel, teamResult, teamScore) => {
        if (!teamPlayers || teamPlayers.length === 0) return '';

        const summary = calculateTeamSummary(teamPlayers);

        return `
            <div class="team-section clutch-team-section">
                <div class="team-header">
                    <span class="team-name">${escapeHtml(teamLabel)}</span>
                    <div class="team-result-group">
                        <span class="team-score">${teamScore || 0}</span>
                        <span class="result-badge ${(teamResult || 'tie').toLowerCase()}">${teamResult || 'TIE'}</span>
                    </div>
                </div>
                ${renderTeamSummary(summary)}
                <div class="player-clutches-list">
                    ${teamPlayers.map(p => renderPlayerClutchSection(p)).join('')}
                </div>
            </div>
        `;
    };

    // Determine team results
    const myTeamScore = myTeam.length > 0 && myTeam[0].team_score !== undefined
        ? myTeam[0].team_score : 0;
    const enemyTeamScore = enemyTeam.length > 0 && enemyTeam[0].team_score !== undefined
        ? enemyTeam[0].team_score : 0;

    const myResult = myTeamScore > enemyTeamScore ? 'WIN' :
                     myTeamScore < enemyTeamScore ? 'LOSS' : 'TIE';
    const enemyResult = enemyTeamScore > myTeamScore ? 'WIN' :
                        enemyTeamScore < myTeamScore ? 'LOSS' : 'TIE';

    container.innerHTML = `
        <div class="clutches-tab-wrapper">
            ${renderTeamSection(myTeam, 'My Team', myResult, myTeamScore)}
            ${renderTeamSection(enemyTeam, 'Enemy Team', enemyResult, enemyTeamScore)}
        </div>
    `;
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { StatsTable, createStatsTableFromAnalysis, renderClutchesTab };
}
