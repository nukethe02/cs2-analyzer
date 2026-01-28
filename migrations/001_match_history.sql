-- OpenSight Match History Schema
-- Migration 001: Add match_history and player_baselines tables
-- Purpose: Support "Your Match" feature with Leetify-style personal performance tracking

-- ============================================================================
-- Match History Table
-- Stores individual match data for each player for historical comparison
-- ============================================================================

CREATE TABLE IF NOT EXISTS match_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    steam_id TEXT NOT NULL,
    demo_hash TEXT NOT NULL,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Match metadata
    map_name TEXT,
    result TEXT,  -- 'win', 'loss', 'draw'

    -- Core stats
    kills INTEGER DEFAULT 0,
    deaths INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    adr REAL DEFAULT 0.0,
    kast REAL DEFAULT 0.0,
    hs_pct REAL DEFAULT 0.0,

    -- Ratings
    aim_rating REAL DEFAULT 50.0,
    utility_rating REAL DEFAULT 50.0,
    hltv_rating REAL DEFAULT 1.0,
    opensight_rating REAL DEFAULT 50.0,

    -- Trade stats
    trade_kill_opportunities INTEGER DEFAULT 0,
    trade_kill_attempts INTEGER DEFAULT 0,
    trade_kill_success INTEGER DEFAULT 0,

    -- Entry stats
    entry_attempts INTEGER DEFAULT 0,
    entry_success INTEGER DEFAULT 0,

    -- Clutch stats
    clutch_situations INTEGER DEFAULT 0,
    clutch_wins INTEGER DEFAULT 0,
    clutch_kills INTEGER DEFAULT 0,

    -- Utility stats
    he_damage INTEGER DEFAULT 0,
    enemies_flashed INTEGER DEFAULT 0,
    flash_assists INTEGER DEFAULT 0,

    -- Advanced metrics
    ttd_median_ms REAL,
    cp_median_deg REAL,

    -- Rounds
    rounds_played INTEGER DEFAULT 0,

    -- Prevent duplicate entries
    UNIQUE(steam_id, demo_hash)
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_match_history_steam ON match_history(steam_id);
CREATE INDEX IF NOT EXISTS idx_match_history_steam_analyzed ON match_history(steam_id, analyzed_at);
CREATE INDEX IF NOT EXISTS idx_match_history_demo ON match_history(demo_hash);


-- ============================================================================
-- Player Baselines Table
-- Stores rolling averages for each metric to compare against current match
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    steam_id TEXT NOT NULL,
    metric TEXT NOT NULL,

    -- Statistical values
    avg_value REAL DEFAULT 0.0,
    std_value REAL DEFAULT 0.0,  -- Standard deviation for percentile calculation
    min_value REAL,
    max_value REAL,

    -- Sample tracking
    sample_count INTEGER DEFAULT 0,

    -- Rolling window (last 30 matches by default)
    window_size INTEGER DEFAULT 30,

    -- Last update timestamp
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Unique constraint per player-metric combination
    UNIQUE(steam_id, metric)
);

-- Index for efficient baseline lookup
CREATE INDEX IF NOT EXISTS idx_player_baselines_steam ON player_baselines(steam_id);
CREATE INDEX IF NOT EXISTS idx_player_baselines_steam_metric ON player_baselines(steam_id, metric);


-- ============================================================================
-- Player Personas Table
-- Tracks assigned personas and their confidence levels
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_personas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    steam_id TEXT NOT NULL UNIQUE,

    -- Current persona
    current_persona TEXT DEFAULT 'the_competitor',
    persona_confidence REAL DEFAULT 0.0,

    -- Persona history (JSON array of last 10 personas)
    persona_history_json TEXT,

    -- Top traits for persona determination
    primary_trait TEXT,
    secondary_trait TEXT,

    -- Last calculation timestamp
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_player_personas_steam ON player_personas(steam_id);
