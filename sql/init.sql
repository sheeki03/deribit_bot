-- Initialize database schema for Deribit Option Flows System

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Articles table
CREATE TABLE IF NOT EXISTS articles (
    article_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source TEXT NOT NULL DEFAULT 'deribit_insights',
    url TEXT UNIQUE NOT NULL,
    slug TEXT,
    title TEXT,
    author TEXT,
    published_at_utc TIMESTAMPTZ,
    published_at_ist TIMESTAMPTZ,
    body_markdown TEXT,
    body_html TEXT,
    content_hash BYTEA,
    image_count INTEGER DEFAULT 0,
    processing_status TEXT DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for articles
CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at_utc);
CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(processing_status);
CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at);

-- Article images table
CREATE TABLE IF NOT EXISTS article_images (
    image_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID REFERENCES articles(article_id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    image_type TEXT CHECK (image_type IN ('greeks_chart', 'flow_heatmap', 'skew_chart', 'price_chart', 'position_diagram', 'unknown')),
    image_hash BYTEA,
    ocr_text TEXT,
    extracted_data JSONB,
    vision_analysis JSONB,
    download_path TEXT,
    file_size_bytes BIGINT,
    width INTEGER,
    height INTEGER,
    processing_status TEXT DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    confidence_score REAL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for article_images
CREATE INDEX IF NOT EXISTS idx_article_images_article_id ON article_images(article_id);
CREATE INDEX IF NOT EXISTS idx_article_images_type ON article_images(image_type);
CREATE INDEX IF NOT EXISTS idx_article_images_status ON article_images(processing_status);

-- Extractions table (per article)
CREATE TABLE IF NOT EXISTS extractions (
    extraction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID REFERENCES articles(article_id) ON DELETE CASCADE,
    tickers_detected TEXT[],
    key_phrases JSONB,
    numbers JSONB,
    skew_terms JSONB,
    gamma_terms JSONB,
    flow_terms JSONB,
    option_strikes JSONB,
    notional_amounts JSONB,
    expiry_dates JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create index for extractions
CREATE INDEX IF NOT EXISTS idx_extractions_article_id ON extractions(article_id);

-- Multimodal scores table
CREATE TABLE IF NOT EXISTS multimodal_scores (
    score_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    article_id UUID REFERENCES articles(article_id) ON DELETE CASCADE,
    asset TEXT CHECK (asset IN ('BTC', 'ETH')),
    text_score REAL CHECK (text_score >= -1 AND text_score <= 1),
    image_score REAL CHECK (image_score >= -1 AND image_score <= 1),
    market_context_score REAL CHECK (market_context_score >= -1 AND market_context_score <= 1),
    meta_score REAL CHECK (meta_score >= -1 AND meta_score <= 1),
    combined_flowscore REAL CHECK (combined_flowscore >= -1 AND combined_flowscore <= 1),
    confidence_level REAL CHECK (confidence_level >= 0 AND confidence_level <= 1),
    signals JSONB,
    component_weights JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (article_id, asset)
);

-- Create indexes for scores
CREATE INDEX IF NOT EXISTS idx_multimodal_scores_article_id ON multimodal_scores(article_id);
CREATE INDEX IF NOT EXISTS idx_multimodal_scores_asset ON multimodal_scores(asset);
CREATE INDEX IF NOT EXISTS idx_multimodal_scores_flowscore ON multimodal_scores(combined_flowscore);
CREATE INDEX IF NOT EXISTS idx_multimodal_scores_confidence ON multimodal_scores(confidence_level);

-- Prices table
CREATE TABLE IF NOT EXISTS prices (
    price_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ts TIMESTAMPTZ NOT NULL,
    asset TEXT NOT NULL,
    price REAL NOT NULL,
    volume REAL,
    source TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for prices
CREATE INDEX IF NOT EXISTS idx_prices_ts_asset ON prices(ts, asset);
CREATE INDEX IF NOT EXISTS idx_prices_asset ON prices(asset);
CREATE INDEX IF NOT EXISTS idx_prices_source ON prices(source);

-- Event returns materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS event_returns AS
WITH article_prices AS (
    SELECT 
        a.article_id,
        a.published_at_utc,
        s.asset,
        s.combined_flowscore,
        s.confidence_level,
        -- Get price at publication time (closest within 1 hour)
        (SELECT p.price 
         FROM prices p 
         WHERE p.asset = s.asset 
           AND p.ts BETWEEN a.published_at_utc - INTERVAL '1 hour' 
                        AND a.published_at_utc + INTERVAL '1 hour'
         ORDER BY ABS(EXTRACT(EPOCH FROM (p.ts - a.published_at_utc)))
         LIMIT 1) as price_at_pub
    FROM articles a
    JOIN multimodal_scores s ON a.article_id = s.article_id
    WHERE a.processing_status = 'completed'
      AND s.confidence_level >= 0.5
),
forward_returns AS (
    SELECT 
        ap.*,
        -- 4h return
        (SELECT p.price 
         FROM prices p 
         WHERE p.asset = ap.asset 
           AND p.ts BETWEEN ap.published_at_utc + INTERVAL '4 hours' - INTERVAL '30 minutes'
                        AND ap.published_at_utc + INTERVAL '4 hours' + INTERVAL '30 minutes'
         ORDER BY ABS(EXTRACT(EPOCH FROM (p.ts - (ap.published_at_utc + INTERVAL '4 hours'))))
         LIMIT 1) as price_4h,
        -- 24h return
        (SELECT p.price 
         FROM prices p 
         WHERE p.asset = ap.asset 
           AND p.ts BETWEEN ap.published_at_utc + INTERVAL '24 hours' - INTERVAL '1 hour'
                        AND ap.published_at_utc + INTERVAL '24 hours' + INTERVAL '1 hour'
         ORDER BY ABS(EXTRACT(EPOCH FROM (p.ts - (ap.published_at_utc + INTERVAL '24 hours'))))
         LIMIT 1) as price_24h,
        -- 72h return
        (SELECT p.price 
         FROM prices p 
         WHERE p.asset = ap.asset 
           AND p.ts BETWEEN ap.published_at_utc + INTERVAL '72 hours' - INTERVAL '2 hours'
                        AND ap.published_at_utc + INTERVAL '72 hours' + INTERVAL '2 hours'
         ORDER BY ABS(EXTRACT(EPOCH FROM (p.ts - (ap.published_at_utc + INTERVAL '72 hours'))))
         LIMIT 1) as price_72h,
        -- 7d return
        (SELECT p.price 
         FROM prices p 
         WHERE p.asset = ap.asset 
           AND p.ts BETWEEN ap.published_at_utc + INTERVAL '7 days' - INTERVAL '4 hours'
                        AND ap.published_at_utc + INTERVAL '7 days' + INTERVAL '4 hours'
         ORDER BY ABS(EXTRACT(EPOCH FROM (p.ts - (ap.published_at_utc + INTERVAL '7 days'))))
         LIMIT 1) as price_7d
    FROM article_prices ap
    WHERE ap.price_at_pub IS NOT NULL
)
SELECT 
    article_id,
    published_at_utc,
    asset,
    combined_flowscore,
    confidence_level,
    price_at_pub,
    price_4h,
    price_24h,
    price_72h,
    price_7d,
    CASE 
        WHEN price_4h IS NOT NULL AND price_at_pub > 0 
        THEN LN(price_4h / price_at_pub) 
        ELSE NULL 
    END as ret_4h,
    CASE 
        WHEN price_24h IS NOT NULL AND price_at_pub > 0 
        THEN LN(price_24h / price_at_pub) 
        ELSE NULL 
    END as ret_24h,
    CASE 
        WHEN price_72h IS NOT NULL AND price_at_pub > 0 
        THEN LN(price_72h / price_at_pub) 
        ELSE NULL 
    END as ret_72h,
    CASE 
        WHEN price_7d IS NOT NULL AND price_at_pub > 0 
        THEN LN(price_7d / price_at_pub) 
        ELSE NULL 
    END as ret_7d
FROM forward_returns;

-- Create indexes for event_returns
CREATE UNIQUE INDEX IF NOT EXISTS idx_event_returns_unique ON event_returns(article_id, asset);
CREATE INDEX IF NOT EXISTS idx_event_returns_flowscore ON event_returns(combined_flowscore);
CREATE INDEX IF NOT EXISTS idx_event_returns_published_at ON event_returns(published_at_utc);

-- Backtests table
CREATE TABLE IF NOT EXISTS backtests (
    backtest_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id TEXT NOT NULL,
    train_start TIMESTAMPTZ NOT NULL,
    train_end TIMESTAMPTZ NOT NULL,
    test_start TIMESTAMPTZ NOT NULL,
    test_end TIMESTAMPTZ NOT NULL,
    strategy_params JSONB NOT NULL,
    performance_metrics JSONB NOT NULL,
    trade_log JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for backtests
CREATE INDEX IF NOT EXISTS idx_backtests_run_id ON backtests(run_id);
CREATE INDEX IF NOT EXISTS idx_backtests_created_at ON backtests(created_at);

-- System health monitoring table
CREATE TABLE IF NOT EXISTS system_health (
    health_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('healthy', 'warning', 'critical')),
    metrics JSONB NOT NULL,
    message TEXT,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create index for system_health
CREATE INDEX IF NOT EXISTS idx_system_health_component ON system_health(component);
CREATE INDEX IF NOT EXISTS idx_system_health_checked_at ON system_health(checked_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_articles_updated_at BEFORE UPDATE ON articles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_article_images_updated_at BEFORE UPDATE ON article_images FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();