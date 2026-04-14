-- EC Price Prediction Database Schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Raw transactions from URA API
CREATE TABLE IF NOT EXISTS ura_transactions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project         TEXT NOT NULL,
    street          TEXT,
    x               FLOAT,
    y               FLOAT,
    market_segment  TEXT,
    area            FLOAT,
    floor_range     TEXT,
    no_of_units     INT,
    contract_date   TEXT,
    type_of_sale    TEXT,
    price           NUMERIC(15, 2),
    property_type   TEXT,
    district        TEXT,
    type_of_area    TEXT,
    tenure          TEXT,
    nett_price      NUMERIC(15, 2),
    ingested_at     TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_transactions_property_type ON ura_transactions(property_type);
CREATE INDEX idx_transactions_contract_date ON ura_transactions(contract_date);
CREATE INDEX idx_transactions_project ON ura_transactions(project);
CREATE INDEX idx_transactions_type_of_sale ON ura_transactions(type_of_sale);

-- Feature store for model training
CREATE TABLE IF NOT EXISTS ec_features (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id      UUID REFERENCES ura_transactions(id),
    project             TEXT NOT NULL,
    district            TEXT,
    area_sqm            FLOAT,
    floor_mid           INT,
    lease_commence_year INT,
    years_from_launch   INT,
    type_of_sale        TEXT,
    market_segment      TEXT,
    price_psm           FLOAT,
    price               NUMERIC(15, 2),
    created_at          TIMESTAMP DEFAULT NOW()
);

-- Model metadata registry
CREATE TABLE IF NOT EXISTS model_registry (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    algorithm       TEXT,
    metrics         JSONB,
    parameters      JSONB,
    artifact_path   TEXT,
    is_active       BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Prediction logs for monitoring
CREATE TABLE IF NOT EXISTS prediction_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version   TEXT NOT NULL,
    input_features  JSONB NOT NULL,
    predicted_price NUMERIC(15, 2),
    latency_ms      FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);
