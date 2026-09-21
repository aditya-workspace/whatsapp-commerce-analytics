-- WhatsApp Commerce Analytics — schema
-- Run against Postgres. Designed for a merchant-scoped multi-tenant
-- operational store, similar in shape to what ordflo's stack (Postgres +
-- Firestore + BigQuery) would expose for the conversational/order layer.

DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS merchants CASCADE;

CREATE TABLE merchants (
    merchant_id   TEXT PRIMARY KEY,
    category      TEXT NOT NULL,
    city          TEXT NOT NULL,
    signup_date   DATE NOT NULL
);

CREATE TABLE conversations (
    conversation_id     TEXT PRIMARY KEY,
    merchant_id         TEXT NOT NULL REFERENCES merchants(merchant_id),
    customer_id         TEXT NOT NULL,
    ts                  TIMESTAMP NOT NULL,
    message_text         TEXT,
    true_intent          TEXT,          -- ground truth (known only because data is synthetic;
                                         -- in production this would come from labeled review samples)
    predicted_intent     TEXT NOT NULL, -- what the AI agent classified
    confidence            NUMERIC(4,3) NOT NULL,
    escalated_to_human    BOOLEAN NOT NULL
);

CREATE TABLE orders (
    order_id        TEXT PRIMARY KEY,
    merchant_id     TEXT NOT NULL REFERENCES merchants(merchant_id),
    conversation_id TEXT REFERENCES conversations(conversation_id),
    customer_id     TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('placed','fulfilled','cancelled')),
    value_inr       NUMERIC(10,2) NOT NULL,
    created_at      TIMESTAMP NOT NULL,
    fulfilled_at    TIMESTAMP
);

CREATE INDEX idx_conv_merchant_ts ON conversations (merchant_id, ts);
CREATE INDEX idx_conv_intent ON conversations (predicted_intent);
CREATE INDEX idx_orders_merchant ON orders (merchant_id, created_at);
CREATE INDEX idx_orders_status ON orders (status);
