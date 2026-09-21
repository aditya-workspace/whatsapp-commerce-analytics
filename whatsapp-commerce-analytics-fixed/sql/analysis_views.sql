-- WhatsApp Commerce Analytics — analysis layer
-- Four views mapped directly onto the JD's "what you will work on" bullets:
--   1. merchant_funnel        -> "merchant activity funnels"
--   2. fulfillment_summary    -> "order fulfillment rates"
--   3. merchant_cohort_retention -> "cohort analysis... merchant growth"
--   4. agent_containment      -> "AI agent response accuracy"

-- ---------------------------------------------------------------------
-- 1. Merchant activation funnel: signup -> first conversation -> first
--    order -> 10th order. Uses window functions to find nth-order dates
--    per merchant without a self-join.
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS merchant_funnel CASCADE;
CREATE VIEW merchant_funnel AS
WITH order_seq AS (
    SELECT
        merchant_id,
        created_at,
        ROW_NUMBER() OVER (PARTITION BY merchant_id ORDER BY created_at) AS order_rank
    FROM orders
),
first_conv AS (
    SELECT merchant_id, MIN(ts) AS first_conversation_at
    FROM conversations
    GROUP BY merchant_id
)
SELECT
    m.merchant_id,
    m.signup_date,
    fc.first_conversation_at,
    MIN(CASE WHEN os.order_rank = 1 THEN os.created_at END)  AS first_order_at,
    MIN(CASE WHEN os.order_rank = 10 THEN os.created_at END) AS tenth_order_at,
    COUNT(os.merchant_id) AS lifetime_orders
FROM merchants m
LEFT JOIN first_conv fc ON fc.merchant_id = m.merchant_id
LEFT JOIN order_seq os ON os.merchant_id = m.merchant_id
GROUP BY m.merchant_id, m.signup_date, fc.first_conversation_at;

-- ---------------------------------------------------------------------
-- 2. Fulfillment rate and time-to-fulfil by merchant category
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS fulfillment_summary CASCADE;
CREATE VIEW fulfillment_summary AS
SELECT
    m.category,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN o.status = 'fulfilled' THEN 1 ELSE 0 END) AS fulfilled_orders,
    ROUND(100.0 * SUM(CASE WHEN o.status = 'fulfilled' THEN 1 ELSE 0 END) / COUNT(*), 1) AS fulfillment_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END) / COUNT(*), 1) AS cancellation_rate_pct,
    PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY EXTRACT(EPOCH FROM (o.fulfilled_at - o.created_at)) / 3600.0
    ) FILTER (WHERE o.status = 'fulfilled') AS median_fulfillment_hours
FROM orders o
JOIN merchants m ON m.merchant_id = o.merchant_id
GROUP BY m.category
ORDER BY fulfillment_rate_pct ASC;

-- ---------------------------------------------------------------------
-- 3. Weekly merchant cohort retention: of merchants who signed up in
--    week W, what share placed >=1 order in each subsequent week offset?
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS merchant_cohort_retention CASCADE;
CREATE VIEW merchant_cohort_retention AS
WITH merchant_weeks AS (
    SELECT merchant_id, DATE_TRUNC('week', signup_date) AS cohort_week
    FROM merchants
),
order_weeks AS (
    SELECT DISTINCT merchant_id, DATE_TRUNC('week', created_at) AS order_week
    FROM orders
),
joined AS (
    SELECT
        mw.cohort_week,
        mw.merchant_id,
        ow.order_week,
        (DATE_PART('day', ow.order_week - mw.cohort_week) / 7)::INT AS week_offset
    FROM merchant_weeks mw
    JOIN order_weeks ow ON ow.merchant_id = mw.merchant_id
    WHERE ow.order_week >= mw.cohort_week
),
cohort_sizes AS (
    SELECT cohort_week, COUNT(*) AS cohort_size
    FROM merchant_weeks
    GROUP BY cohort_week
)
SELECT
    j.cohort_week,
    cs.cohort_size,
    j.week_offset,
    COUNT(DISTINCT j.merchant_id) AS active_merchants,
    ROUND(100.0 * COUNT(DISTINCT j.merchant_id) / cs.cohort_size, 1) AS retention_pct
FROM joined j
JOIN cohort_sizes cs ON cs.cohort_week = j.cohort_week
WHERE j.week_offset BETWEEN 0 AND 12
GROUP BY j.cohort_week, cs.cohort_size, j.week_offset
ORDER BY j.cohort_week, j.week_offset;

-- ---------------------------------------------------------------------
-- 4. Agent containment: share of conversations resolved by the AI agent
--    without human escalation, by intent, plus a rough accuracy proxy
--    (only meaningful here because true_intent exists in this synthetic
--    dataset for evaluation purposes — see README for the production
--    equivalent: sampled human-reviewed labels).
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS agent_containment CASCADE;
CREATE VIEW agent_containment AS
SELECT
    predicted_intent,
    COUNT(*) AS total_conversations,
    ROUND(100.0 * SUM(CASE WHEN escalated_to_human THEN 0 ELSE 1 END) / COUNT(*), 1) AS containment_rate_pct,
    ROUND(100.0 * SUM(CASE WHEN predicted_intent = true_intent THEN 1 ELSE 0 END) / COUNT(*), 1) AS accuracy_pct,
    ROUND(AVG(confidence), 3) AS avg_confidence
FROM conversations
GROUP BY predicted_intent
ORDER BY total_conversations DESC;
