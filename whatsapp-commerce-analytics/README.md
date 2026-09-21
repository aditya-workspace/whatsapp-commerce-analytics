# WhatsApp Commerce Analytics

Operational analytics for a WhatsApp-first SMB ordering platform: merchant activation funnels, order fulfillment, cohort retention, and — the part that's usually missing from a data analyst portfolio — **AI conversational agent accuracy analysis**, including where to set the human-escalation confidence threshold.

**[Live dashboard →](#)** *(deploy link — see Setup)* · [SQL views](sql/analysis_views.sql) · [Agent accuracy notebook](notebooks/agent_accuracy_eda.py)

## Key findings

*(From the reference run committed in this repo — 300 merchants, 6 months, 14.8k conversations, 4.4k orders. Regenerate with a different `--seed`/volume and these will shift slightly; the patterns hold.)*

- **Agent accuracy varies sharply by intent, and it's not the high-volume intents that are the problem.** `place_order` and `track_order` — over half of all volume — run at 93%+ accuracy. But `get_refund` (68% precision) and `complaint` (73% precision) are the weak points, and they're exactly the intents where a wrong auto-response is most costly to a customer relationship.
- **A ~0.45–0.50 confidence threshold is the escalation cutoff that matters.** Below it, error rate on auto-handled conversations exceeds 5%; above 0.85 it drops under 1%. Escalating anything under ~0.45 to a human keeps ~94% of volume automated while cutting silent misclassification meaningfully — a concrete, data-derived rule rather than "add more training data."
- **`place_order` → `check_payment_methods` is the single largest confusion pair** (329 misclassifications) — customers asking a payment question inside an order-placing message. This looks like an intent-boundary/prompt fix, not a model problem.
- **Merchant activation funnel has real drop-off, and it's a repeat-usage problem, not a discovery problem**: 88% of signed-up merchants (265/300) place a first order, but only 50% (151/300) reach their 10th order — the gap opens up *after* merchants have already tried the platform once.

## Why this project

Built for evaluating fit with WhatsApp-first commerce/agent-analytics roles (ordflo-style: PostgreSQL + BI dashboards + conversational AI metrics). The goal was to cover the specific gaps a SQL-Server-and-PowerBI portfolio usually has: Postgres, Python/Pandas EDA, a live deployed dashboard, and analysis of AI agent performance rather than just transactional data.

## Methodology — what's real data vs. simulated

Real WhatsApp Business message logs are private customer data and aren't publicly available (Meta's platform terms don't permit release). Rather than either giving up or faking it wholesale, this project is a **calibrated hybrid**:

| Component | Source |
|---|---|
| Intent taxonomy & base rates | Modeled on the [Bitext Customer Support Intent dataset](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants) (20k+ real utterances, 27 intents) — restricted to the subset relevant to order-taking commerce agents |
| Order volume / fulfillment-time shape | Calibrated against the [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (real transactions) |
| Merchant signup/churn pattern | Exponential signup arrival + ~28% early-churn rate, reflecting typical SMB platform activation curves |
| **AI agent prediction accuracy** | **Simulated with a documented, non-uniform error model** (see `simulate_agent_prediction()` in `scripts/generate_data.py`): correctness probability varies by intent difficulty, confidence is drawn from Beta distributions that differ for correct vs. incorrect predictions (so confidence is correlated with, but doesn't perfectly determine, correctness — the same shape a real classifier produces), and misclassifications are drawn from a semantic confusion map rather than uniformly at random |

This is the one part that has to be simulated, since it's the exact thing being evaluated — but it's built so the resulting precision/recall/threshold analysis is a meaningful exercise in interpreting classifier behavior, not just decoration on random numbers.

## Architecture

```
scripts/generate_data.py   → merchants.csv, conversations.csv, orders.csv
scripts/load_data.py       → loads CSVs into Postgres
sql/schema.sql             → table DDL
sql/analysis_views.sql     → 4 views: merchant_funnel, fulfillment_summary,
                              merchant_cohort_retention, agent_containment
notebooks/agent_accuracy_eda.py → Pandas/NumPy precision-recall + threshold
                                   sweep + confusion analysis, saves charts
dashboard/app.py           → Streamlit app reading the 4 SQL views
```

## Setup

```bash
# 1. Generate data
cd scripts && python3 generate_data.py --out ../data --merchants 300 --months 6

# 2. Load into Postgres (adjust --dsn for your instance)
python3 load_data.py --dsn "dbname=whatsapp_commerce user=postgres password=postgres host=localhost"

# 3. Build analysis views
psql -d whatsapp_commerce -f ../sql/analysis_views.sql

# 4. Run the EDA notebook (generates charts used by the dashboard)
cd ../notebooks && python3 agent_accuracy_eda.py

# 5. Run the dashboard
cd ../dashboard && streamlit run app.py
```

### Deploying the live dashboard

1. Push this repo to GitHub.
2. Spin up a free hosted Postgres instance (e.g. [Neon](https://neon.tech) or [Supabase](https://supabase.com)) and run steps 2–4 above against it.
3. Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud), pointing at `dashboard/app.py`, with `DB_DSN` set as a secret to your hosted instance's connection string.

## What I'd do with real data

- Replace `true_intent` (only knowable here because the data is synthetic) with a sampled, human-reviewed label set for ongoing accuracy monitoring in production.
- Add a BigQuery layer for longer-horizon/cross-platform analysis, per the JD's stack.
- A/B test the recommended 0.50 escalation threshold against actual downstream outcomes (does a wrongly-auto-handled order cost more than an unnecessary human escalation?) rather than optimizing error rate alone.

## Tech

Python (Pandas, NumPy, psycopg2), PostgreSQL (CTEs, window functions, `PERCENTILE_CONT`), Streamlit, Matplotlib/Seaborn.
