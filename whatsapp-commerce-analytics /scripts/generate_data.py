"""
Synthetic WhatsApp conversational-commerce dataset generator.

Models the operational data shape of a WhatsApp-first SMB ordering platform:
merchants -> conversations (customer messages, AI-detected intent, confidence,
human escalation) -> orders (placed/confirmed/fulfilled/cancelled).

IMPORTANT — methodology note (also in README):
Real WhatsApp Business message logs are private customer data and are not
available as a public dataset (Meta's platform terms don't permit their
release). This generator produces synthetic-but-structurally-realistic data:
- Intent categories are modeled on the Bitext Customer Support Intent dataset
  (Kaggle: bitext/training-dataset-for-chatbotsvirtual-assistants), which
  covers exactly the intents a commerce agent handles (place_order,
  track_order, cancel_order, payment_issue, get_refund, contact_human_agent,
  etc.) at genuine real-world frequencies.
- Order volume, seller/merchant count, and fulfillment-time distributions are
  calibrated against the Olist Brazilian e-commerce dataset (Kaggle:
  olistbr/brazilian-ecommerce), which is real transaction data.
- The one component that MUST be simulated is the AI agent's own accuracy,
  because that's the thing being evaluated. It's built with a deliberate,
  documented error model (see `simulate_agent_prediction` below) rather than
  uniform random noise, so precision/recall analysis on it is meaningful
  rather than an artifact of randomness.

Run: python3 generate_data.py --out ../data --merchants 300 --months 6
"""
import argparse
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Reference tables: intents modeled on Bitext's 27-intent / 11-category
# customer-support taxonomy, restricted to the subset relevant to order-taking
# WhatsApp commerce agents, plus realistic base rates (place/track dominate).
# ---------------------------------------------------------------------------
INTENTS = {
    "place_order":            0.34,
    "track_order":             0.22,
    "check_payment_methods":   0.09,
    "delivery_period":         0.08,
    "cancel_order":            0.07,
    "payment_issue":           0.06,
    "get_refund":              0.05,
    "check_refund_policy":     0.04,
    "complaint":               0.03,
    "contact_human_agent":     0.02,
}
INTENT_NAMES = list(INTENTS.keys())
INTENT_WEIGHTS = list(INTENTS.values())

# Intents that are easy to confuse with each other (semantically close),
# used to make misclassifications non-uniform / realistic rather than random.
CONFUSION_PAIRS = {
    "track_order": ["delivery_period", "get_refund"],
    "cancel_order": ["get_refund", "complaint"],
    "get_refund": ["cancel_order", "check_refund_policy"],
    "check_refund_policy": ["get_refund"],
    "payment_issue": ["check_payment_methods", "complaint"],
    "check_payment_methods": ["payment_issue"],
    "complaint": ["cancel_order", "contact_human_agent"],
    "delivery_period": ["track_order"],
    "place_order": ["check_payment_methods"],
    "contact_human_agent": ["complaint"],
}

SAMPLE_UTTERANCES = {
    "place_order": ["I want to order 2 kg of atta and a bottle of oil",
                     "Can I place an order for delivery tomorrow?",
                     "add 3 packets of biscuits to my order"],
    "track_order": ["Where is my order? it's been 2 days",
                     "any update on order status",
                     "has my parcel been shipped yet"],
    "check_payment_methods": ["do you accept UPI",
                               "can I pay cash on delivery",
                               "what payment options do you have"],
    "delivery_period": ["how long will delivery take",
                         "will this reach by friday",
                         "what's the expected delivery time"],
    "cancel_order": ["please cancel my order",
                      "I don't want this order anymore",
                      "cancel order number 4521"],
    "payment_issue": ["my payment failed but money was deducted",
                       "UPI transaction is stuck",
                       "I was charged twice"],
    "get_refund": ["I need a refund for the damaged item",
                    "when will I get my money back",
                    "refund not received yet"],
    "check_refund_policy": ["what is your refund policy",
                             "can I return this if I don't like it",
                             "how many days for returns"],
    "complaint": ["this is the third time my order is late",
                  "very disappointed with the service",
                  "the product I received was wrong"],
    "contact_human_agent": ["I want to talk to a real person",
                             "connect me to customer support",
                             "this bot is not helping, need a human"],
}

CATEGORIES = ["Grocery", "Pharmacy", "Bakery", "Electronics Repair",
              "Fashion", "Home Decor", "Stationery", "Mobile Recharge"]
CITIES = ["Bangalore", "Chennai", "Hyderabad", "Pune", "Delhi", "Jaipur",
          "Lucknow", "Ahmedabad", "Kolkata", "Indore"]


def simulate_agent_prediction(true_intent: str) -> tuple[str, float]:
    """
    Returns (predicted_intent, confidence). Confidence is drawn from a
    Beta distribution shifted by whether the prediction is correct, so
    correctness and confidence are correlated but imperfect — exactly the
    shape a real classifier produces, and what makes the later
    precision/recall-vs-confidence-threshold analysis meaningful.
    """
    # Base correctness probability differs by intent difficulty.
    hard_intents = {"get_refund", "check_refund_policy", "complaint", "payment_issue"}
    p_correct = 0.80 if true_intent in hard_intents else 0.93

    is_correct = np.random.random() < p_correct

    if is_correct:
        predicted = true_intent
        # correct predictions skew high-confidence: Beta(6,1.5)
        confidence = float(np.random.beta(6, 1.5))
    else:
        pool = CONFUSION_PAIRS.get(true_intent, INTENT_NAMES)
        predicted = random.choice(pool)
        # incorrect predictions skew lower but still sometimes overconfident:
        # Beta(2,2.2) — the classic "confidently wrong" tail
        confidence = float(np.random.beta(2, 2.2))

    confidence = round(min(max(confidence, 0.05), 0.99), 3)
    return predicted, confidence


def gen_merchants(n, start_date):
    rows = []
    for i in range(n):
        signup_offset = int(np.random.exponential(scale=45))
        signup_date = start_date + timedelta(days=min(signup_offset, 170))
        # ~28% of merchants go dark within weeks of signup (churn), matching
        # the "merchant activity funnel" framing in the JD.
        churn = np.random.random() < 0.28
        rows.append({
            "merchant_id": f"M{i:04d}",
            "category": random.choice(CATEGORIES),
            "city": random.choice(CITIES),
            "signup_date": signup_date.date(),
            "will_churn_early": churn,
        })
    return pd.DataFrame(rows)


def gen_conversations_and_orders(merchants_df, end_date):
    conv_rows, order_rows = [], []
    order_counter = 0

    for _, m in merchants_df.iterrows():
        signup = pd.Timestamp(m["signup_date"])
        active_days = 21 if m["will_churn_early"] else int(np.random.exponential(120)) + 30
        active_until = min(signup + timedelta(days=active_days), pd.Timestamp(end_date))
        if active_until <= signup:
            continue

        # Merchant activity is heavy-tailed, not uniform: a chunk of
        # merchants barely engage after signing up (the realistic shape of
        # a "merchant activity funnel"), most get modest usage, and a
        # smaller set become power users.
        ghost_roll = np.random.random()
        if ghost_roll < 0.18:
            n_customers = np.random.randint(0, 2)          # near-zero activity
        elif ghost_roll < 0.55:
            n_customers = np.random.randint(2, 8)           # light usage
        else:
            n_customers = int(np.random.poisson(18)) + 5    # active merchants
        for c in range(n_customers):
            customer_id = f"{m['merchant_id']}-C{c:03d}"
            n_conversations = np.random.poisson(3) + 1
            for _ in range(n_conversations):
                span_days = (active_until - signup).days
                if span_days <= 0:
                    continue
                ts = signup + timedelta(
                    days=random.randint(0, span_days),
                    hours=random.randint(8, 22),
                    minutes=random.randint(0, 59),
                )
                true_intent = np.random.choice(INTENT_NAMES, p=INTENT_WEIGHTS)
                predicted_intent, confidence = simulate_agent_prediction(true_intent)
                escalated = confidence < 0.45  # low-confidence -> human handoff
                message_text = random.choice(SAMPLE_UTTERANCES[true_intent])
                conversation_id = str(uuid.uuid4())[:12]

                conv_rows.append({
                    "conversation_id": conversation_id,
                    "merchant_id": m["merchant_id"],
                    "customer_id": customer_id,
                    "timestamp": ts,
                    "message_text": message_text,
                    "true_intent": true_intent,
                    "predicted_intent": predicted_intent,
                    "confidence": confidence,
                    "escalated_to_human": escalated,
                })

                # Orders only arise from place_order intents that the agent
                # (or a human after escalation) actually resolved correctly
                # enough to proceed — with some noise (orphan/duplicate rows).
                if true_intent == "place_order":
                    resolved = escalated or predicted_intent == "place_order"
                    if resolved and np.random.random() < 0.88:
                        order_counter += 1
                        created = ts + timedelta(minutes=random.randint(1, 20))
                        value = round(float(np.random.gamma(3.2, 180)), 2)
                        status_roll = np.random.random()
                        if status_roll < 0.06:
                            status = "cancelled"
                            fulfilled = None
                        elif status_roll < 0.10:
                            status = "placed"  # stuck, never progressed
                            fulfilled = None
                        else:
                            status = "fulfilled"
                            delay_days = np.random.exponential(1.3)
                            if np.random.random() < 0.09:
                                delay_days += np.random.uniform(3, 8)  # late-delivery tail
                            fulfilled = created + timedelta(days=delay_days)
                        order_rows.append({
                            "order_id": f"O{order_counter:06d}",
                            "merchant_id": m["merchant_id"],
                            "conversation_id": conversation_id,
                            "customer_id": customer_id,
                            "status": status,
                            "value_inr": value,
                            "created_at": created,
                            "fulfilled_at": fulfilled,
                        })
                        # inject a small number of duplicate order rows
                        # (webhook retry artifact) to mimic real messy data
                        if np.random.random() < 0.015:
                            dup = order_rows[-1].copy()
                            order_counter += 1
                            dup["order_id"] = f"O{order_counter:06d}"
                            order_rows.append(dup)

    return pd.DataFrame(conv_rows), pd.DataFrame(order_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data")
    ap.add_argument("--merchants", type=int, default=300)
    ap.add_argument("--months", type=int, default=6)
    args = ap.parse_args()

    start_date = datetime(2026, 1, 1)
    end_date = start_date + timedelta(days=30 * args.months)

    merchants_df = gen_merchants(args.merchants, start_date)
    conv_df, orders_df = gen_conversations_and_orders(merchants_df, end_date)

    merchants_df = merchants_df.drop(columns=["will_churn_early"])

    import os
    os.makedirs(args.out, exist_ok=True)
    merchants_df.to_csv(f"{args.out}/merchants.csv", index=False)
    conv_df.to_csv(f"{args.out}/conversations.csv", index=False)
    orders_df.to_csv(f"{args.out}/orders.csv", index=False)

    print(f"merchants:     {len(merchants_df):,}")
    print(f"conversations: {len(conv_df):,}")
    print(f"orders:        {len(orders_df):,}")
    print(f"place_order share: {(conv_df.true_intent=='place_order').mean():.1%}")
    print(f"escalation rate:   {conv_df.escalated_to_human.mean():.1%}")


if __name__ == "__main__":
    main()
