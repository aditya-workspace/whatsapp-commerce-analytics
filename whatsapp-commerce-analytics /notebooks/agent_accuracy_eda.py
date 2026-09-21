"""
Agent accuracy EDA — the piece that doesn't fit cleanly in SQL.

Answers:
  1. What's the AI agent's overall precision/recall by intent?
  2. Where's the confidence threshold below which auto-handling is riskier
     than escalating to a human?
  3. Which intents get confused with which — where should the model (or
     the prompt) be improved first?

Run: python3 agent_accuracy_eda.py --dsn "dbname=whatsapp_commerce user=postgres password=postgres host=localhost"
Outputs PNGs to ../dashboard/assets/
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns

sns.set_theme(style="whitegrid")


def load(dsn):
    conn = psycopg2.connect(dsn)
    df = pd.read_sql(
        "SELECT true_intent, predicted_intent, confidence, escalated_to_human "
        "FROM conversations", conn)
    conn.close()
    return df


def precision_recall_by_intent(df):
    rows = []
    for intent in sorted(df.true_intent.unique()):
        tp = ((df.predicted_intent == intent) & (df.true_intent == intent)).sum()
        fp = ((df.predicted_intent == intent) & (df.true_intent != intent)).sum()
        fn = ((df.predicted_intent != intent) & (df.true_intent == intent)).sum()
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else np.nan
        rows.append({"intent": intent, "precision": precision, "recall": recall,
                      "f1": f1, "support": (df.true_intent == intent).sum()})
    return pd.DataFrame(rows).sort_values("support", ascending=False)


def threshold_sweep(df):
    """
    For each candidate confidence threshold t: if we escalate everything
    below t to a human instead of letting the agent act, what fraction of
    the AUTO-HANDLED (confidence >= t) predictions are wrong?
    This is the curve that sets the escalation threshold in production.
    """
    thresholds = np.arange(0.30, 0.96, 0.05)
    rows = []
    for t in thresholds:
        auto = df[df.confidence >= t]
        if len(auto) == 0:
            continue
        error_rate = (auto.predicted_intent != auto.true_intent).mean()
        coverage = len(auto) / len(df)
        rows.append({"threshold": round(t, 2), "auto_handled_share": coverage,
                      "error_rate_when_auto_handled": error_rate})
    return pd.DataFrame(rows)


def confusion_top_pairs(df, n=10):
    wrong = df[df.predicted_intent != df.true_intent]
    pairs = (wrong.groupby(["true_intent", "predicted_intent"])
                   .size().reset_index(name="count")
                   .sort_values("count", ascending=False).head(n))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="dbname=whatsapp_commerce user=postgres password=postgres host=localhost")
    ap.add_argument("--out", default="../dashboard/assets")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load(args.dsn)
    print(f"loaded {len(df):,} conversations")

    # 1. Precision/recall per intent
    pr = precision_recall_by_intent(df)
    print("\n== precision/recall by intent ==")
    print(pr.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    pr_sorted = pr.sort_values("f1")
    ax.barh(pr_sorted.intent, pr_sorted.f1, color="#4C72B0")
    ax.set_xlabel("F1 score")
    ax.set_title("Agent classification F1 by intent")
    fig.tight_layout()
    fig.savefig(f"{args.out}/f1_by_intent.png", dpi=130)
    plt.close(fig)

    # 2. Confidence distribution: correct vs incorrect
    fig, ax = plt.subplots(figsize=(8, 5))
    df["is_correct"] = df.predicted_intent == df.true_intent
    sns.kdeplot(data=df[df.is_correct], x="confidence", ax=ax, label="Correct", fill=True, alpha=0.4)
    sns.kdeplot(data=df[~df.is_correct], x="confidence", ax=ax, label="Incorrect", fill=True, alpha=0.4)
    ax.set_title("Model confidence: correct vs incorrect predictions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out}/confidence_distribution.png", dpi=130)
    plt.close(fig)

    # 3. Threshold sweep -> escalation cutoff recommendation
    sweep = threshold_sweep(df)
    print("\n== threshold sweep ==")
    print(sweep.to_string(index=False))
    # Recommended threshold: lowest t where error_rate_when_auto_handled <= 5%
    candidates = sweep[sweep.error_rate_when_auto_handled <= 0.05]
    recommended = candidates.threshold.min() if len(candidates) else sweep.threshold.max()
    print(f"\nRecommended auto-handle confidence floor: {recommended:.2f}")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sweep.threshold, sweep.error_rate_when_auto_handled * 100, "o-", color="#C44E52", label="Error rate if auto-handled")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Error rate (%)", color="#C44E52")
    ax1.axhline(5, ls="--", color="gray", lw=1)
    ax2 = ax1.twinx()
    ax2.plot(sweep.threshold, sweep.auto_handled_share * 100, "s-", color="#4C72B0", label="Share auto-handled")
    ax2.set_ylabel("Share of conversations auto-handled (%)", color="#4C72B0")
    ax1.set_title("Escalation threshold trade-off")
    fig.tight_layout()
    fig.savefig(f"{args.out}/threshold_tradeoff.png", dpi=130)
    plt.close(fig)

    # 4. Top confusion pairs
    pairs = confusion_top_pairs(df)
    print("\n== top confused intent pairs ==")
    print(pairs.to_string(index=False))

    pr.to_csv(f"{args.out}/precision_recall_by_intent.csv", index=False)
    sweep.to_csv(f"{args.out}/threshold_sweep.csv", index=False)
    pairs.to_csv(f"{args.out}/top_confusion_pairs.csv", index=False)
    print(f"\nSaved charts + tables to {args.out}/")


if __name__ == "__main__":
    main()
