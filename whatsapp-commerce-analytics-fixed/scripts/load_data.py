"""
Loads the generated CSVs into Postgres.

Usage:
    python3 load_data.py --dsn "dbname=whatsapp_commerce user=postgres password=postgres host=localhost"
"""
import argparse
import psycopg2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="dbname=whatsapp_commerce user=postgres password=postgres host=localhost")
    ap.add_argument("--data", default="../data")
    ap.add_argument("--schema", default="../sql/schema.sql")
    args = ap.parse_args()

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = True
    cur = conn.cursor()

    with open(args.schema) as f:
        cur.execute(f.read())
    print("schema created")

    for table, path, cols in [
        ("merchants", f"{args.data}/merchants.csv",
         "merchant_id, category, city, signup_date"),
        ("conversations", f"{args.data}/conversations.csv",
         "conversation_id, merchant_id, customer_id, ts, message_text, "
         "true_intent, predicted_intent, confidence, escalated_to_human"),
        ("orders", f"{args.data}/orders.csv",
         "order_id, merchant_id, conversation_id, customer_id, status, "
         "value_inr, created_at, fulfilled_at"),
    ]:
        with open(path) as f:
            next(f)  # skip header
            cur.copy_expert(f"COPY {table} ({cols}) FROM STDIN WITH CSV NULL ''", f)
        cur.execute(f"SELECT count(*) FROM {table}")
        print(f"{table}: {cur.fetchone()[0]:,} rows loaded")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
