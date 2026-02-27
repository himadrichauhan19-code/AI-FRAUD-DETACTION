import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from uuid import uuid4

from flask import Flask, flash, redirect, render_template, request, session, url_for

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

# Demo auth store. In production, use a real database + hashed passwords.
USERS = {}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = DATA_DIR / "transactions.csv"
MODEL_FILE = MODEL_DIR / "fraud_profile.json"
MODEL_META_FILE = MODEL_DIR / "model_info.json"
MODEL_NAME = "Naive Bayes Fraud Model"
MODEL_VERSION = "1.0.0"

CSV_HEADERS = [
    "id",
    "user",
    "amount",
    "merchant",
    "location",
    "app_source",
    "payment_method",
    "timestamp",
    "hour",
    "predicted_risk",
    "predicted_label",
    "actual_label",
]

MODEL_CACHE = {"model": None, "mtime": None}


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_META_FILE.exists():
        with MODEL_META_FILE.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": MODEL_NAME,
                    "version": MODEL_VERSION,
                    "description": "History-based Naive Bayes style fraud scoring model",
                    "features": [
                        "amount",
                        "hour",
                        "day_of_week",
                        "month",
                        "time_bucket",
                        "location",
                        "app_source",
                        "payment_method",
                        "merchant",
                    ],
                },
                f,
            )

    if not DATA_FILE.exists():
        with DATA_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
        return

    # Header/schema migration for older CSV files.
    with DATA_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        current_headers = reader.fieldnames or []
        rows = list(reader)

    if current_headers == CSV_HEADERS:
        return

    migrated_rows = []
    for row in rows:
        migrated = {k: row.get(k, "") for k in CSV_HEADERS}
        migrated_rows.append(migrated)

    with DATA_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(migrated_rows)


def load_transactions() -> List[Dict]:
    ensure_storage()
    with DATA_FILE.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_transaction(row: dict) -> None:
    ensure_storage()
    with DATA_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(row)


def save_transactions(rows: List[Dict]) -> None:
    with DATA_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_timestamp(row: dict) -> datetime:
    timestamp = str(row.get("timestamp", "")).strip()
    if timestamp.endswith(" UTC"):
        timestamp = timestamp[:-4]
    try:
        return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        fallback_hour = _to_int(row.get("hour"), default=datetime.utcnow().hour)
        now = datetime.utcnow()
        return now.replace(hour=max(0, min(23, fallback_hour)), minute=0, second=0, microsecond=0)


def _features_from_row(row: dict) -> dict:
    dt = _parse_timestamp(row)
    hour = _to_int(row.get("hour"), default=dt.hour)

    if hour < 6:
        time_bucket = "night"
    elif hour < 12:
        time_bucket = "morning"
    elif hour < 18:
        time_bucket = "afternoon"
    else:
        time_bucket = "evening"

    return {
        "amount": _to_float(row.get("amount")),
        "hour": hour,
        "day_of_week": dt.weekday(),
        "day_of_month": dt.day,
        "month": dt.month,
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "time_bucket": time_bucket,
        "merchant": row.get("merchant", ""),
        "location": row.get("location", ""),
        "app_source": row.get("app_source", "unknown"),
        "payment_method": row.get("payment_method", ""),
        "user": row.get("user", ""),
    }


def fraud_risk_score(amount: float, hour: int, payment_method: str) -> int:
    """Rule-based fallback when history model is not ready."""
    score = 0

    if amount > 10000:
        score += 60
    elif amount > 5000:
        score += 35
    elif amount > 2000:
        score += 20

    if hour < 6 or hour > 22:
        score += 20

    if payment_method.lower() in {"gift card", "crypto"}:
        score += 25

    return min(score, 100)


def get_model():
    if not MODEL_FILE.exists():
        return None

    mtime = MODEL_FILE.stat().st_mtime
    if MODEL_CACHE["model"] is None or MODEL_CACHE["mtime"] != mtime:
        with MODEL_FILE.open("r", encoding="utf-8") as f:
            MODEL_CACHE["model"] = json.load(f)
        MODEL_CACHE["mtime"] = mtime

    return MODEL_CACHE["model"]


def train_model_from_history() -> bool:
    rows = load_transactions()
    labeled = [r for r in rows if r.get("actual_label") in {"0", "1"}]

    if len(labeled) < 10:
        return False

    labels = [_to_int(r.get("actual_label")) for r in labeled]
    if len(set(labels)) < 2:
        return False

    total_all = len(labeled)
    total_fraud = sum(labels)

    def update_counter(container: dict, key: str, is_fraud: int) -> None:
        if key not in container:
            container[key] = {"fraud": 0, "total": 0}
        container[key]["total"] += 1
        container[key]["fraud"] += is_fraud

    counters = {
        "hour": {},
        "day_of_week": {},
        "month": {},
        "time_bucket": {},
        "location": {},
        "app_source": {},
        "payment_method": {},
        "merchant": {},
    }

    fraud_amounts = []
    legit_amounts = []

    for row in labeled:
        f = _features_from_row(row)
        y = _to_int(row.get("actual_label"))

        update_counter(counters["hour"], str(f["hour"]), y)
        update_counter(counters["day_of_week"], str(f["day_of_week"]), y)
        update_counter(counters["month"], str(f["month"]), y)
        update_counter(counters["time_bucket"], str(f["time_bucket"]), y)
        update_counter(counters["location"], str(f["location"]).lower(), y)
        update_counter(counters["app_source"], str(f["app_source"]).lower(), y)
        update_counter(counters["payment_method"], str(f["payment_method"]).lower(), y)
        update_counter(counters["merchant"], str(f["merchant"]).lower(), y)

        amount = _to_float(row.get("amount"))
        if y == 1:
            fraud_amounts.append(amount)
        else:
            legit_amounts.append(amount)

    def amount_stats(values: List[float]) -> Dict:
        if not values:
            return {"mean": 0.0, "std": 0.0}
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return {"mean": mean, "std": var**0.5}

    model = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "global_rate": total_fraud / max(total_all, 1),
        "total_count": total_all,
        "fraud_count": total_fraud,
        "counters": counters,
        "amount_legit": amount_stats(legit_amounts),
        "amount_fraud": amount_stats(fraud_amounts),
    }

    with MODEL_FILE.open("w", encoding="utf-8") as f:
        json.dump(model, f)

    MODEL_CACHE["model"] = model
    MODEL_CACHE["mtime"] = MODEL_FILE.stat().st_mtime
    return True


def predict_risk(row: dict) -> Tuple[int, str]:
    model = get_model()
    if model is None:
        return (
            fraud_risk_score(
                _to_float(row.get("amount")),
                _to_int(row.get("hour")),
                row.get("payment_method", ""),
            ),
            "heuristic",
        )

    features = _features_from_row(row)
    counters = model.get("counters", {})
    global_rate = float(model.get("global_rate", 0.5))
    total_count = int(model.get("total_count", 1))

    def category_prob(counter_name: str, key: str) -> float:
        alpha = 1.0
        bucket = counters.get(counter_name, {})
        record = bucket.get(key, {"fraud": 0, "total": 0})
        bucket_size = max(len(bucket), 1)

        numer = float(record.get("fraud", 0)) + alpha
        denom = float(record.get("total", 0)) + (alpha * bucket_size)
        if denom <= 0:
            return global_rate
        return numer / denom

    probs = [
        global_rate,
        category_prob("hour", str(features["hour"])),
        category_prob("day_of_week", str(features["day_of_week"])),
        category_prob("month", str(features["month"])),
        category_prob("time_bucket", str(features["time_bucket"])),
        category_prob("location", str(features["location"]).lower()),
        category_prob("app_source", str(features["app_source"]).lower()),
        category_prob("payment_method", str(features["payment_method"]).lower()),
        category_prob("merchant", str(features["merchant"]).lower()),
    ]

    weights = [0.26, 0.09, 0.07, 0.05, 0.09, 0.14, 0.12, 0.09, 0.09]
    score = sum(p * w for p, w in zip(probs, weights))

    amount = _to_float(row.get("amount"))
    legit_stats = model.get("amount_legit", {"mean": 0.0, "std": 0.0})
    legit_mean = float(legit_stats.get("mean", 0.0))
    legit_std = float(legit_stats.get("std", 0.0))

    if legit_std > 0:
        if amount > legit_mean + 3 * legit_std:
            score += 0.25
        elif amount > legit_mean + 2 * legit_std:
            score += 0.15

    # Low history fallback blend with rules.
    if total_count < 30:
        rule_score = fraud_risk_score(
            _to_float(row.get("amount")),
            _to_int(row.get("hour")),
            row.get("payment_method", ""),
        ) / 100.0
        blend = min(1.0, max(0.0, total_count / 30.0))
        score = (blend * score) + ((1.0 - blend) * rule_score)

    risk = int(round(max(0.0, min(1.0, score)) * 100))
    return risk, str(model.get("model_name", "history-model"))


def set_feedback(tx_id: str, user: str, label: int) -> bool:
    rows = load_transactions()
    updated = False

    for row in rows:
        if row.get("id") == tx_id and row.get("user") == user:
            row["actual_label"] = str(label)
            updated = True
            break

    if not updated:
        return False

    save_transactions(rows)
    return True


def normalize_row(row: dict) -> dict:
    risk = _to_int(row.get("predicted_risk"))
    predicted_label = _to_int(row.get("predicted_label"))
    actual = row.get("actual_label")

    if actual not in {"0", "1"}:
        actual_label = None
    else:
        actual_label = _to_int(actual)

    return {
        "id": row.get("id"),
        "timestamp": row.get("timestamp", ""),
        "amount": _to_float(row.get("amount")),
        "merchant": row.get("merchant", ""),
        "location": row.get("location", ""),
        "app_source": row.get("app_source", "unknown"),
        "payment_method": row.get("payment_method", ""),
        "risk": risk,
        "status": "Flagged" if predicted_label == 1 else "Approved",
        "actual_label": actual_label,
    }


@app.route("/")
def index():
    if session.get("user"):
        return redirect(url_for("payment"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user"):
        return redirect(url_for("payment"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not username or not password:
            flash("Username and password are required.", "error")
            return redirect(url_for("register"))
        if len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
            return redirect(url_for("register"))
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for("register"))
        if password != confirm_password:
            flash("Password and confirm password do not match.", "error")
            return redirect(url_for("register"))

        if username in USERS:
            flash("User already exists. Please log in.", "error")
            return redirect(url_for("login"))

        USERS[username] = password
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("payment"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if USERS.get(username) != password:
            flash("Invalid username or password.", "error")
            return redirect(url_for("login"))

        session["user"] = username
        flash("Logged in successfully.", "success")
        return redirect(url_for("payment"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out.", "success")
    return redirect(url_for("index"))


@app.route("/payment", methods=["GET", "POST"])
def payment():
    if "user" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("login"))

    if request.method == "POST":
        amount_raw = request.form.get("amount", "0").strip()
        merchant = request.form.get("merchant", "").strip()
        location = request.form.get("location", "").strip()
        app_source = request.form.get("app_source", "Other").strip()
        payment_method = request.form.get("payment_method", "Card").strip()

        try:
            amount = float(amount_raw)
        except ValueError:
            flash("Amount must be a number.", "error")
            return redirect(url_for("payment"))

        now = datetime.utcnow()
        row_for_prediction = {
            "amount": amount,
            "hour": now.hour,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "merchant": merchant,
            "location": location,
            "app_source": app_source,
            "payment_method": payment_method,
            "user": session["user"],
        }

        risk, source = predict_risk(row_for_prediction)
        predicted_label = 1 if risk >= 50 else 0
        status = "Flagged" if predicted_label else "Approved"

        tx = {
            "id": uuid4().hex,
            "user": session["user"],
            "amount": f"{amount:.2f}",
            "merchant": merchant,
            "location": location,
            "app_source": app_source,
            "payment_method": payment_method,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "hour": str(now.hour),
            "predicted_risk": str(risk),
            "predicted_label": str(predicted_label),
            "actual_label": "",
        }
        append_transaction(tx)

        flash(f"Transaction processed via {source}. Risk: {risk}% ({status}).", "success")
        return redirect(url_for("dashboard"))

    return render_template("payment.html")


@app.route("/feedback/<tx_id>", methods=["POST"])
def feedback(tx_id: str):
    if "user" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("login"))

    label_text = request.form.get("label", "").strip().lower()
    if label_text not in {"fraud", "legit"}:
        flash("Invalid feedback label.", "error")
        return redirect(url_for("dashboard"))

    label = 1 if label_text == "fraud" else 0
    updated = set_feedback(tx_id=tx_id, user=session["user"], label=label)

    if not updated:
        flash("Transaction not found for feedback.", "error")
        return redirect(url_for("dashboard"))

    trained = train_model_from_history()
    if trained:
        flash("Feedback saved and model retrained from history.", "success")
    else:
        flash("Feedback saved. More labeled history needed for model training.", "success")

    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("login"))

    rows = load_transactions()
    user_transactions = [normalize_row(r) for r in rows if r.get("user") == session["user"]]
    user_transactions.sort(key=lambda x: x["timestamp"], reverse=True)

    return render_template("dashboard.html", transactions=user_transactions)


ensure_storage()
train_model_from_history()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)

