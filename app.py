"""
app.py — Pennywise Flask API (Phase 7: Supabase + Render Production)

Auth endpoints  (Supabase Auth)
--------------------------------
    POST  /auth/signup     { email, password }       → supabase.auth.sign_up  (sends OTP email)
    POST  /auth/verify     { email, token }           → supabase.auth.verify_otp (type='signup')
    POST  /auth/login      { email, password }        → supabase.auth.sign_in_with_password
    POST  /auth/logout                                → clear session

Session check
-------------
    GET   /me              → { authenticated, email }

Protected endpoints  (require session['user_id'] — returns 401 otherwise)
--------------------------------------------------------------------------
    GET    /expenses                 ?q=<search>
    POST   /expenses                 { title, amount }  → ML predict → insert
    PUT    /expenses/<id>            { title, amount, category }
    DELETE /expenses/<id>
    GET    /budget
    POST   /budget                   { limit }
    GET    /summary
    GET    /wrapped
    GET    /export
"""

import csv
import io
import os
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv
from flask import Flask, jsonify, make_response, request, session
from flask_cors import CORS
from supabase import create_client, Client

import ml_service

# ---------------------------------------------------------------------------
# Environment & App setup
# ---------------------------------------------------------------------------

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
FLASK_SECRET_KEY = os.environ["FLASK_SECRET_KEY"]

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# Allow the deployed frontend origin; adjust FRONTEND_ORIGIN in .env as needed
FRONTEND_ORIGIN = os.environ.get(
    "FRONTEND_ORIGIN",
    "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:5000",
)
origins = [o.strip() for o in FRONTEND_ORIGIN.split(",")]

# This line tells the server to trust your specific Render URL
CORS(app, supports_credentials=True, origins=origins)

# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def current_month() -> str:
    return datetime.now().strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Auth guard decorator
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Unauthorized. Please log in."}), 401
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Auth routes  (Supabase Auth)
# ---------------------------------------------------------------------------

@app.route("/auth/signup", methods=["POST"])
def signup():
    data     = request.get_json(silent=True) or {}
    email    = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or "@" not in email:
        return jsonify({"error": "A valid email is required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    try:
        # Supabase sends a 6-digit OTP email automatically when email confirmations
        # are set to "OTP" in the Supabase Auth settings.
        res = supabase.auth.sign_up({"email": email, "password": password})
        if res.user is None:
            return jsonify({"error": "Signup failed. Please try again."}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "message": "Account created! Check your email for the 6-digit verification code.",
        "email": email,
    }), 201


@app.route("/auth/verify", methods=["POST"])
def verify_otp():
    """Verify the email OTP sent by Supabase after sign-up."""
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    token = (data.get("token") or "").strip()

    if not email or not token:
        return jsonify({"error": "email and token are required."}), 400

    try:
        res = supabase.auth.verify_otp({"email": email, "token": token, "type": "signup"})
        if res.user is None:
            return jsonify({"error": "Invalid or expired OTP."}), 401
    except Exception as exc:
        return jsonify({"error": str(exc)}), 401

    session["user_id"] = res.user.id
    session["email"]   = res.user.email

    return jsonify({"message": "Email verified! Welcome to Pennywise.", "email": res.user.email}), 200


@app.route("/auth/login", methods=["POST"])
def login():
    data     = request.get_json(silent=True) or {}
    email    = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "email and password are required."}), 400

    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if res.user is None:
            return jsonify({"error": "Invalid email or password."}), 401
    except Exception as exc:
        return jsonify({"error": "Invalid email or password."}), 401

    session["user_id"] = res.user.id
    session["email"]   = res.user.email

    return jsonify({"message": "Logged in successfully.", "email": res.user.email}), 200


@app.route("/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out."}), 200


@app.route("/me", methods=["GET"])
def me():
    """Frontend polls this on load to restore session state."""
    if "user_id" not in session:
        return jsonify({"authenticated": False}), 200
    return jsonify({"authenticated": True, "email": session.get("email")}), 200


# ---------------------------------------------------------------------------
# Expenses  (GET list + POST create)
# ---------------------------------------------------------------------------

@app.route("/expenses", methods=["GET"])
@login_required
def get_expenses():
    uid    = session["user_id"]
    search = (request.args.get("q") or "").strip() or None

    query = (
        supabase.table("expenses")
        .select("*")
        .eq("user_id", uid)
        .order("date", desc=True)
    )
    if search:
        # Supabase ilike for case-insensitive search across title and category
        query = query.or_(f"title.ilike.%{search}%,category.ilike.%{search}%")

    res = query.execute()
    return jsonify(res.data), 200


@app.route("/expenses", methods=["POST"])
@login_required
def add_expense():
    uid    = session["user_id"]
    data   = request.get_json(silent=True) or {}
    title  = (data.get("title") or "").strip()
    amount = data.get("amount")

    if not title:
        return jsonify({"error": "title is required"}), 400
    try:
        amount = float(amount)
        if amount <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "amount must be a positive number"}), 400

    # ML prediction happens before every insert
    category, confidence = ml_service.predict_category(title)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "user_id":  uid,
        "title":    title,
        "amount":   amount,
        "category": category,
        "date":     now,
    }

    res = supabase.table("expenses").insert(row).execute()
    inserted = res.data[0] if res.data else {}

    return jsonify({
        **inserted,
        "confidence": confidence,
        "message": "Expense added successfully",
    }), 201


# ---------------------------------------------------------------------------
# Expense  (GET / PUT / DELETE single row)
# ---------------------------------------------------------------------------

@app.route("/expenses/<int:expense_id>", methods=["GET"])
@login_required
def get_expense(expense_id):
    uid = session["user_id"]
    res = (
        supabase.table("expenses")
        .select("*")
        .eq("id", expense_id)
        .eq("user_id", uid)
        .execute()
    )
    if not res.data:
        return jsonify({"error": "Expense not found"}), 404
    return jsonify(res.data[0]), 200


@app.route("/expenses/<int:expense_id>", methods=["PUT"])
@login_required
def update_expense(expense_id):
    uid  = session["user_id"]
    data = request.get_json(silent=True) or {}

    title    = (data.get("title") or "").strip()
    amount   = data.get("amount")
    category = (data.get("category") or "").strip()

    if not title:
        return jsonify({"error": "title is required"}), 400
    try:
        amount = float(amount)
        if amount <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "amount must be a positive number"}), 400
    if not category:
        return jsonify({"error": "category is required"}), 400

    res = (
        supabase.table("expenses")
        .update({"title": title, "amount": amount, "category": category})
        .eq("id", expense_id)
        .eq("user_id", uid)
        .execute()
    )
    if not res.data:
        return jsonify({"error": "Expense not found or not yours"}), 404
    return jsonify({"message": "Expense updated successfully", **res.data[0]}), 200


@app.route("/expenses/<int:expense_id>", methods=["DELETE"])
@login_required
def delete_expense(expense_id):
    uid = session["user_id"]
    res = (
        supabase.table("expenses")
        .delete()
        .eq("id", expense_id)
        .eq("user_id", uid)
        .execute()
    )
    if not res.data:
        return jsonify({"error": "Expense not found or not yours"}), 404
    return jsonify({"message": "Expense deleted successfully", "id": expense_id}), 200


# ---------------------------------------------------------------------------
# Budget  (GET + POST/upsert)
# ---------------------------------------------------------------------------

@app.route("/budget", methods=["GET"])
@login_required
def get_budget():
    uid   = session["user_id"]
    month = current_month()
    return jsonify(_budget_payload(uid, month)), 200


@app.route("/budget", methods=["POST"])
@login_required
def set_budget():
    uid   = session["user_id"]
    data  = request.get_json(silent=True) or {}
    limit = data.get("limit")

    try:
        limit = float(limit)
        if limit <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "limit must be a positive number"}), 400

    month = current_month()

    # Upsert: insert or overwrite the (user_id, month) row
    supabase.table("budgets").upsert({
        "user_id":      uid,
        "month":        month,
        "limit_amount": limit,
    }).execute()

    return jsonify({"message": "Budget set successfully", **_budget_payload(uid, month)}), 200


def _budget_payload(uid: str, month: str) -> dict:
    """Return budget + spent totals for a user/month."""
    budget_res = (
        supabase.table("budgets")
        .select("limit_amount")
        .eq("user_id", uid)
        .eq("month", month)
        .execute()
    )
    limit_amount = budget_res.data[0]["limit_amount"] if budget_res.data else None

    # Sum expenses for the month
    exp_res = (
        supabase.table("expenses")
        .select("amount")
        .eq("user_id", uid)
        .like("date", f"{month}%")
        .execute()
    )
    spent = sum(row["amount"] for row in (exp_res.data or []))

    return {
        "month":        month,
        "limit_amount": limit_amount,
        "spent":        spent,
        "over_budget":  (limit_amount is not None) and (spent > limit_amount),
    }


# ---------------------------------------------------------------------------
# Summary  (Chart.js data)
# ---------------------------------------------------------------------------

@app.route("/summary", methods=["GET"])
@login_required
def get_summary():
    uid   = session["user_id"]
    month = current_month()

    res = (
        supabase.table("expenses")
        .select("amount,category")
        .eq("user_id", uid)
        .like("date", f"{month}%")
        .execute()
    )
    rows = res.data or []

    total = sum(r["amount"] for r in rows)

    by_category: dict[str, float] = {}
    for r in rows:
        by_category[r["category"]] = by_category.get(r["category"], 0) + r["amount"]
    # Sort descending by value
    by_category = dict(sorted(by_category.items(), key=lambda x: x[1], reverse=True))

    return jsonify({"month": month, "total": total, "by_category": by_category}), 200


# ---------------------------------------------------------------------------
# Wrapped  (Spotify-style insights)
# ---------------------------------------------------------------------------

@app.route("/wrapped", methods=["GET"])
@login_required
def get_wrapped():
    uid   = session["user_id"]
    month = current_month()

    # All expenses this month
    cur_res = (
        supabase.table("expenses")
        .select("*")
        .eq("user_id", uid)
        .like("date", f"{month}%")
        .execute()
    )
    cur_rows = cur_res.data or []

    # Top category
    cat_totals: dict[str, float] = {}
    for r in cur_rows:
        cat_totals[r["category"]] = cat_totals.get(r["category"], 0) + r["amount"]
    top_category = None
    if cat_totals:
        top_cat_name = max(cat_totals, key=lambda k: cat_totals[k])
        top_category = {"category": top_cat_name, "cat_total": cat_totals[top_cat_name]}

    # Biggest splurge
    biggest_splurge = max(cur_rows, key=lambda r: r["amount"]) if cur_rows else None

    # Month-over-month trend
    year, mon = int(month[:4]), int(month[5:])
    prev_month_str = f"{year - 1}-12" if mon == 1 else f"{year}-{mon - 1:02d}"

    current_total = sum(r["amount"] for r in cur_rows)

    prev_res = (
        supabase.table("expenses")
        .select("amount")
        .eq("user_id", uid)
        .like("date", f"{prev_month_str}%")
        .execute()
    )
    prev_total = sum(r["amount"] for r in (prev_res.data or []))

    if prev_total == 0:
        pct_change = None
        direction  = "new"
    else:
        raw_pct    = ((current_total - prev_total) / prev_total) * 100
        pct_change = round(raw_pct, 1)
        direction  = "up" if pct_change > 0 else ("down" if pct_change < 0 else "same")

    return jsonify({
        "month":           month,
        "top_category":    top_category,
        "biggest_splurge": biggest_splurge,
        "trend": {
            "current_total": current_total,
            "prev_total":    prev_total,
            "prev_month":    prev_month_str,
            "pct_change":    pct_change,
            "direction":     direction,
        },
    }), 200


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------

@app.route("/export", methods=["GET"])
@login_required
def export_csv():
    uid = session["user_id"]
    res = (
        supabase.table("expenses")
        .select("*")
        .eq("user_id", uid)
        .order("date", desc=True)
        .execute()
    )
    expenses = res.data or []

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Title", "Amount (₹)", "Category", "Date"])
    for e in expenses:
        writer.writerow([e.get("id"), e.get("title"), e.get("amount"), e.get("category"), e.get("date")])
    csv_data = output.getvalue()
    output.close()

    response = make_response(csv_data)
    response.headers["Content-Type"]        = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = "attachment; filename=pennywise_report.csv"
    return response

@app.route("/")
def home():
    return app.send_static_file('index.html')

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
