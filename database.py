"""
database.py — Pennywise (Phase 6: Multi-User)

Tables
------
users    : id, email, password_hash, otp, otp_expiry
expenses : id, user_id, title, amount, category, date
budgets  : user_id + month (composite PK), limit_amount

Every data-access function now requires a user_id so rows are always
scoped to the authenticated user. Auth helper functions (create_user,
get_user_by_email, save_otp, get_user_by_id) are kept separate.
"""

import sqlite3
from datetime import datetime, date

DB_PATH = "pennywise.db"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            email        TEXT    NOT NULL UNIQUE,
            password     TEXT    NOT NULL,
            otp          TEXT,
            otp_expiry   TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title    TEXT    NOT NULL,
            amount   REAL    NOT NULL,
            category TEXT    NOT NULL,
            date     TEXT    NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            month        TEXT    NOT NULL,
            limit_amount REAL    NOT NULL,
            PRIMARY KEY (user_id, month)
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# User / Auth functions
# ---------------------------------------------------------------------------

def create_user(email: str, password_hash: str) -> int:
    """Insert a new user. Returns the new user id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (email, password) VALUES (?, ?)",
        (email, password_hash),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_user_by_email(email: str) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def save_otp(user_id: int, otp: str, expiry: str) -> None:
    """Persist a freshly-generated OTP and its expiry timestamp (ISO string)."""
    conn = get_connection()
    conn.execute(
        "UPDATE users SET otp = ?, otp_expiry = ? WHERE id = ?",
        (otp, expiry, user_id),
    )
    conn.commit()
    conn.close()


def clear_otp(user_id: int) -> None:
    """Wipe OTP columns after successful verification."""
    conn = get_connection()
    conn.execute(
        "UPDATE users SET otp = NULL, otp_expiry = NULL WHERE id = ?",
        (user_id,),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Expense functions  (all scoped by user_id)
# ---------------------------------------------------------------------------

def insert_expense(user_id: int, title: str, amount: float, category: str) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO expenses (user_id, title, amount, category, date) VALUES (?, ?, ?, ?, ?)",
        (user_id, title, amount, category, date_str),
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def fetch_all_expenses(user_id: int, search: str = None) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    if search:
        pattern = f"%{search}%"
        cursor.execute(
            """SELECT * FROM expenses
               WHERE user_id = ? AND (title LIKE ? OR category LIKE ?)
               ORDER BY date DESC""",
            (user_id, pattern, pattern),
        )
    else:
        cursor.execute(
            "SELECT * FROM expenses WHERE user_id = ? ORDER BY date DESC",
            (user_id,),
        )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def fetch_expense_by_id(user_id: int, expense_id: int) -> dict | None:
    """Returns the expense only if it belongs to user_id (prevents IDOR)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM expenses WHERE id = ? AND user_id = ?",
        (expense_id, user_id),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_expense(user_id: int, expense_id: int, title: str, amount: float, category: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE expenses SET title = ?, amount = ?, category = ? WHERE id = ? AND user_id = ?",
        (title, amount, category, expense_id, user_id),
    )
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


def delete_expense(user_id: int, expense_id: int) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM expenses WHERE id = ? AND user_id = ?",
        (expense_id, user_id),
    )
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0


# ---------------------------------------------------------------------------
# Budget functions  (all scoped by user_id)
# ---------------------------------------------------------------------------

def set_budget(user_id: int, month: str, limit_amount: float) -> None:
    conn = get_connection()
    conn.execute(
        "REPLACE INTO budgets (user_id, month, limit_amount) VALUES (?, ?, ?)",
        (user_id, month, limit_amount),
    )
    conn.commit()
    conn.close()


def get_budget_and_spent(user_id: int, month: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT limit_amount FROM budgets WHERE user_id = ? AND month = ?",
        (user_id, month),
    )
    row = cursor.fetchone()
    limit_amount = row["limit_amount"] if row else None

    cursor.execute(
        "SELECT COALESCE(SUM(amount), 0) AS spent FROM expenses WHERE user_id = ? AND date LIKE ?",
        (user_id, f"{month}%"),
    )
    spent = cursor.fetchone()["spent"]

    conn.close()
    return {
        "month":        month,
        "limit_amount": limit_amount,
        "spent":        spent,
        "over_budget":  (limit_amount is not None) and (spent > limit_amount),
    }


# ---------------------------------------------------------------------------
# Summary / Analytics  (all scoped by user_id)
# ---------------------------------------------------------------------------

def get_monthly_summary(user_id: int, month: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COALESCE(SUM(amount), 0) AS total FROM expenses WHERE user_id = ? AND date LIKE ?",
        (user_id, f"{month}%"),
    )
    total = cursor.fetchone()["total"]

    cursor.execute(
        """SELECT category, SUM(amount) AS cat_total
           FROM expenses
           WHERE user_id = ? AND date LIKE ?
           GROUP BY category
           ORDER BY cat_total DESC""",
        (user_id, f"{month}%"),
    )
    by_category = {row["category"]: row["cat_total"] for row in cursor.fetchall()}

    conn.close()
    return {"month": month, "total": total, "by_category": by_category}


def get_monthly_wrapped(user_id: int, month_str: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    prefix = f"{month_str}%"

    cursor.execute(
        """SELECT category, SUM(amount) AS cat_total
           FROM expenses WHERE user_id = ? AND date LIKE ?
           GROUP BY category ORDER BY cat_total DESC LIMIT 1""",
        (user_id, prefix),
    )
    row = cursor.fetchone()
    top_category = dict(row) if row else None

    cursor.execute(
        "SELECT * FROM expenses WHERE user_id = ? AND date LIKE ? ORDER BY amount DESC LIMIT 1",
        (user_id, prefix),
    )
    row = cursor.fetchone()
    biggest_splurge = dict(row) if row else None

    year, month = int(month_str[:4]), int(month_str[5:])
    prev_month_str = f"{year - 1}-12" if month == 1 else f"{year}-{month - 1:02d}"

    cursor.execute(
        "SELECT COALESCE(SUM(amount), 0) AS total FROM expenses WHERE user_id = ? AND date LIKE ?",
        (user_id, prefix),
    )
    current_total = cursor.fetchone()["total"]

    cursor.execute(
        "SELECT COALESCE(SUM(amount), 0) AS total FROM expenses WHERE user_id = ? AND date LIKE ?",
        (user_id, f"{prev_month_str}%"),
    )
    prev_total = cursor.fetchone()["total"]

    if prev_total == 0:
        pct_change = None
        direction  = "new"
    else:
        raw_pct    = ((current_total - prev_total) / prev_total) * 100
        pct_change = round(raw_pct, 1)
        direction  = "up" if pct_change > 0 else ("down" if pct_change < 0 else "same")

    conn.close()
    return {
        "month":           month_str,
        "top_category":    top_category,
        "biggest_splurge": biggest_splurge,
        "trend": {
            "current_total": current_total,
            "prev_total":    prev_total,
            "prev_month":    prev_month_str,
            "pct_change":    pct_change,
            "direction":     direction,
        },
    }
