"""
SDV local user database — SQLite-backed store for driver profiles and face embeddings.

Schema
------
users      : id, username (UNIQUE), settings (JSON blob), created_at
embeddings : id, user_id (FK→users.id), embedding (pickle'd numpy array)
"""

import json
import pickle
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── connection helper ─────────────────────────────────────────────────────────

def _conn(db_path: str) -> sqlite3.Connection:
    # isolation_level=None → autocommit; bypasses Python's implicit BEGIN/COMMIT
    # which require fcntl write-locks that macOS blocks for Homebrew Python.
    con = sqlite3.connect(db_path, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode = MEMORY")
    con.execute("PRAGMA foreign_keys = ON")
    return con


# ── schema init ───────────────────────────────────────────────────────────────

def init_db(db_path: str) -> None:
    con = _conn(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    UNIQUE NOT NULL,
            settings   TEXT    NOT NULL DEFAULT '{}',
            created_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            embedding BLOB    NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    con.close()


# ── users ─────────────────────────────────────────────────────────────────────

def add_user(db_path: str, username: str, settings: Dict[str, Any]) -> int:
    """Insert a new user row; returns the new user id."""
    con = _conn(db_path)
    cur = con.execute(
        "INSERT INTO users (username, settings) VALUES (?, ?)",
        (username, json.dumps(settings)),
    )
    uid = cur.lastrowid
    con.close()
    return uid


def get_user_by_name(db_path: str, username: str) -> Optional[Dict[str, Any]]:
    con = _conn(db_path)
    row = con.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    con.close()
    if row is None:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "settings": json.loads(row["settings"]),
        "created_at": row["created_at"],
    }


def list_users(db_path: str) -> List[Dict[str, Any]]:
    con = _conn(db_path)
    rows = con.execute(
        "SELECT id, username, created_at FROM users ORDER BY username"
    ).fetchall()
    con.close()
    return [{"id": r["id"], "username": r["username"], "created_at": r["created_at"]} for r in rows]


def delete_user(db_path: str, user_id: int) -> None:
    """Delete user and cascade-delete their embeddings."""
    con = _conn(db_path)
    con.execute("DELETE FROM users WHERE id = ?", (user_id,))
    con.close()


def get_user_settings(db_path: str, user_id: int) -> Dict[str, Any]:
    con = _conn(db_path)
    row = con.execute(
        "SELECT settings FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    con.close()
    return json.loads(row["settings"]) if row else {}


def update_user_settings(db_path: str, user_id: int, settings: Dict[str, Any]) -> None:
    con = _conn(db_path)
    con.execute(
        "UPDATE users SET settings = ? WHERE id = ?",
        (json.dumps(settings), user_id),
    )
    con.close()


# ── embeddings ────────────────────────────────────────────────────────────────

def add_embedding(db_path: str, user_id: int, embedding: np.ndarray) -> None:
    blob = pickle.dumps(embedding.astype(np.float32))
    con = _conn(db_path)
    con.execute(
        "INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)",
        (user_id, blob),
    )
    con.close()


def get_all_embeddings(db_path: str) -> List[Tuple[int, str, np.ndarray]]:
    """Return [(user_id, username, embedding_array), ...] for every stored embedding."""
    con = _conn(db_path)
    rows = con.execute("""
        SELECT e.user_id, u.username, e.embedding
        FROM   embeddings e
        JOIN   users      u ON e.user_id = u.id
    """).fetchall()
    con.close()
    return [(r["user_id"], r["username"], pickle.loads(r["embedding"])) for r in rows]


def embedding_count(db_path: str, user_id: int) -> int:
    con = _conn(db_path)
    row = con.execute(
        "SELECT COUNT(*) AS n FROM embeddings WHERE user_id = ?", (user_id,)
    ).fetchone()
    con.close()
    return row["n"] if row else 0
