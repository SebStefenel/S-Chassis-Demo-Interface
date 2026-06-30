"""SDV local user database — SQLite store for driver profiles and face embeddings."""

import json
import pickle
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _conn(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(db_path: str) -> None:
    with _conn(db_path) as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT    UNIQUE NOT NULL,
                settings   TEXT    NOT NULL DEFAULT '{}',
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL,
                embedding BLOB    NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)


def add_user(db_path: str, username: str, settings: Dict[str, Any]) -> int:
    with _conn(db_path) as con:
        cur = con.execute(
            "INSERT INTO users (username, settings) VALUES (?, ?)",
            (username, json.dumps(settings)),
        )
        return cur.lastrowid


def get_user_by_name(db_path: str, username: str) -> Optional[Dict[str, Any]]:
    with _conn(db_path) as con:
        row = con.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "settings": json.loads(row["settings"]),
        "created_at": row["created_at"],
    }


def list_users(db_path: str) -> List[Dict[str, Any]]:
    with _conn(db_path) as con:
        rows = con.execute(
            "SELECT id, username, created_at FROM users ORDER BY username"
        ).fetchall()
    return [{"id": r["id"], "username": r["username"], "created_at": r["created_at"]} for r in rows]


def delete_user(db_path: str, user_id: int) -> None:
    with _conn(db_path) as con:
        con.execute("DELETE FROM users WHERE id = ?", (user_id,))


def get_user_settings(db_path: str, user_id: int) -> Dict[str, Any]:
    with _conn(db_path) as con:
        row = con.execute("SELECT settings FROM users WHERE id = ?", (user_id,)).fetchone()
    return json.loads(row["settings"]) if row else {}


def update_user_settings(db_path: str, user_id: int, settings: Dict[str, Any]) -> None:
    with _conn(db_path) as con:
        con.execute(
            "UPDATE users SET settings = ? WHERE id = ?",
            (json.dumps(settings), user_id),
        )


def add_embedding(db_path: str, user_id: int, embedding: np.ndarray) -> None:
    blob = pickle.dumps(embedding.astype(np.float32))
    with _conn(db_path) as con:
        con.execute(
            "INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)",
            (user_id, blob),
        )


def get_all_embeddings(db_path: str) -> List[Tuple[int, str, np.ndarray]]:
    """Return [(user_id, username, embedding_array), ...] for every stored embedding."""
    with _conn(db_path) as con:
        rows = con.execute("""
            SELECT e.user_id, u.username, e.embedding
            FROM   embeddings e
            JOIN   users      u ON e.user_id = u.id
        """).fetchall()
    return [(r["user_id"], r["username"], pickle.loads(r["embedding"])) for r in rows]


def embedding_count(db_path: str, user_id: int) -> int:
    with _conn(db_path) as con:
        row = con.execute(
            "SELECT COUNT(*) AS n FROM embeddings WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row["n"] if row else 0
