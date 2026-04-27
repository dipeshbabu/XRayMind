"""SQLite persistence layer for XRayMind case workflow."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB_PATH = "outputs/xraymind_cases.sqlite3"
SCHEMA_VERSION = 3


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()


def dumps_json(value: Any) -> str:
    """Serialize JSON consistently for SQLite storage."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def loads_json(value: str | None, default: Any = None) -> Any:
    """Deserialize JSON from SQLite, returning default for blank values."""

    if value in (None, ""):
        return default
    return json.loads(value)


def dict_factory(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict[str, Any]:
    """Return SQLite rows as dictionaries."""

    return {column[0]: row[idx] for idx, column in enumerate(cursor.description)}


class SQLiteStore:
    """Small SQLite wrapper for cases, predictions, reviews, jobs, and audit events."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = dict_factory
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        """Create database tables and run additive schema migrations."""

        with sqlite3.connect(self.path) as conn:
            conn.row_factory = dict_factory
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    image_id TEXT,
                    source_filename TEXT,
                    model_name TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority TEXT NOT NULL DEFAULT 'routine',
                    patient_context TEXT,
                    tags TEXT,
                    assigned_to TEXT,
                    due_at TEXT,
                    needs_second_reader INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    top_k INTEGER NOT NULL,
                    prediction_json TEXT NOT NULL,
                    max_probability REAL,
                    low_confidence INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER NOT NULL,
                    reviewer TEXT,
                    decision TEXT NOT NULL,
                    notes TEXT,
                    final_labels TEXT,
                    review_round INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS hosted_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    payload TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER,
                    event_type TEXT NOT NULL,
                    payload TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE SET NULL
                );
                """
            )
            self._migrate(conn)
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
                CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases(priority);
                CREATE INDEX IF NOT EXISTS idx_cases_assigned_to ON cases(assigned_to);
                CREATE INDEX IF NOT EXISTS idx_cases_second_reader ON cases(needs_second_reader);
                CREATE INDEX IF NOT EXISTS idx_predictions_case_id ON predictions(case_id);
                CREATE INDEX IF NOT EXISTS idx_reviews_case_id ON reviews(case_id);
                CREATE INDEX IF NOT EXISTS idx_audit_case_id ON audit_events(case_id);
                CREATE INDEX IF NOT EXISTS idx_jobs_tenant_status ON hosted_jobs(tenant_id, status);
                CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON hosted_jobs(status, created_at);
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Apply idempotent additive migrations for existing local SQLite DBs."""

        table_columns = {row["name"] for row in conn.execute("PRAGMA table_info(cases)").fetchall()}
        case_migrations = {
            "assigned_to": "ALTER TABLE cases ADD COLUMN assigned_to TEXT",
            "due_at": "ALTER TABLE cases ADD COLUMN due_at TEXT",
            "needs_second_reader": "ALTER TABLE cases ADD COLUMN needs_second_reader INTEGER NOT NULL DEFAULT 0",
        }
        for column, statement in case_migrations.items():
            if column not in table_columns:
                conn.execute(statement)

        review_columns = {row["name"] for row in conn.execute("PRAGMA table_info(reviews)").fetchall()}
        if "review_round" not in review_columns:
            conn.execute("ALTER TABLE reviews ADD COLUMN review_round INTEGER NOT NULL DEFAULT 1")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hosted_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                job_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                payload TEXT NOT NULL,
                result TEXT,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            )
            """
        )

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
        """Execute a write query and return the last inserted row id."""

        with self.connect() as conn:
            cursor = conn.execute(query, params)
            return int(cursor.lastrowid)

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        """Fetch one row as a dictionary."""

        with self.connect() as conn:
            return conn.execute(query, params).fetchone()

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Fetch all rows as dictionaries."""

        with self.connect() as conn:
            return list(conn.execute(query, params).fetchall())
