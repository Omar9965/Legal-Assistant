"""
Memory — Conversation persistence using LangGraph SQLite checkpointer.

Provides thread-based conversation tracking so the agent remembers past interactions.
"""

from langgraph.checkpoint.sqlite import SqliteSaver
from utils.config import CHECKPOINT_DB_PATH
import sqlite3


def get_checkpointer() -> SqliteSaver:
    """
    Return a SQLite-backed LangGraph checkpointer.

    Uses a persistent SQLite database so conversation history
    survives between sessions.

    Returns:
        SqliteSaver instance configured with the checkpoint DB path.
    """
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return checkpointer


def close_checkpointer(checkpointer: SqliteSaver) -> None:
    """Close the underlying SQLite connection of a checkpointer."""
    checkpointer.conn.close()