import json
import os

HISTORY_FILE = os.path.join(
    os.path.dirname(__file__), "chat_history.json"
)


def load_history() -> list[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_history(messages: list[dict]) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Failed to save chat history: {e}")


def clear_history() -> None:
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
        except OSError as e:
            print(f"Failed to clear chat history: {e}")