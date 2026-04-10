"""Manifest helpers for DSGR data pipeline."""


def init_split_stats(split: str, source: str):
    return {
        "split": split,
        "source": source,
        "total_input": 0,
        "total_output": 0,
        "success": 0,
        "fallback": 0,
        "skipped": 0,
        "error_types": {},
        "notes": [],
    }


def add_error(stats: dict, err: Exception):
    err_name = type(err).__name__
    stats["error_types"][err_name] = stats["error_types"].get(err_name, 0) + 1


def add_note(stats: dict, note: str, limit: int = 20):
    if len(stats["notes"]) < limit:
        stats["notes"].append(note)

