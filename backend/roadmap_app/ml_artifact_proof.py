from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

PROOF_FILE_METADATA = "metadata.json"
PROOF_FILE_EVAL = "eval_report.json"
PROOF_FILE_SHADOW = "shadow_report.json"
PROOF_FILE_UPLIFT_7D = "uplift_report_7d.json"
PROOF_FILE_UPLIFT_30D = "uplift_report_30d.json"

PROOF_STATUS_OK = "ok"
PROOF_STATUS_MISSING = "missing"
PROOF_STATUS_INVALID = "invalid"
PROOF_STATUS_STALE = "stale"


def artifact_dir_for_model_path(model_path: str | Path | None) -> Path:
    path = Path(str(model_path or "").strip()).expanduser()
    return path.parent if path.suffix else path


def artifact_file_path(model_path: str | Path | None, filename: str) -> Path:
    return (artifact_dir_for_model_path(model_path) / str(filename or "").strip()).expanduser()


def _path_iso_utc(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=dt_timezone.utc).isoformat()
    except Exception:
        return None


def _normalized_path_str(value: str | Path | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve())
    except Exception:
        return str(Path(raw).expanduser())


@lru_cache(maxsize=256)
def _load_json_cached(path_str: str, mtime_ns: int) -> dict[str, Any] | None:
    del mtime_ns
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_json_file(path: str | Path | None) -> dict[str, Any] | None:
    raw = str(path or "").strip()
    if not raw:
        return None
    file_path = Path(raw).expanduser()
    if not file_path.exists() or not file_path.is_file():
        return None
    try:
        return _load_json_cached(str(file_path.resolve()), int(file_path.stat().st_mtime_ns))
    except Exception:
        return None


def proof_file_state(
    *,
    model_path: str | Path | None,
    filename: str,
    expected_model_version: str | None = None,
) -> dict[str, Any]:
    model = Path(str(model_path or "").strip()).expanduser()
    path = artifact_file_path(model_path, filename)
    out: dict[str, Any] = {
        "name": str(filename or "").strip(),
        "path": str(path),
        "exists": bool(path.exists() and path.is_file()),
        "status": PROOF_STATUS_MISSING,
        "reason": f"missing_{path.stem}",
        "mtime_utc": _path_iso_utc(path) if path.exists() else None,
        "payload_model_version": "",
        "payload_model_path": "",
    }
    if not out["exists"]:
        return out

    payload = load_json_file(path)
    if payload is None:
        out["status"] = PROOF_STATUS_INVALID
        out["reason"] = f"invalid_{path.stem}"
        return out

    payload_model_version = str(payload.get("model_version") or "").strip()
    payload_model_path = str(payload.get("model_path") or "").strip()
    out["payload_model_version"] = payload_model_version
    out["payload_model_path"] = payload_model_path

    if model.exists() and model.is_file():
        try:
            if int(path.stat().st_mtime_ns) < int(model.stat().st_mtime_ns):
                out["status"] = PROOF_STATUS_STALE
                out["reason"] = f"stale_{path.stem}_older_than_model"
                return out
        except Exception:
            pass

    expected_version = str(expected_model_version or "").strip()
    if expected_version and payload_model_version and payload_model_version != expected_version:
        out["status"] = PROOF_STATUS_STALE
        out["reason"] = f"stale_{path.stem}_model_version_mismatch"
        return out

    normalized_expected_path = _normalized_path_str(model)
    normalized_payload_path = _normalized_path_str(payload_model_path)
    if normalized_expected_path and normalized_payload_path and normalized_expected_path != normalized_payload_path:
        out["status"] = PROOF_STATUS_STALE
        out["reason"] = f"stale_{path.stem}_model_path_mismatch"
        return out

    out["status"] = PROOF_STATUS_OK
    out["reason"] = "ok"
    return out


def proof_bundle_status(
    *,
    model_path: str | Path | None,
    required_files: list[str],
    optional_files: list[str] | None = None,
    expected_model_version: str | None = None,
) -> dict[str, Any]:
    optional_files = optional_files or []
    ordered_files: list[str] = []
    for filename in [*(required_files or []), *optional_files]:
        token = str(filename or "").strip()
        if token and token not in ordered_files:
            ordered_files.append(token)

    file_states = {
        filename: proof_file_state(
            model_path=model_path,
            filename=filename,
            expected_model_version=expected_model_version,
        )
        for filename in ordered_files
    }
    missing_required = [
        filename for filename in required_files if file_states.get(filename, {}).get("status") == PROOF_STATUS_MISSING
    ]
    invalid_required = [
        filename for filename in required_files if file_states.get(filename, {}).get("status") == PROOF_STATUS_INVALID
    ]
    stale_required = [
        filename for filename in required_files if file_states.get(filename, {}).get("status") == PROOF_STATUS_STALE
    ]
    required_complete = not bool(missing_required or invalid_required or stale_required)
    reason = "ok"
    if missing_required:
        reason = str(file_states[missing_required[0]].get("reason") or "missing_required_proof")
    elif invalid_required:
        reason = str(file_states[invalid_required[0]].get("reason") or "invalid_required_proof")
    elif stale_required:
        reason = str(file_states[stale_required[0]].get("reason") or "stale_required_proof")

    return {
        "model_path": str(Path(str(model_path or "").strip()).expanduser()) if str(model_path or "").strip() else "",
        "artifact_dir": str(artifact_dir_for_model_path(model_path)) if str(model_path or "").strip() else "",
        "required_files": [str(x) for x in required_files if str(x or "").strip()],
        "optional_files": [str(x) for x in optional_files if str(x or "").strip()],
        "files": file_states,
        "required_complete": bool(required_complete),
        "missing_required": missing_required,
        "invalid_required": invalid_required,
        "stale_required": stale_required,
        "reason": reason,
    }
