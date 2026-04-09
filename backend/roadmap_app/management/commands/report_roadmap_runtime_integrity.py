from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from roadmap_app.integrity import (
    active_fragrance_runtime_integrity_counts,
    legacy_bad_fragrance_completion_details,
)
from roadmap_app.ml_artifact_proof import artifact_dir_for_model_path


def _build_payload() -> dict:
    runtime_counts = active_fragrance_runtime_integrity_counts()
    legacy_counts = legacy_bad_fragrance_completion_details(recent_days=30)
    return {
        "generated_at_utc": datetime.now(dt_timezone.utc).isoformat(),
        "runtime": runtime_counts,
        "legacy": legacy_counts,
        "fragrance_runtime_status": (
            "pass" if int(runtime_counts.get("active_fragrance_slot_mismatch_count", 0)) == 0 else "fail"
        ),
    }


def _build_markdown(payload: dict) -> str:
    runtime = payload.get("runtime") or {}
    legacy = payload.get("legacy") or {}
    lines = [
        "# Roadmap Runtime Integrity",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc')}`",
        f"- active_fragrance_next_steps_total: `{runtime.get('active_fragrance_next_steps_total')}`",
        "- active_fragrance_next_steps_with_recommended_product: "
        f"`{runtime.get('active_fragrance_next_steps_with_recommended_product')}`",
        f"- active_fragrance_slot_mismatch_count: `{runtime.get('active_fragrance_slot_mismatch_count')}`",
        f"- fragrance_runtime_status: `{payload.get('fragrance_runtime_status')}`",
        f"- bad_fragrance_completed_exact_match_count: `{legacy.get('bad_fragrance_completed_exact_match_count')}`",
        "- bad_fragrance_completed_exact_match_recent_30d: "
        f"`{legacy.get('bad_fragrance_completed_exact_match_recent_30d')}`",
        f"- fragrance_completed_step_state_drift_count: `{legacy.get('step_state_drift_count')}`",
        "- fragrance_completed_step_state_drift_recent_30d: "
        f"`{legacy.get('step_state_drift_recent_30d')}`",
        f"- fragrance_legacy_bucket: `{legacy.get('legacy_bucket')}`",
    ]
    return "\n".join(lines)


class Command(BaseCommand):
    help = "Report fragrance runtime integrity and optionally write a reusable qualification report bundle."

    def add_arguments(self, parser):
        parser.add_argument("--output-json", type=str, default=None)
        parser.add_argument("--output-md", type=str, default=None)
        parser.add_argument(
            "--sync-runtime-artifacts",
            action="store_true",
            default=False,
            help="Write fragrance slot integrity report into configured nextstep artifact dirs as fragrance_slot_report.{json,md}.",
        )

    def handle(self, *args, **options):
        payload = _build_payload()
        markdown = _build_markdown(payload)

        output_json_raw = str(options.get("output_json") or "").strip()
        output_md_raw = str(options.get("output_md") or "").strip()
        if output_json_raw:
            output_json = Path(output_json_raw).resolve()
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.stdout.write(f"[report_roadmap_runtime_integrity] json={output_json}")
        if output_md_raw:
            output_md = Path(output_md_raw).resolve()
            output_md.parent.mkdir(parents=True, exist_ok=True)
            output_md.write_text(markdown, encoding="utf-8")
            self.stdout.write(f"[report_roadmap_runtime_integrity] md={output_md}")

        if bool(options.get("sync_runtime_artifacts")):
            artifact_dirs: set[str] = set()
            for setting_name in [
                "ROADMAP_NEXTSTEP_V4_MODEL_PATH",
                "ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH",
            ]:
                raw_path = str(getattr(settings, setting_name, "") or "").strip()
                if not raw_path:
                    continue
                artifact_dirs.add(str(artifact_dir_for_model_path(raw_path)))
            for raw_dir in sorted(artifact_dirs):
                artifact_dir = Path(raw_dir).resolve()
                artifact_dir.mkdir(parents=True, exist_ok=True)
                json_path = artifact_dir / "fragrance_slot_report.json"
                md_path = artifact_dir / "fragrance_slot_report.md"
                json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                md_path.write_text(markdown, encoding="utf-8")
                self.stdout.write(f"[report_roadmap_runtime_integrity] synced={json_path}")

        runtime = payload["runtime"]
        legacy = payload["legacy"]
        self.stdout.write(
            f"active_fragrance_next_steps_total={runtime['active_fragrance_next_steps_total']}"
        )
        self.stdout.write(
            "active_fragrance_next_steps_with_recommended_product="
            f"{runtime['active_fragrance_next_steps_with_recommended_product']}"
        )
        self.stdout.write(
            f"active_fragrance_slot_mismatch_count={runtime['active_fragrance_slot_mismatch_count']}"
        )
        self.stdout.write(f"fragrance_runtime_status={payload['fragrance_runtime_status']}")
        self.stdout.write(
            f"bad_fragrance_completed_exact_match_count={legacy['bad_fragrance_completed_exact_match_count']}"
        )
        self.stdout.write(
            "bad_fragrance_completed_exact_match_recent_30d="
            f"{legacy['bad_fragrance_completed_exact_match_recent_30d']}"
        )
        self.stdout.write(f"fragrance_completed_step_state_drift_count={legacy['step_state_drift_count']}")
        self.stdout.write(
            "fragrance_completed_step_state_drift_recent_30d="
            f"{legacy['step_state_drift_recent_30d']}"
        )
        self.stdout.write(f"fragrance_legacy_bucket={legacy['legacy_bucket']}")
