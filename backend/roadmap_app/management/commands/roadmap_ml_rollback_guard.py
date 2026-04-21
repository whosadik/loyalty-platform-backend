from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from roadmap_app.ml_rollback_guard import (
    DEFAULT_MAX_ERROR_RATE_PCT,
    DEFAULT_MAX_FALLBACK_RATE_PCT,
    DEFAULT_MAX_P95_LATENCY_MS,
    DEFAULT_MIN_SAMPLE_SIZE,
    DEFAULT_WINDOW_MINUTES,
    GuardThresholds,
    enforce_rollback_guard,
    evaluate_rollback_guard,
)


class Command(BaseCommand):
    help = (
        "Evaluate the roadmap ML rollback guard against RoadmapMLInvocation log. "
        "Without --enforce, only reports metrics & breaches. With --enforce, flips "
        "runtime_freeze_ml=true via runtime_config when any category breaches."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument("--window-minutes", type=int, default=DEFAULT_WINDOW_MINUTES)
        parser.add_argument("--max-error-rate-pct", type=float, default=DEFAULT_MAX_ERROR_RATE_PCT)
        parser.add_argument("--max-fallback-rate-pct", type=float, default=DEFAULT_MAX_FALLBACK_RATE_PCT)
        parser.add_argument("--max-p95-latency-ms", type=float, default=DEFAULT_MAX_P95_LATENCY_MS)
        parser.add_argument("--min-sample-size", type=int, default=DEFAULT_MIN_SAMPLE_SIZE)
        parser.add_argument(
            "--enforce",
            action="store_true",
            help="Actually trip runtime_freeze_ml when thresholds breached (otherwise dry-run).",
        )
        parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
        parser.add_argument("--actor", default="rollback_guard_cli")

    def handle(self, *args, **options) -> None:
        thresholds = GuardThresholds(
            max_error_rate_pct=options["max_error_rate_pct"],
            max_fallback_rate_pct=options["max_fallback_rate_pct"],
            max_p95_latency_ms=options["max_p95_latency_ms"],
            min_sample_size=options["min_sample_size"],
        )
        if options["enforce"]:
            report = enforce_rollback_guard(
                window_minutes=options["window_minutes"],
                thresholds=thresholds,
                actor=options["actor"],
            )
        else:
            report = evaluate_rollback_guard(
                window_minutes=options["window_minutes"],
                thresholds=thresholds,
            )
            report["action_taken"] = "dry_run"
            report["frozen_after"] = report["frozen_before"]

        if options["json"]:
            self.stdout.write(
                json.dumps(report, indent=2, sort_keys=True, default=str)
            )
            return

        self.stdout.write(
            f"window={report['window_minutes']}m  cutoff={report['cutoff_utc']}"
        )
        self.stdout.write(
            f"frozen_before={report['frozen_before']} any_breach={report['any_breach']} "
            f"action={report['action_taken']} frozen_after={report.get('frozen_after')}"
        )
        if not report["per_category"]:
            self.stdout.write("  (no invocations in window)")
            return
        for cat in sorted(report["per_category"]):
            payload = report["per_category"][cat]
            p95 = payload["p95_latency_ms"]
            p95_txt = "n/a" if p95 is None else f"{p95:.2f}"
            self.stdout.write(
                f"  [{cat}] total={payload['total']} attempts={payload['predict_attempts']} "
                f"errors={payload['errors']} err%={payload['error_rate_pct']:.2f} "
                f"fb={payload['fallbacks']} fb%={payload['fallback_rate_pct']:.2f} "
                f"p95={p95_txt}ms"
                + (" [sample<min]" if payload.get("insufficient_sample") else "")
            )
            for breach in payload.get("breaches", []):
                self.stdout.write(
                    f"    BREACH {breach['metric']}={breach['value']:.2f} > {breach['threshold']}"
                )
