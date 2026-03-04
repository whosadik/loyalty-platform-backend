from __future__ import annotations

from django.db import connection
from django.core.management.base import BaseCommand

from admin_tools.postgres_sequences import (
    apply_postgres_sequences,
    default_tables_with_id_pk,
    inspect_postgres_sequences,
)


class Command(BaseCommand):
    help = "Inspect/fix PostgreSQL table sequences to match MAX(id)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--tables",
            nargs="*",
            default=None,
            help="Specific DB tables to process (default: all managed tables with PK column 'id').",
        )
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument("--dry-run", action="store_true", help="Only print planned setval values.")
        mode.add_argument("--apply", action="store_true", help="Apply setval for selected tables.")

    def handle(self, *args, **options):
        if connection.vendor != "postgresql":
            self.stdout.write(self.style.WARNING("Non-PostgreSQL DB detected. Sequence fix skipped."))
            return

        apply_mode = bool(options.get("apply"))
        dry_run = bool(options.get("dry_run")) or not apply_mode
        raw_tables = options.get("tables") or []
        tables = list(raw_tables) if raw_tables else default_tables_with_id_pk()

        if not tables:
            self.stdout.write(self.style.WARNING("No candidate tables found."))
            return

        rows = apply_postgres_sequences(tables=tables) if apply_mode else inspect_postgres_sequences(tables=tables)
        title = "apply" if apply_mode and not dry_run else "dry-run"
        self.stdout.write(f"[fix_postgres_sequences] mode={title}, tables={len(tables)}")
        self.stdout.write("table | seq_name | max_id | old_nextval | new_nextval | status")
        self.stdout.write("-" * 120)

        errors = 0
        drifted = 0
        applied = 0
        for row in rows:
            if row.get("drifted"):
                drifted += 1
            if row.get("status") == "applied":
                applied += 1
            if row.get("status") == "error":
                errors += 1

            table = str(row.get("table") or "-")
            seq_name = str(row.get("sequence") or "-")
            max_id = row.get("max_id")
            old_next = row.get("old_nextval")
            new_next = row.get("new_nextval")
            status = str(row.get("status") or "-")
            self.stdout.write(
                f"{table} | {seq_name} | {max_id if max_id is not None else '-'} | "
                f"{old_next if old_next is not None else '-'} | "
                f"{new_next if new_next is not None else '-'} | {status}"
            )
            err = row.get("error")
            if err:
                self.stdout.write(self.style.WARNING(f"  error: {err}"))

        self.stdout.write(
            f"[fix_postgres_sequences] scanned={len(rows)}, drifted={drifted}, "
            f"applied={applied}, errors={errors}"
        )
