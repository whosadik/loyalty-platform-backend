from __future__ import annotations

from collections import OrderedDict
from typing import Any

from django.apps import apps
from django.db import connection


def quote_qualified_name(name: str) -> str:
    parts = []
    for raw in str(name or "").split("."):
        part = raw.strip().strip('"')
        if not part:
            continue
        parts.append(connection.ops.quote_name(part))
    return ".".join(parts)


def default_tables_with_id_pk() -> list[str]:
    tables: "OrderedDict[str, None]" = OrderedDict()
    for model in apps.get_models():
        opts = model._meta
        if opts.proxy or not opts.managed:
            continue
        pk = opts.pk
        if pk is None:
            continue
        if str(pk.column) != "id":
            continue
        tables[opts.db_table] = None
    return list(tables.keys())


def inspect_postgres_sequences(*, tables: list[str] | None = None) -> list[dict[str, Any]]:
    table_list = list(OrderedDict((str(t), None) for t in (tables or default_tables_with_id_pk())).keys())
    rows: list[dict[str, Any]] = []
    if connection.vendor != "postgresql":
        return rows

    with connection.cursor() as cursor:
        for table_name in table_list:
            row: dict[str, Any] = {
                "table": table_name,
                "sequence": None,
                "max_id": None,
                "old_nextval": None,
                "new_nextval": None,
                "drifted": None,
                "status": "preview",
                "error": None,
            }
            try:
                cursor.execute("SELECT pg_get_serial_sequence(%s, %s)", [table_name, "id"])
                seq_row = cursor.fetchone()
                seq_name = seq_row[0] if seq_row else None
                row["sequence"] = seq_name
                if not seq_name:
                    row["status"] = "skipped_no_sequence"
                    rows.append(row)
                    continue

                q_table = quote_qualified_name(table_name)
                q_col = connection.ops.quote_name("id")
                cursor.execute(f"SELECT COALESCE(MAX({q_col}), 0) FROM {q_table}")
                max_id = int(cursor.fetchone()[0] or 0)
                row["max_id"] = max_id
                row["new_nextval"] = max_id + 1 if max_id > 0 else 2

                old_nextval = None
                try:
                    q_seq = quote_qualified_name(seq_name)
                    cursor.execute(f"SELECT last_value, is_called FROM {q_seq}")
                    seq_state = cursor.fetchone()
                    if seq_state:
                        last_value = int(seq_state[0])
                        is_called = bool(seq_state[1])
                        old_nextval = last_value + 1 if is_called else last_value
                except Exception:
                    old_nextval = None
                row["old_nextval"] = old_nextval
                if old_nextval is not None:
                    row["drifted"] = bool(old_nextval <= max_id)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
            rows.append(row)
    return rows


def apply_postgres_sequences(*, tables: list[str] | None = None) -> list[dict[str, Any]]:
    rows = inspect_postgres_sequences(tables=tables)
    if connection.vendor != "postgresql":
        return rows

    by_table = {str(row["table"]): row for row in rows}
    with connection.cursor() as cursor:
        for table_name, row in by_table.items():
            if row.get("status") == "error":
                continue
            seq_name = row.get("sequence")
            if not seq_name:
                continue
            try:
                max_id = int(row.get("max_id") or 0)
                set_to = max_id if max_id > 0 else 1
                cursor.execute("SELECT setval(%s, %s, true)", [seq_name, set_to])
                row["new_nextval"] = set_to + 1
                row["status"] = "applied"
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
    return list(by_table.values())
