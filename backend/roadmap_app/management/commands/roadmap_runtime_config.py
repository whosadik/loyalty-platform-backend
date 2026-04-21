from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from roadmap_app import runtime_config


class Command(BaseCommand):
    help = (
        "View or modify the roadmap runtime config table. "
        "Admin override for ML flags (e.g. runtime_freeze_ml) without editing .env."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--list",
            action="store_true",
            help="List all current overrides (default action when no --set/--unset given).",
        )
        parser.add_argument(
            "--set",
            nargs="*",
            default=[],
            metavar="KEY=VALUE",
            help="Set one or more overrides. Repeatable.",
        )
        parser.add_argument(
            "--unset",
            nargs="*",
            default=[],
            metavar="KEY",
            help="Remove one or more overrides. Repeatable.",
        )
        parser.add_argument(
            "--note",
            default="",
            help="Optional note stored with each --set change.",
        )
        parser.add_argument(
            "--by",
            default="",
            help="Optional identifier for who made the change (stored in updated_by).",
        )

    def handle(self, *args, **options) -> None:
        did_write = False
        for pair in options["set"]:
            if "=" not in pair:
                raise CommandError(
                    f"Invalid --set value (expected KEY=VALUE): {pair!r}"
                )
            key, value = pair.split("=", 1)
            key = key.strip()
            if not key:
                raise CommandError(f"Empty key in --set value: {pair!r}")
            try:
                runtime_config.set_value(
                    key,
                    value,
                    updated_by=options["by"],
                    note=options["note"],
                )
            except ValueError as exc:
                raise CommandError(str(exc))
            self.stdout.write(self.style.SUCCESS(f"set   {key}={value}"))
            did_write = True

        for key in options["unset"]:
            key = key.strip()
            if not key:
                continue
            removed = runtime_config.unset_value(key)
            label = "unset" if removed else "noop "
            self.stdout.write(f"{label} {key}")
            did_write = True

        if options["list"] or not did_write:
            current = runtime_config.list_values()
            if not current:
                self.stdout.write("(no runtime overrides set)")
            else:
                for key in sorted(current):
                    self.stdout.write(f"{key}={current[key]}")
