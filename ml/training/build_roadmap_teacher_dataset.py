from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def setup_django() -> None:
    root = Path(__file__).resolve().parents[2]
    backend_dir = root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
    import django

    django.setup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3650)
    parser.add_argument("--out-dir", type=str, default="data/ml/roadmap_teacher_v1")
    parser.add_argument("--include-ga", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_django()

    from django.core.management import call_command

    call_command(
        "build_roadmap_teacher_dataset",
        days=int(args.days),
        out_dir=str(args.out_dir),
        include_ga=bool(args.include_ga),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
