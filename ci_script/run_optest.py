#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Run optest with a CI-local fix for pytest summaries followed by cleanup logs.

optest 1.0.0 reads counts and duration from the last non-empty log line. Patch
those readers before running the CLI, so its existing subprocess-failure check
still applies. Repairing the exported JSON instead could hide a crash after a
passing pytest summary: accuracy rows do not retain the subprocess exit code.
This compatibility fix can be removed once the installed optest handles this.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SESSION_RE = re.compile(r"^=+ test session starts =+$")
_SUMMARY_RE = re.compile(
    r"^=+\s+(?P<stats>.+?)\s+in\s+(?P<seconds>\d+(?:\.\d+)?)s"
    r"(?:\s+\([^)]*\))?\s+=+$"
)
_COUNT_RE = re.compile(
    r"(?P<count>\d+)\s+"
    r"(?P<status>passed|failed|skipped|errors?|xfailed|xpassed|deselected|warnings?|rerun)"
)


def _pytest_summary(log_path: Path) -> tuple[dict[str, int], float] | None:
    """Read the last completed summary, without reusing an older test session.

    Stream the log so long cleanup output cannot push the summary out of a
    fixed-size tail buffer. Only a complete pytest tally with a duration counts.
    """
    summary = None
    try:
        with log_path.open(encoding="utf-8", errors="replace") as stream:
            for raw_line in stream:
                line = _ANSI_RE.sub("", raw_line).strip()
                if _SESSION_RE.fullmatch(line):
                    summary = None
                    continue
                match = _SUMMARY_RE.fullmatch(line)
                if match is None:
                    continue
                counts = {"passed": 0, "failed": 0, "skipped": 0}
                for item in match["stats"].split(","):
                    tally = _COUNT_RE.fullmatch(item.strip())
                    if tally is None:
                        break
                    status = tally["status"]
                    count = int(tally["count"])
                    if status in ("passed", "xpassed"):
                        counts["passed"] += count
                    elif status in ("failed", "error", "errors"):
                        counts["failed"] += count
                    elif status in ("skipped", "xfailed"):
                        counts["skipped"] += count
                else:
                    counts["total"] = sum(counts.values())
                    if counts["total"]:
                        summary = counts, round(float(match["seconds"]), 2)
    except OSError:
        return None
    return summary


def _pytest_counts(log_path: Path) -> dict[str, int] | None:
    summary = _pytest_summary(log_path)
    return summary[0] if summary is not None else None


def _pytest_duration(log_path: Path) -> float | None:
    summary = _pytest_summary(log_path)
    return summary[1] if summary is not None else None


def main(argv: list[str] | None = None) -> int:
    from optest import runner

    # Fail visibly if optest changes this private interface, rather than silently
    # running without the workaround. No installed package files are modified.
    for name in ("_pytest_counts", "_pytest_duration"):
        if not callable(getattr(runner, name, None)):
            raise RuntimeError(f"Unsupported optest runner: missing {name}")
    runner._pytest_counts = _pytest_counts
    runner._pytest_duration = _pytest_duration

    from optest.cli import main as optest_main

    print("[aiter/ci] Using pytest summary parser with trailing-output support", flush=True)
    return optest_main(argv)


if __name__ == "__main__":
    sys.exit(main())
