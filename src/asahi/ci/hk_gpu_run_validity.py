# SPDX-License-Identifier: MIT
# Copyright 2026 Ryan Murray <ryan@aurorasilicon.org>
"""Compare kernel snapshots for a GPU qualification run on a shared boot."""

import errno
import re


ENOSPC = re.compile(rf"ENOSPC|\berr(?:or)?\s*[=:]\s*-?{errno.ENOSPC}\b")


def submit_failure_counts(log):
    lines = [line for line in log.splitlines() if "submit failed" in line]
    return len(lines), sum(bool(ENOSPC.search(line)) for line in lines)


def evaluate(before, after):
    before_count, before_enospc = submit_failure_counts(before)
    after_count, after_enospc = submit_failure_counts(after)
    return {
        "submit_failed_before": before_count,
        "submit_failed_after": after_count,
        "submit_enospc_before": before_enospc,
        "submit_enospc_after": after_enospc,
        "valid": not (before_enospc or after_enospc) and
                 after_count <= before_count,
    }
