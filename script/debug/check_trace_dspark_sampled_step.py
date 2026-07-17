#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


def _float_after(name: str, text: str) -> float | None:
    match = re.search(rf"{re.escape(name)}=([-+0-9.eE]+)", text)
    return None if match is None else float(match.group(1))


def _section(text: str, name: str) -> str:
    marker = f"TRACE {name}"
    start = text.find(marker)
    if start < 0:
        return ""
    next_start = text.find("\nTRACE ", start + len(marker))
    return text[start:] if next_start < 0 else text[start:next_start]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    text = args.log_path.read_text(errors="replace")
    failures: list[str] = []

    packed = _section(text, "packed_raw_alignment")
    if not packed:
        failures.append("missing TRACE packed_raw_alignment")
    else:
        if "token_match=True" not in packed:
            failures.append("packed/raw token mismatch")
        hidden = _float_after("hidden_max_abs_diff", packed)
        last = _float_after("verifier_last_max_abs_diff", packed)
        if hidden is None or not math.isclose(hidden, 0.0, abs_tol=args.tol):
            failures.append(f"packed/raw hidden diff={hidden}")
        if last is None or not math.isclose(last, 0.0, abs_tol=args.tol):
            failures.append(f"packed/raw verifier_last diff={last}")

    sampled = _section(text, "sampled_verifier_prefix_compare.summary")
    if not sampled:
        failures.append("missing sampled verifier prefix summary")
    elif re.search(r",[-+0-9.eE]+$", sampled, re.MULTILINE) is None:
        failures.append("sampled prefix summary has no rows")
    else:
        for line in sampled.splitlines():
            parts = line.strip().split(",")
            if len(parts) == 7 and parts[0].isdigit():
                diff_doc = abs(float(parts[-1]))
                if diff_doc > args.tol:
                    failures.append(f"sampled verifier diff_doc={diff_doc} line={line.strip()}")
                    break

    if "TRACE gt_doc_prefix.hidden_projection" not in text:
        failures.append("missing gt doc-prefix hidden projection")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        print(f"log={args.log_path}")
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
