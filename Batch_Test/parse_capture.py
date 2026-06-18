#!/usr/bin/env python
# encoding: utf-8

"""
Offline UART capture splitter.

Run after grabbing a raw capture from the FPGA, e.g.:
    stty -F /dev/ttyUSB0 1000000 raw -echo
    sed '/TEST_MODEL_PASS_END/q' /dev/ttyUSB0 > capture.bin

Then:
    python parse_capture.py capture.bin

Re-uses uart_monitor.process_session() so the per-layer .txt format is identical
to a live Python run.  This is the gold-standard reference: if Python is losing
bytes online, the cat-capture parsed by this script will still be complete.
"""
import sys
import os
import re

from uart_monitor import (
    process_session, OUTPUT_DIR,
    LAYER_BOUNDARY_RE, HEADER_RE, DATA_OPEN_RE, NO_BUF_RE,
    SESSION_MARKERS, END_MARKER, RESULT_RE,
)


# The first timing line is "CreateWithBitmap" (MobileNet) or "Load Input"
# (ResNet-18 / LeNet) depending on the firmware test function — accept either.
TIMING_RE = re.compile(
    r"(?:CreateWithBitmap|Load Input)\s*:\s*([0-9.]+)\s*s.*?"
    r"Model Parse\s*:\s*([0-9.]+)\s*s.*?"
    r"Inference\s*:\s*([0-9.]+)\s*s.*?"
    r"Total\s*:\s*([0-9.]+)\s*s",
    re.DOTALL
)

TIMING_FIELDS = ["load_input", "model_parse", "inference", "total"]
TIMING_LABELS = {
    "load_input":  "Load Input",
    "model_parse": "Model Parse",
    "inference":   "Inference",
    "total":       "Total",
}


def parse_timing(text):
    m = TIMING_RE.search(text)
    if not m:
        return None
    return {
        "load_input":  float(m.group(1)),
        "model_parse": float(m.group(2)),
        "inference":   float(m.group(3)),
        "total":       float(m.group(4)),
    }


def quick_summary(text):
    """Print a one-shot integrity summary before doing the heavy parse."""
    boundaries = list(LAYER_BOUNDARY_RE.finditer(text))
    layer_ids = sorted({int(re.search(r'\d+', m.group(0)).group(0))
                        for m in boundaries})

    headers = list(HEADER_RE.finditer(text))
    data_opens = list(DATA_OPEN_RE.finditer(text))
    no_bufs = list(NO_BUF_RE.finditer(text))

    header_ids   = sorted({int(m.group(1)) for m in headers})
    data_ids     = sorted({int(m.group(1)) for m in data_opens})
    no_buf_ids   = sorted({int(m.group(1)) for m in no_bufs})

    text_lower = text.lower()
    has_end = END_MARKER.decode() in text_lower

    found_start = None
    for mk in SESSION_MARKERS:
        idx = text_lower.find(mk.decode())
        if idx >= 0 and (found_start is None or idx < found_start):
            found_start = idx

    result = RESULT_RE.search(text)
    top5_str = (f"top1={result.group(1)}  top2={result.group(2)}  "
                f"top3={result.group(3)}  top4={result.group(4)}  "
                f"top5={result.group(5)}") if result else "NOT FOUND"

    timing = parse_timing(text)

    print()
    print("─" * 64)
    print("  CAPTURE INTEGRITY SUMMARY")
    print("─" * 64)
    print(f"  bytes total:              {len(text):>10,}")
    print(f"  start marker found at:    {found_start if found_start is not None else 'NOT FOUND'}")
    print(f"  END marker present:       {'YES' if has_end else 'NO'}")
    print(f"  TOP-5 RESULT:             {top5_str}")
    print(f"  [Layer N] boundaries:     {len(boundaries):>10,}")
    print(f"  unique layer indices:     {len(layer_ids):>10}    {layer_ids[:5]}...{layer_ids[-3:] if len(layer_ids) >= 3 else ''}")
    print(f"  header lines (count=…):   {len(headers):>10,}")
    print(f"  data lines (data:[ …):    {len(data_opens):>10,}")
    print(f"  no-buffer lines:          {len(no_bufs):>10,}")
    print(f"  layers with HEADER:       {len(header_ids):>10}")
    print(f"  layers with DATA:         {len(data_ids):>10}")
    missing_data = sorted(set(header_ids) - set(data_ids) - set(no_buf_ids))
    extra_data   = sorted(set(data_ids) - set(header_ids))
    if missing_data:
        print(f"  ⚠ header but NO data:    {missing_data}")
    if extra_data:
        print(f"  ⚠ data but NO header:    {extra_data}")
    print("─" * 64)
    if timing:
        print()
        print("─" * 64)
        print("  FIRMWARE TIMING")
        print("─" * 64)
        for k in TIMING_FIELDS:
            print(f"  {TIMING_LABELS[k]:<20s} {timing[k]:>10.6f} s")
        print("─" * 64)
    else:
        print("  (no timing data found)")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_capture.py <capture.bin> [more.bin ...]")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, capture_path in enumerate(sys.argv[1:], start=1):
        if not os.path.exists(capture_path):
            print(f"ERROR: {capture_path} not found")
            continue

        print(f"\n{'═' * 64}")
        print(f"  PROCESSING {capture_path}  ({os.path.getsize(capture_path):,} bytes)")
        print(f"{'═' * 64}")

        with open(capture_path, "rb") as f:
            raw = f.read()

        text = raw.decode("utf-8", errors="replace")

        # Quick pre-flight integrity check
        quick_summary(text)

        # Full parse + per-layer save (same code path as live monitor)
        process_session(text, i)


if __name__ == "__main__":
    main()
