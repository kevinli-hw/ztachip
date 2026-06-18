"""
ztachip UART Monitor
Connects to serial port, prints first 10 values of each layer's ch0 output,
saves full results to per-session txt files, split by channel based on
flat (CHW) vs interleaved (HWC) storage layout.

Usage:
    python uart_monitor.py [port] [baud]
    python uart_monitor.py COM3 115200
    python uart_monitor.py /dev/ttyUSB1 115200
"""

import serial
import re
import os
import sys
import threading
import queue
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────
PORT       = sys.argv[1] if len(sys.argv) > 1 else "COM3"
BAUD_RATE  = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
OUTPUT_DIR = "layer_outputs"

# Regex patterns matching the exact printf formats in test.cpp
#   "[Layer %3d] %-9s out_buf=%-3d count=%-7zu min=%4d max=%4d shape=[...]\n"
#   "[Layer %3d] %-9s out_buf=%-3d count=%-7zu min=%4d max=%4d [interleaved] shape=[...]\n"
#   "[Layer %3d] %-9s out_buf=%-3d [no buffer]\n"
#   "[Layer %3d] %-9s first8:[ v1 v2 ... vN]\n"
HEADER_RE = re.compile(
    r'\[Layer\s*(\d+)\]\s+(\S+)\s+out_buf=(\d+)\s+count=(\d+)\s+'
    r'min=\s*(-?\d+)\s+max=\s*(-?\d+)'
)
NO_BUF_RE = re.compile(r'\[Layer\s*(\d+)\]\s+(\S+)\s+out_buf=(\d+)\s+\[no buffer\]')
# Match the START of a data line; payload is whatever sits after the ":[".
# We don't anchor to ']' or end-of-line because the segment splitter (below)
# bounds payloads by the NEXT '[Layer NNN]' opener, which is robust to lost \n.
DATA_OPEN_RE = re.compile(r'\[Layer\s*(\d+)\]\s+(\S+)\s+\w+:\[', re.DOTALL)
SHAPE_RE     = re.compile(r'shape=\[([^\]]*)\]')
INT_RE       = re.compile(r'-?\d+')

# Splits the session text on every "[Layer NNN]" opener — robust to dropped \n
# between log lines.  Note: shape literals like "[1,7,7,160]" do NOT match
# because we require "Layer" + whitespace + digits.
LAYER_BOUNDARY_RE = re.compile(r'\[Layer\s*\d+\]')

# Final classification result line printed once per pass:
#   "RESULT top1=265 top2=283 top3=287 top4=281 top5=282\n"
# Anything from this line onward (RESULT + PASS_END marker) must be stripped
# from the text BEFORE per-layer parsing — otherwise the last layer's data
# segment runs all the way to EOF and the integer regex inside it gobbles up
# 265, 283, 287, 281, 282 as five extra "data" values, corrupting the layer
# dump and silently producing the wrong file content.
RESULT_RE = re.compile(
    r'RESULT\s+top1=(\d+)\s+top2=(\d+)\s+top3=(\d+)\s+top4=(\d+)\s+top5=(\d+)'
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_session_dir():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"session_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def parse_shape(s):
    """'1,28,28,8' -> [1,28,28,8]"""
    if not s:
        return None
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except ValueError:
        return None


def shape_to_chw(shape):
    """
    Normalize any dumped shape to (C, H, W).  UART prints output_shape in
    TFLite NHWC order, e.g. [1, 112, 112, 32]. 2D (FC/Mean) becomes (C, 1, 1).

    Returns (C, H, W) or None if shape can't be normalized.
    """
    if not shape:
        return None
    d = [x for x in shape if x > 0]
    if len(d) == 4:        # NHWC: N, H, W, C
        _, H, W, C = d
        return (C, H, W)
    if len(d) == 3:        # HWC
        H, W, C = d
        return (C, H, W)
    if len(d) == 2:        # N, C  (FC / Mean)
        return (d[1], 1, 1)
    if len(d) == 1:
        return (d[0], 1, 1)
    return None


def split_channels(values, shape, interleaved):
    """
    Returns list of (channel_idx, 1d_values, H, W) tuples.

    flat  (CHW layout): ch_c = values[c*H*W : (c+1)*H*W]
    inter (HWC layout): ch_c = values[c : H*W*C : C]
    """
    chw = shape_to_chw(shape)
    if chw is None:
        return [(0, values, None, None)]

    C, H, W = chw
    expected = C * H * W
    if len(values) < expected:
        return [(0, values, None, None)]

    out = []
    if interleaved:
        for c in range(C):
            out.append((c, values[c:expected:C], H, W))
    else:
        hw = H * W
        for c in range(C):
            out.append((c, values[c * hw:(c + 1) * hw], H, W))
    return out


def print_layer_summary(layer_idx, layer_type, values, info):
    first10 = values[:10]
    bar = "─" * 64
    chw = info.get("shape")
    raw = info.get("shape_raw")
    raw_str = (f"[{','.join(str(d) for d in raw)}]" if raw else "?")
    chw_str = f"(C={chw[0]},H={chw[1]},W={chw[2]})" if chw else "?"
    print(f"\n{bar}")
    print(f"  Layer {layer_idx:>3d}  │  {layer_type}  │  shape={raw_str} -> {chw_str}"
          + ("  [interleaved]" if info.get("interleaved") else "  [flat]"))
    print(f"  count={info.get('count','?')}  min={info.get('min','?')}  max={info.get('max','?')}")
    print(f"  raw[0:10]:  " + "  ".join(f"{v:>4}" for v in first10))
    print(bar)


def save_layer(session_dir, layer_idx, layer_type, values, info):
    fname = os.path.join(session_dir, f"layer_{layer_idx:03d}_{layer_type}.txt")
    chw = info.get("shape")            # (C, H, W) or None
    interleaved = info.get("interleaved", False)
    channels = split_channels(values, info.get("shape_raw"), interleaved)

    with open(fname, "w") as f:
        if chw:
            f.write(f"shape=({chw[0]}, {chw[1]}, {chw[2]}), dtype=int8\n")
        else:
            f.write("shape=?, dtype=int8\n")
        f.write(f"# Layer {layer_idx} {layer_type}  "
                f"layout={'interleaved (HWC)' if interleaved else 'flat (CHW)'}  "
                f"count={info.get('count','?')}  "
                f"min={info.get('min','?')}  max={info.get('max','?')}  "
                f"parsed={len(values)}\n\n")

        for c_idx, ch_vals, H, W in channels:
            f.write(f"===== channel {c_idx} =====\n")
            if H is not None and W is not None and H * W > 0:
                for r in range(H):
                    row = ch_vals[r * W : (r + 1) * W]
                    f.write(" ".join(str(v) for v in row) + "\n")
            else:
                f.write(" ".join(str(v) for v in ch_vals) + "\n")
            f.write("\n")

    print(f"  → saved: {fname}  ({len(channels)} channel{'s' if len(channels)!=1 else ''})")


# ── Session processor (called between sessions) ────────────────────────────────

def process_session(text, session_idx):
    """
    Parse a fully-captured session's raw text and write per-layer files.
    Called only AFTER the session is complete (next 'model test start' arrived
    or user hit Ctrl+C), so disk I/O here cannot stall the UART read loop.
    """
    session_dir = make_session_dir()
    print(f"\n  → session {session_idx} dir: {session_dir}")

    # ── Pull out the top-5 RESULT line and TRUNCATE it (and everything after,
    # i.e. the PASS_END marker) from the text before layer parsing.  Otherwise
    # the LAST layer's data segment extends all the way to EOF and the integer
    # regex eats the top-1..top-5 IDs as bogus extra "data" values.
    result_match = RESULT_RE.search(text)
    top5 = None
    if result_match:
        top5 = [int(result_match.group(i)) for i in range(1, 6)]
        text = text[:result_match.start()]

    # Also strip the "=== Timing Results ===" block that sits between the last
    # layer's data line and the RESULT line.  Without this, its float numbers
    # (e.g. "0.001234") get parsed as extra integers by INT_RE, inflating the
    # last layer's value count (typically +8: two ints per float × 4 lines).
    timing_match = re.search(r'\n=== Timing Results.*', text, re.DOTALL)
    if timing_match:
        text = text[:timing_match.start()]

    layer_info     = {}            # idx -> info dict (filled when HEADER seen)
    layers_saved   = set()          # idxs for which save_layer was called
    layers_no_buf  = set()          # idxs for which "[no buffer]" was seen
    unmatched_segments = 0          # segments that started with [Layer N] but matched no pattern

    # ── Split text into per-layer segments by [Layer NNN] boundaries ────────
    # This is robust to dropped '\n' between two log lines: even if a header
    # and a data line merge, they each start with '[Layer N]' so they get split
    # apart correctly.  Each segment ends right before the NEXT '[Layer M]'
    # opener, which prevents the data parser from over-eating into the next
    # layer's content.
    boundaries = list(LAYER_BOUNDARY_RE.finditer(text))
    if not boundaries:
        print("  ⚠ no '[Layer N]' markers found in session text")

    for i, m in enumerate(boundaries):
        seg_start = m.start()
        seg_end   = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
        segment   = text[seg_start:seg_end]

        # ── Header? (look for "out_buf=" + "count=" pattern) ─────────────
        h = HEADER_RE.search(segment)
        if h:
            idx       = int(h.group(1))
            sm        = SHAPE_RE.search(segment)
            raw_shape = parse_shape(sm.group(1)) if sm else None
            chw       = shape_to_chw(raw_shape)
            info      = {
                "type":        h.group(2),
                "out_buf":     int(h.group(3)),
                "count":       int(h.group(4)),
                "min":         int(h.group(5)),
                "max":         int(h.group(6)),
                "shape_raw":   raw_shape,
                "shape":       chw,
                "interleaved": "[interleaved]" in segment,
            }
            layer_info[idx] = info
            raw_str = (f"[{','.join(str(d) for d in raw_shape)}]"
                       if raw_shape else "?")
            chw_str = f"(C={chw[0]},H={chw[1]},W={chw[2]})" if chw else "?"
            print(f"\n[Layer {idx:>3d}] {info['type']:9s}  "
                  f"count={info['count']:<7} min={info['min']:4} max={info['max']:4}  "
                  f"shape={raw_str} -> {chw_str}"
                  + ("  [interleaved]" if info["interleaved"] else ""),
                  flush=True)
            continue

        # ── No-buffer? ───────────────────────────────────────────────────
        n = NO_BUF_RE.search(segment)
        if n:
            idx = int(n.group(1))
            print(f"\n[Layer {idx:>3d}] {n.group(2):9s}  [no buffer]")
            layers_no_buf.add(idx)
            continue

        # ── Data? ────────────────────────────────────────────────────────
        d = DATA_OPEN_RE.search(segment)
        if d:
            idx     = int(d.group(1))
            ltype   = d.group(2)
            payload = segment[d.end():]
            # strip trailing whitespace + optional ']' (if the line's terminator
            # made it through)
            payload = payload.rstrip().rstrip(']').rstrip()
            values  = [int(x) for x in INT_RE.findall(payload)]
            info    = layer_info.get(idx, {"type": ltype})

            chw = info.get("shape")
            expected = info.get("count")
            if expected is not None and len(values) != expected:
                if len(values) > expected:
                    print(f"  ⚠ Layer {idx}: got {len(values)} values, "
                          f"expected {expected} — OVER-count "
                          f"(likely a previous layer's '\\n' was lost; this segment "
                          f"swallowed garbage from a corrupted neighbour).")
                else:
                    print(f"  ⚠ Layer {idx}: got {len(values)} values, "
                          f"expected {expected} — bytes lost in transit; "
                          f"channel split will fallback to single channel.")

            print_layer_summary(idx, ltype, values, info)
            save_layer(session_dir, idx, ltype, values, info)
            layers_saved.add(idx)
            continue

        # Segment matched neither header / no-buf / data — corrupted.
        unmatched_segments += 1

    # ── Diagnostic summary ──────────────────────────────────────────────────
    headers_seen = set(layer_info.keys())
    saved_or_no_buf = layers_saved | layers_no_buf
    missing_data = sorted(headers_seen - saved_or_no_buf)
    no_header_but_saved = sorted(layers_saved - headers_seen)

    print(f"\n  ✓ session {session_idx} done — {len(layers_saved)} layers saved, "
          f"{len(layers_no_buf)} with no buffer")

    if top5 is not None:
        print(f"  ✓ TOP-5 RESULT:  top1={top5[0]}  top2={top5[1]}  "
              f"top3={top5[2]}  top4={top5[3]}  top5={top5[4]}")
    else:
        print("  ⚠ no RESULT top1=… line found in session "
              "(model didn't reach the print? capture truncated?)")

    if missing_data:
        print(f"  ⚠ {len(missing_data)} layer(s) had a header but NO matching data line:")
        print(f"     missing data for layers: {missing_data}")
        print(f"     (header arrived but data line was lost or corrupted in transit)")
    if no_header_but_saved:
        print(f"  ⚠ {len(no_header_but_saved)} layer(s) had data but no header — "
              f"file saved without shape info: {no_header_but_saved}")
    if unmatched_segments:
        print(f"  ⚠ {unmatched_segments} '[Layer …]' segments did not match any "
              f"known pattern (likely truncated)")
    print()


# ── Main loop: read-only, defer processing to session boundary ─────────────────

# Any of these substrings (case-insensitive) marks the beginning of a session.
# Different test entry points in test.cpp print different banners:
#   test_mobinet()       → "Total layers: 68"
#   test_model()         → "model test start."
#   test_mobilenet_v2()  → "mobilenet test start."
SESSION_MARKERS = (
    b"total layers:",
    b"model test start",
    b"mobilenet test start",
)

# End-of-pass marker printed by test_mobinet() when one full inference completes.
# When seen, the current session is finalized immediately — no need to wait for
# the next session's start marker.
END_MARKER = b"test_mobinet_pass_end"


def _find_next_marker(buf, start):
    """
    Find the earliest occurrence of any SESSION_MARKER in buf[start:].
    Returns (absolute_pos, marker_len) or (-1, 0) if none found.
    """
    tail = bytes(buf[start:]).lower()
    best_idx = -1
    best_len = 0
    for mk in SESSION_MARKERS:
        i = tail.find(mk)
        if i < 0:
            continue
        if best_idx < 0 or i < best_idx:
            best_idx = i
            best_len = len(mk)
    if best_idx < 0:
        return (-1, 0)
    return (start + best_idx, best_len)


def _reader_loop(ser, out_q, stop_event):
    """
    Dedicated reader thread.  Does NOTHING but pull bytes from the serial port
    and stuff them onto a queue.  This keeps the OS RX kernel buffer drained
    even while the main thread is doing slow work (regex sweeps, save_layer,
    print spam, etc.).
    """
    while not stop_event.is_set():
        try:
            chunk = ser.read(65536)
        except (serial.SerialException, OSError):
            break
        if chunk:
            out_q.put(chunk)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Opening {PORT} @ {BAUD_RATE} baud …  (Ctrl+C to flush & stop)")
    try:
        # very short timeout — we just want to drain the OS RX buffer ASAP.
        ser = serial.Serial(PORT, BAUD_RATE, timeout=0.05)
        # Windows: bump kernel-side RX buffer so we can absorb a full session
        # even if Python is briefly slow.  Linux ignores set_buffer_size; Linux
        # users should drop /sys/.../latency_timer to 1 instead.
        try:
            ser.set_buffer_size(rx_size=16 * 1024 * 1024,
                                tx_size=64 * 1024)
        except Exception:
            pass  # not on Windows, or pyserial too old — fine
    except serial.SerialException as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Background reader thread keeps draining the OS buffer regardless of
    # what the main thread is busy doing.  This is the fix for byte loss at
    # 1Mbaud+ when the main thread spends 10-100ms doing regex/save work.
    rx_q       = queue.Queue()
    stop_event = threading.Event()
    reader     = threading.Thread(
        target=_reader_loop, args=(ser, rx_q, stop_event),
        name="uart-reader", daemon=True,
    )
    reader.start()

    print("Reader thread running.  Main thread waits on session boundaries; "
          "OS RX buffer is drained continuously by the reader thread.\n")

    raw                = bytearray()  # bytes accumulated since current session start
    session_start      = None         # offset in `raw` of current session marker
    cur_marker_len     = 0            # length of the marker that opened current session
    sessions_seen      = 0
    last_heartbeat     = 0

    try:
        while True:
            # Pull whatever the reader thread has queued up.  Blocking get with
            # a tiny timeout so we don't busy-spin when no data is arriving.
            try:
                chunk = rx_q.get(timeout=0.05)
                raw.extend(chunk)
                # drain anything else queued without blocking, so we batch
                # processing and only run the marker scan once per burst
                while True:
                    try:
                        raw.extend(rx_q.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                pass

            # Cheap progress heartbeat so the user knows it's alive.
            if len(raw) - last_heartbeat >= 512 * 1024:
                state = (f"session {sessions_seen} in progress"
                         if session_start is not None else "waiting for session")
                print(f"  … buffered {len(raw):>9,} bytes  ({state})", flush=True)
                last_heartbeat = len(raw)

            # ── END marker check (only meaningful while a session is active) ──
            # When test_mobinet() finishes one pass, it prints
            #   "========== TEST_MOBINET_PASS_END =========="
            # We finalize the current session immediately on seeing this — no
            # need to wait for the next session's start banner.
            if session_start is not None:
                end_from = session_start + cur_marker_len
                end_idx_in_tail = bytes(raw[end_from:]).lower().find(END_MARKER)
                if end_idx_in_tail >= 0:
                    end_pos = end_from + end_idx_in_tail + len(END_MARKER)
                    # extend through the rest of the END marker's line so the
                    # processed text ends cleanly at a newline
                    nl = raw.find(b"\n", end_pos)
                    if nl >= 0:
                        end_pos = nl + 1
                    session_bytes = bytes(raw[session_start:end_pos])
                    print(f"\n{'═'*64}")
                    print(f"  SESSION {sessions_seen}  COMPLETED (END marker) — "
                          f"{len(session_bytes):,} bytes — processing…")
                    print(f"{'═'*64}", flush=True)
                    text = session_bytes.decode("utf-8", errors="replace")
                    process_session(text, sessions_seen)
                    # discard processed bytes; wait for next session
                    raw            = bytearray(raw[end_pos:])
                    session_start  = None
                    cur_marker_len = 0
                    last_heartbeat = 0
                    continue

            # Look for the NEXT session marker.  If we're already inside a
            # session, skip past the current marker so we don't re-find it.
            search_from = (session_start + cur_marker_len
                           if session_start is not None else 0)
            if search_from >= len(raw):
                continue
            marker_pos, marker_len = _find_next_marker(raw, search_from)
            if marker_pos < 0:
                continue

            if session_start is None:
                # First session of the run starts here.
                sessions_seen  += 1
                session_start   = marker_pos
                cur_marker_len  = marker_len
                print(f"\n{'═'*64}")
                print(f"  SESSION {sessions_seen}  STARTED  (marker @ {marker_pos})")
                print(f"{'═'*64}", flush=True)
            else:
                # Previous session is complete — bytes [session_start, marker_pos).
                session_bytes = bytes(raw[session_start:marker_pos])
                print(f"\n{'═'*64}")
                print(f"  SESSION {sessions_seen}  CAPTURED — "
                      f"{len(session_bytes):,} bytes — processing…")
                print(f"{'═'*64}", flush=True)
                text = session_bytes.decode("utf-8", errors="replace")
                process_session(text, sessions_seen)

                # Trim raw buffer: discard everything before the new marker.
                raw            = bytearray(raw[marker_pos:])
                session_start  = 0
                cur_marker_len = marker_len
                last_heartbeat = 0
                sessions_seen += 1
                print(f"\n{'═'*64}")
                print(f"  SESSION {sessions_seen}  STARTED")
                print(f"{'═'*64}", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopping — flushing pending session if any…")
        if session_start is not None and len(raw) > session_start:
            session_bytes = bytes(raw[session_start:])
            print(f"  → SESSION {sessions_seen}: {len(session_bytes):,} bytes "
                  "captured at Ctrl+C — processing…")
            text = session_bytes.decode("utf-8", errors="replace")
            process_session(text, sessions_seen)
        elif len(raw) > 0:
            # No marker was ever detected (e.g. user started the script after the
            # banner had already gone past, or the FPGA's banner didn't match any
            # known pattern).  Treat the whole capture as one anonymous session
            # so the data isn't lost.
            print(f"  → no session marker seen, but {len(raw):,} bytes captured — "
                  "processing as anonymous session…")
            text = bytes(raw).decode("utf-8", errors="replace")
            process_session(text, 0)
        else:
            print("  (nothing captured)")
        print("Done.")
    finally:
        stop_event.set()
        try:
            ser.close()
        except Exception:
            pass
        reader.join(timeout=1.0)


if __name__ == "__main__":
    main()
