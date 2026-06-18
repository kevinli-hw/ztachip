"""
Batch ImageNet accuracy test for ztachip ResNet-18 (Linux only).

One persistent GDB session loads ztachip.elf once, then for each image we
`restore` the preprocessed input into the firmware input array and `continue`
for a single inference. UART output is captured with `sed` (more reliable than
pyserial at high baud, which drops bytes):

    sed '/TEST_MODEL_PASS_END/q' <port> > capture_NNN.bin

Per-image loop (CPU halted at `break test` between inferences):
    restore <input.bin> binary &ILSVRC2012_val_00000001_raw
    start sed
    continue        # one test() iteration runs, sed captures, halts at next break

The firmware's test_resnet18() does fopen("ILSVRC2012_val_00000001_raw.bin"),
which the fake filesystem in SW/base/simplelib.c maps to the C array
`ILSVRC2012_val_00000001_raw`. We restore each image over that array while the
CPU is halted, so the following fread() reads the fresh bytes. The blob must be
exactly 3*224*224 = 150528 bytes.

GDB runs in a PTY so its stdout is line-buffered. Each command is followed by
`echo \\n<MARKER>\\n`; we read until <MARKER> appears, so a blocking `continue`
returns only after the breakpoint is hit.

Preprocessing (torch-style, see preprocess_imagenet): aspect-preserving resize
of the shorter side to 256, center crop to 224x224, scale to [0, 1], normalize
with ImageNet mean/std, quantize to int8 with the model's input scale/zero_point,
reorder HWC->CHW. Returns 150528 raw bytes.

Labels: ResNet-18 outputs 1000 classes (ids 0..999, no background). The manifest
label ids must use the same 0..999 indexing (a 1001-class MobileNet manifest with
background at id 0 is off by one).
"""

import argparse
import csv
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import tempfile
import termios
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_PORT      = "/dev/ttyUSB1"
DEFAULT_BAUD      = 4_000_000
DEFAULT_OPENOCD   = 3333
INFERENCE_TIMEOUT = 240        # seconds for one inference's UART output
GDB_TIMEOUT       = 120

PROJECT_ROOT = Path.home() / "workdir" / "ztachip"
ELF_DEFAULT  = PROJECT_ROOT / "SW" / "build" / "ztachip.elf"
END_MARKER   = "TEST_MODEL_PASS_END"

# Symbol of the array backing fopen("ILSVRC2012_val_00000001_raw.bin") in the
# firmware's fake filesystem (see SW/base/simplelib.c) — `restore` target.
RESTORE_SYMBOL = "ILSVRC2012_val_00000001_raw"

INPUT_W = 224
INPUT_H = 224
INPUT_BYTES = 3 * INPUT_H * INPUT_W   # 150528, must equal sizeof(array)

INPUT_SCALE      = 0.01865844801068306
INPUT_ZERO_POINT = -14


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_imagenet(img_path: str) -> bytes:
    """
    Torch-style ResNet preprocessing: aspect-preserving resize of the shorter
    side to 256 (bilinear), center crop to 224x224, scale to [0, 1], normalize
    with ImageNet mean/std, quantize to int8 with the model's input
    scale/zero_point, reorder HWC -> CHW. Returns INPUT_BYTES (150528) bytes.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    resize_shorter_side = 256

    if h < w:
        new_h = resize_shorter_side
        new_w = int(round(w * resize_shorter_side / h))
    else:
        new_w = resize_shorter_side
        new_h = int(round(h * resize_shorter_side / w))

    img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    left = (new_w - INPUT_W) // 2
    top = (new_h - INPUT_H) // 2
    img = img.crop((left, top, left + INPUT_W, top + INPUT_H))

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = np.asarray(img).astype(np.float32) / 255.0   # HWC, RGB, [0, 1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD          # normalize

    arr_q = np.round(arr / INPUT_SCALE + INPUT_ZERO_POINT)
    arr_q = np.clip(arr_q, -128, 127).astype(np.int8)
    arr_q = np.transpose(arr_q, (2, 0, 1))   # HWC -> CHW
    raw = arr_q.tobytes()
    assert len(raw) == INPUT_BYTES, f"bad raw size {len(raw)}"
    return raw


# ─── Serial helpers (no pyserial.read) ────────────────────────────────────────

def stty_configure(port: str, baud: int) -> None:
    """Equivalent to:  stty -F <port> <baud> raw -echo"""
    subprocess.run(
        ["stty", "-F", port, str(baud), "raw", "-echo"],
        check=True,
    )


def start_sed_capture(port: str, marker: str, out_path: Path) -> subprocess.Popen:
    """
    Launch:  sed '/<marker>/q' <port> > <out_path>
    Returns the Popen handle. Caller waits or kills it.
    """
    out_f = open(out_path, "wb")
    return subprocess.Popen(
        ["sed", f"/{marker}/q", port],
        stdout=out_f,
        stderr=subprocess.DEVNULL,
    )


def wait_sed(proc: subprocess.Popen, timeout: float) -> bool:
    """
    Wait for sed to exit (it does so the moment it sees the marker).
    Returns True on clean exit, False on timeout (process is killed).
    """
    try:
        proc.wait(timeout=timeout)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        return False


# ─── Persistent GDB session ───────────────────────────────────────────────────

class GdbTimeout(Exception):
    pass


class GdbSession:
    """
    Long-running GDB process driving the whole batch over a PTY.

      __init__:          one-time setup — file/target/reset/load/break, then
                         continue to halt at the first test() entry.
      restore(raw_path): restore <raw_path> binary &ILSVRC2012_val_00000001_raw
      cont(timeout):     continue, block until the next break is hit
      close():           quit and reap the process

    Each command is followed by `echo \\n<MARKER>\\n`; reading until MARKER
    means the command finished. A PTY (not a pipe) keeps GDB's stdout
    line-buffered so we don't block on its 4 KB pipe buffer.
    """

    def __init__(self, gdb_path: str, elf: str, openocd_port: int,
                  bp_symbol: str = "test", load_timeout: float = 180,
                  log_path: str = ""):
        # PTY pair — child end becomes GDB's stdin/stdout/stderr.
        parent_fd, child_fd = pty.openpty()
        # Disable terminal echo on the child so we don't read back what we wrote.
        attrs = termios.tcgetattr(child_fd)
        attrs[3] &= ~(termios.ECHO | termios.ECHOCTL | termios.ECHONL)
        termios.tcsetattr(child_fd, termios.TCSANOW, attrs)

        # NOTE: do NOT pass --nx. The user's interactive GDB sessions read
        # .gdbinit which may contain target-specific settings (RISC-V arch,
        # breakpoint behavior) that affect whether `continue` correctly
        # halts at our software breakpoint. Skipping .gdbinit caused the
        # script to hang on `continue` while manual GDB worked.
        self.proc = subprocess.Popen(
            [gdb_path, "-q"],
            stdin=child_fd, stdout=child_fd, stderr=child_fd,
            preexec_fn=os.setsid,
            close_fds=True,
        )
        os.close(child_fd)
        self._fd = parent_fd
        self._buf = ""
        self._counter = 0
        self._closed = False
        # Optional full-trace log of all GDB I/O for debugging
        self._log = open(log_path, "w") if log_path else None

        # Drain banner
        self._drain(0.4)

        try:
            self._cmd("set pagination off")
            self._cmd("set confirm off")
            self._cmd("set remote memory-write-packet-size 16384")
            self._cmd("set remote memory-write-packet-size fixed")
            self._cmd(f"file {elf}", timeout=30)
            self._cmd(f"target extended-remote localhost:{openocd_port}",
                      timeout=30)
            self._cmd("monitor reset halt", timeout=30)

            # `load` writes the ELF over JTAG; this can take 10s–several minutes
            # depending on JTAG speed and ELF size. GDB emits a final summary
            # line like:
            #   Start address 0x00000000, load size 4880760
            # which we parse out and report to the user.
            print("  load: writing ELF to target (may take a while)…",
                  flush=True)
            t0 = time.monotonic()
            out = self._cmd("load", timeout=load_timeout)
            dt = time.monotonic() - t0
            m = re.search(r"Start address\s+(\S+),\s*load size\s+(\d+)", out)
            if m:
                addr = m.group(1).rstrip(",")
                size = int(m.group(2))
                rate = size / dt / 1024 if dt > 0 else 0
                print(f"  load: done in {dt:.1f}s — start={addr} "
                      f"size={size:,} bytes ({rate:.1f} KiB/s)",
                      flush=True)
            else:
                print(f"  load: done in {dt:.1f}s "
                      f"(no 'Start address' line in output — check ELF)",
                      flush=True)

            # JTAG settling pause: writing the breakpoint (`break test`) right
            # after `load` returns can race with OpenOCD/JTAG still finishing
            # post-load housekeeping; if the ebreak instruction lands wrong,
            # CPU will never halt at the break and `continue` runs forever.
            # 0.2s is plenty in practice and doesn't add per-batch cost since
            # this only runs once at startup.
            time.sleep(0.2)
            self._cmd(f"break {bp_symbol}", timeout=10)
            # Run through crt0 + main() until first test() breakpoint hit
            print("  initial run: crt0 + main() → first test() break…",
                  flush=True)
            self._cmd("continue", timeout=load_timeout)
        except Exception:
            self.close()
            raise

    # ── low-level IO ──────────────────────────────────────────────────────────

    def _send(self, s: str) -> None:
        os.write(self._fd, s.encode("utf-8"))
        if self._log:
            self._log.write(f"<<< SEND: {s!r}\n")
            self._log.flush()

    def _read_chunk(self, timeout: float) -> str:
        r, _, _ = select.select([self._fd], [], [], timeout)
        if not r:
            return ""
        try:
            chunk = os.read(self._fd, 4096)
        except OSError:
            return ""
        text = chunk.decode("utf-8", errors="replace")
        if self._log and text:
            self._log.write(f">>> RECV: {text!r}\n")
            self._log.flush()
        return text

    def _drain(self, idle_timeout: float) -> str:
        """Read everything available; stop after `idle_timeout` of silence."""
        deadline = time.monotonic() + idle_timeout
        out = ""
        while time.monotonic() < deadline:
            chunk = self._read_chunk(0.05)
            if chunk:
                out += chunk
                deadline = time.monotonic() + idle_timeout
        return out

    def _read_until(self, marker: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        out = self._buf
        self._buf = ""
        while True:
            idx = out.find(marker)
            if idx >= 0:
                self._buf = out[idx + len(marker):]
                return out[:idx]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GdbTimeout(f"timed out after {timeout}s waiting for "
                                  f"GDB; tail:\n{out[-800:]}")
            chunk = self._read_chunk(min(remaining, 1.0))
            if not chunk:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"GDB exited; tail:\n{out[-800:]}")
                continue
            out += chunk

    def _cmd(self, command: str, timeout: float = 15) -> str:
        self._counter += 1
        marker = f"___GDB_DONE_{self._counter}___"
        self._send(f"{command}\n")
        # `echo` interprets \\n as a newline; the literal backslash-n stays in
        # the python string so GDB sees `echo \n___GDB_DONE_N___\n`.
        self._send(f"echo \\n{marker}\\n\n")
        return self._read_until(marker, timeout)

    # ── public API ────────────────────────────────────────────────────────────

    def restore(self, raw_path: str, timeout: float = 30) -> str:
        return self._cmd(f"restore {raw_path} binary &{RESTORE_SYMBOL}",
                          timeout=timeout)

    def cont(self, timeout: float) -> str:
        return self._cmd("continue", timeout=timeout)

    def close(self, timeout: float = 5) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._send("quit\n")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        if self._log:
            try:
                self._log.close()
            except Exception:
                pass


# ─── Top-1/top-5 parser (slim, drops per-layer logs) ──────────────────────────

_RESULT_RE = re.compile(
    rb"RESULT top1=(\d+) top2=(\d+) top3=(\d+) top4=(\d+) top5=(\d+)"
)

def parse_top5(path: Path):
    m = _RESULT_RE.search(path.read_bytes())
    if not m:
        return None
    return int(m.group(1)), [int(m.group(i)) for i in range(1, 6)]


_TIMING_RE = re.compile(
    rb"Load Input\s*:\s*([0-9.]+)\s*s.*?"
    rb"Model Parse\s*:\s*([0-9.]+)\s*s.*?"
    rb"Inference\s*:\s*([0-9.]+)\s*s.*?"
    rb"Total\s*:\s*([0-9.]+)\s*s",
    re.DOTALL
)

TIMING_FIELDS = ["load_input", "model_parse", "inference", "total"]

def parse_timing(path: Path):
    m = _TIMING_RE.search(path.read_bytes())
    if not m:
        return None
    return {
        "load_input":  float(m.group(1)),
        "model_parse": float(m.group(2)),
        "inference":   float(m.group(3)),
        "total":       float(m.group(4)),
    }


# ─── Manifest ────────────────────────────────────────────────────────────────

def load_manifest(path: str):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                pairs.append((parts[0], int(parts[1])))
    return pairs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default="",
                    help="Full manifest: 'filename label_id' per line "
                         "(used with --start/--count)")
    ap.add_argument("--test-list", default="",
                    help="Test list file: 'filename label_id' per line; "
                         "when given, overrides --labels/--start/--count")
    ap.add_argument("--output", default="results.csv")
    ap.add_argument("--port",   default=DEFAULT_PORT)
    ap.add_argument("--baud",   type=int, default=DEFAULT_BAUD)
    ap.add_argument("--gdb",    default="riscv32-unknown-elf-gdb")
    ap.add_argument("--elf",    default=str(ELF_DEFAULT))
    ap.add_argument("--openocd-port", type=int, default=DEFAULT_OPENOCD)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--keep-captures", action="store_true",
                    help="don't delete capture_NNN.bin after parsing")
    ap.add_argument("--captures-dir", default="captures")
    ap.add_argument("--no-stty", action="store_true",
                    help="skip stty configuration (assume already configured)")
    ap.add_argument("--sed-startup-delay", type=float, default=0.1,
                    help="seconds to let sed open the device before resuming CPU")
    ap.add_argument("--inference-timeout", type=float, default=INFERENCE_TIMEOUT,
                    help="seconds for one inference (Phase B continue) "
                         "(default: %(default)s)")
    ap.add_argument("--break-symbol", default="test",
                    help="function name to set breakpoint at — CPU halts here "
                         "between iterations for clean input swap (default: %(default)s)")
    ap.add_argument("--gdb-log", default="",
                    help="if set, log every byte sent to / received from GDB "
                         "to this file (for debugging communication issues)")
    ap.add_argument("--parser", choices=["top5", "external", "both"],
                    default="top5",
                    help="how to parse capture file: "
                         "'top5' = built-in (fast, only RESULT line); "
                         "'external' = invoke --external-parser only; "
                         "'both' = run external parser AND extract top5 "
                         "(recommended if you want layer-by-layer analysis)")
    ap.add_argument("--external-parser", default="",
                    help="path to your existing parse_capture.py "
                         "(invoked as: python <parser> <capture_file>)")
    args = ap.parse_args()

    if not Path(args.elf).exists():
        sys.exit(f"ELF not found: {args.elf}")
    if not shutil.which("sed"):
        sys.exit("sed not found in PATH")
    if not shutil.which(args.gdb):
        sys.exit(f"GDB not found: {args.gdb}")
    if args.parser in ("external", "both") and not args.external_parser:
        sys.exit(f"--parser={args.parser} requires --external-parser <path>")
    if args.external_parser and not Path(args.external_parser).exists():
        sys.exit(f"External parser not found: {args.external_parser}")

    # Configure tty once
    if not args.no_stty:
        stty_configure(args.port, args.baud)
        print(f"# stty -F {args.port} {args.baud} raw -echo")

    if args.test_list:
        if not Path(args.test_list).exists():
            sys.exit(f"Test list not found: {args.test_list}")
        manifest = load_manifest(args.test_list)
        print(f"Test list: {args.test_list} ({len(manifest)} images)")
    elif args.labels:
        if not Path(args.labels).exists():
            sys.exit(f"Labels file not found: {args.labels}")
        manifest = load_manifest(args.labels)
        if args.start: manifest = manifest[args.start:]
        if args.count: manifest = manifest[: args.count]
    else:
        sys.exit("must specify --labels or --test-list")

    cap_dir = Path(args.captures_dir)
    cap_dir.mkdir(exist_ok=True)

    print(f"Images : {len(manifest)}")
    print(f"Port   : {args.port} @ {args.baud}")
    print(f"ELF    : {args.elf}")
    print("-" * 78)

    # Open the UART port BEFORE starting GDB, and hold it for the whole batch.
    # On boards that share the same USB chip between JTAG and UART (e.g.
    # FT2232HQ on ArtyA7: channel A = JTAG, channel B = UART), opening
    # channel B mid-flight can briefly disturb channel A and cause a JTAG
    # `continue` packet to be dropped — CPU then doesn't actually resume even
    # though OpenOCD reports "running". By opening the UART side first and
    # never closing it during the run, the USB state is stable when GDB then
    # opens JTAG, and per-image sed `open()` calls on the same port don't
    # trigger any USB-level state change.
    try:
        port_guard = os.open(args.port, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError as e:
        sys.exit(f"failed to open guard fd on {args.port}: {e}")
    print(f"Held guard fd on {args.port} (stabilizes USB before GDB starts)")

    # Spawn the persistent GDB session ONCE. Inside __init__ it does the full
    # one-time setup: file/target/reset/load/break/continue → CPU is halted at
    # the first test() entry, ready for the per-image restore + continue loop.
    print("Starting GDB (one-time load + initial run to break test())…",
          flush=True)
    try:
        gdb = GdbSession(
            args.gdb, args.elf, args.openocd_port,
            bp_symbol=args.break_symbol,
            load_timeout=GDB_TIMEOUT,
            log_path=args.gdb_log,
        )
    except Exception as e:
        sys.exit(f"GDB session setup failed: {e}")
    print("GDB ready — CPU halted at first test() break.")
    print("-" * 78)

    total = c1 = c5 = 0
    sum_t_restore = sum_t_infer = 0.0
    n_timed = 0
    sum_fw = {k: 0 for k in TIMING_FIELDS}
    min_fw = {k: float('inf') for k in TIMING_FIELDS}
    max_fw = {k: 0 for k in TIMING_FIELDS}
    n_fw = 0
    open_mode = "a" if args.start > 0 and Path(args.output).exists() else "w"

    try:
        with open(args.output, open_mode, newline="") as cf:
            w = csv.writer(cf)
            if open_mode == "w":
                w.writerow(["idx", "image", "gt", "pred1",
                             "top5", "ok1", "ok5", "bytes",
                             "fw_load_s", "fw_parse_s",
                             "fw_infer_s", "fw_total_s"])

            for i, (fname, gt) in enumerate(manifest):
                gidx = args.start + i
                p = Path(args.images) / fname
                tag = f"[{gidx+1:>6}]"

                if not p.exists():
                    print(f"{tag} SKIP missing {fname}")
                    continue
                print(f"{tag} {fname:<45} gt={gt:>4}",
                      end="  ", flush=True)

                # 1. preprocess (resize/crop/normalize/quantize/CHW -> raw int8)
                try:
                    raw = preprocess_imagenet(str(p))
                except Exception as e:
                    print(f"IMG_ERR {e}")
                    continue

                # 2. write the raw input to a temp file (GDB restore reads from disk)
                with tempfile.NamedTemporaryFile(suffix=".bin",
                                                  delete=False) as bf:
                    bf.write(raw)
                    raw_path = bf.name
                t_restore = t_infer = 0.0
                try:
                    # 3. restore into the firmware input array (CPU halted at break test)
                    try:
                        t0 = time.monotonic()
                        gdb.restore(raw_path, timeout=30)
                        t_restore = time.monotonic() - t0
                    except (GdbTimeout, RuntimeError) as e:
                        # GDB session is in an undefined state — abort batch
                        print(f"GDB_RESTORE_FAIL — aborting batch: {e}")
                        break

                    # 4. start sed BEFORE we let the CPU run
                    cap_path = cap_dir / f"capture_{gidx:06d}.bin"
                    sed_proc = start_sed_capture(args.port, END_MARKER,
                                                  cap_path)
                    time.sleep(args.sed_startup_delay)

                    # 5. continue → one test() iteration runs, sed captures its
                    #    UART output, CPU halts at the next break → continue returns
                    try:
                        t0 = time.monotonic()
                        gdb.cont(timeout=args.inference_timeout + 30)
                        t_infer = time.monotonic() - t0
                    except (GdbTimeout, RuntimeError) as e:
                        sed_proc.kill()
                        print(f"GDB_CONT_FAIL — aborting batch: {e}")
                        break

                    # 6. sed should already have exited on the PASS_END marker
                    if not wait_sed(sed_proc, 10):
                        print(f"SED_TIMEOUT (cap={cap_path.name})")
                        continue

                    # accumulate timing stats only on a successful inference
                    sum_t_restore += t_restore
                    sum_t_infer   += t_infer
                    n_timed       += 1
                finally:
                    try:
                        os.unlink(raw_path)
                    except OSError:
                        pass

                n = cap_path.stat().st_size
                timing = parse_timing(cap_path)
                if timing:
                    n_fw += 1
                    for k in TIMING_FIELDS:
                        sum_fw[k] += timing[k]
                        min_fw[k] = min(min_fw[k], timing[k])
                        max_fw[k] = max(max_fw[k], timing[k])
                fw_cols = ([timing[k] for k in TIMING_FIELDS]
                           if timing else ["", "", "", ""])

                # 8a. invoke external parser (if requested)
                if args.parser in ("external", "both"):
                    try:
                        subprocess.run(
                            [sys.executable, args.external_parser,
                             str(cap_path)],
                            check=False, timeout=60,
                        )
                    except Exception as e:
                        sys.stderr.write(f"  external parser err: {e}\n")

                t_str = f"[restore={t_restore:.2f}s infer={t_infer:.1f}s]"
                if timing:
                    t_str += (f"  fw[load={timing['load_input']:.3f}"
                              f" parse={timing['model_parse']:.3f}"
                              f" infer={timing['inference']:.3f}"
                              f" total={timing['total']:.3f}]s")

                # 8b. extract top-1/top-5 from RESULT line for accuracy stats
                if args.parser == "external":
                    w.writerow([gidx, fname, gt, "", "", "", "", n] + fw_cols)
                    cf.flush()
                    print(f"({n}B  external parser invoked)  {t_str}")
                    if not args.keep_captures:
                        cap_path.unlink(missing_ok=True)
                    continue

                parsed = parse_top5(cap_path)
                if parsed is None:
                    print(f"NO_RESULT ({n}B)  {t_str}")
                    continue
                pred, top5 = parsed
                ok1 = int(pred == gt)
                ok5 = int(gt in top5)
                total += 1
                c1 += ok1
                c5 += ok5

                tick = "OK  " if ok1 else "FAIL"
                print(f"pred={pred:>4}  {tick}  "
                      f"Top1={c1/total*100:5.1f}%  "
                      f"Top5={c5/total*100:5.1f}%  ({n}B)  {t_str}")

                w.writerow([gidx, fname, gt, pred, top5, ok1, ok5, n] + fw_cols)
                cf.flush()

                if not args.keep_captures:
                    cap_path.unlink(missing_ok=True)
    finally:
        gdb.close()
        try:
            os.close(port_guard)
        except OSError:
            pass

    if total:
        print("-" * 78)
        print(f"DONE  {total} images  "
              f"Top1={c1/total:.4f} ({c1}/{total})  "
              f"Top5={c5/total:.4f} ({c5}/{total})")
    if n_timed:
        avg_r = sum_t_restore / n_timed
        avg_i = sum_t_infer / n_timed
        print(f"      avg per-image: restore={avg_r:.2f}s  "
              f"infer={avg_i:.1f}s  total={avg_r + avg_i:.1f}s  "
              f"({n_timed} images)")
    if n_fw:
        print(f"\n{'=' * 78}")
        print(f"Firmware Timing Summary ({n_fw} images, values in seconds)")
        print(f"{'=' * 78}")
        labels = {"load_input":  "Load Input",
                  "model_parse": "Model Parse",
                  "inference":   "Inference",
                  "total":       "Total"}
        print(f"  {'Phase':<20s} {'Avg':>10s} {'Min':>10s} {'Max':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        for k in TIMING_FIELDS:
            avg_s = sum_fw[k] / n_fw
            min_s = min_fw[k]
            max_s = max_fw[k]
            print(f"  {labels[k]:<20s} {avg_s:>10.4f} {min_s:>10.4f} {max_s:>10.4f}")


if __name__ == "__main__":
    main()
