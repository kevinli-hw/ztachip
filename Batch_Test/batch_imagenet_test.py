"""
Batch ImageNet accuracy test for ztachip MobileNetV2 (Linux only).

Serial reception strategy
-------------------------
We do NOT use pyserial.read() — it has been observed to drop bytes at high
baud rates. Instead we shell out to the same `sed` pipeline you've been
using manually:

    sed '/TEST_MOBINET_PASS_END/q' /dev/ttyUSBx > capture_NNN.bin

`sed` reads from the tty directly via the kernel and exits when the marker
is seen. Python only orchestrates start/stop, file naming, and parsing.

Persistent GDB + breakpoint-synchronized flow
---------------------------------------------
ONE GDB process drives the whole batch via a PTY. All setup (file/target/
reset/load/break/continue-to-first-break) happens once at startup. A break
at the start of test() bounds every inference cycle so BMP swaps land at a
clean point — no race against test() reading classifier_input.

  Startup (one-time, in GdbSession.__init__):
      set pagination off / set confirm off
      file <elf>
      target extended-remote localhost:<port>
      monitor reset halt
      load
      break test
      continue                ← runs crt0 + main(), halts at first test()

  Per image (in the main loop):
      restore <bmp> binary &classifier_input
      flush serial buffer
      start sed
      continue                ← runs test() (UART output, sed captures),
                                returns to while(1), calls test() again,
                                hits break → CPU halted, GDB returns
      wait sed (already exited)

  Shutdown:
      quit

Synchronization with GDB
------------------------
GDB runs in a PTY (so its stdout is line-buffered). Each command is
followed by an `echo \\n<MARKER>\\n` and we read GDB's output until
<MARKER> appears. For blocking commands like `continue`, the echo only
runs after the breakpoint is hit, so we naturally block until inference
is done.

Preprocessing
-------------
Direct resize to 224x224 (matches the keras-style load_img(target_size=...)
flow used in PC reference scripts). NO -128, NO normalization in Python.

Why no -128:
  The firmware feeds the raw uint8 RGB tensor straight into the TFLite
  engine. The engine looks up the input tensor's quantization params from
  the model file — for mobilenet_v2_1.0_224_quant these are
      scale = 0.0078125,  zero_point = 128
  and the first conv layer internally computes
      real_value = (uint8 - 128) * scale
  So the -128 IS happening, just inside the model, not in our pixel buffer.

  If you swap to an int8-input model (zero_point=0), then yes, you'd need
  -128 somewhere — but that change belongs in the firmware's CreateWithBitmap
  or in the tflite engine, not in this Python script. Verify your model
  with:
      tf.lite.Interpreter(model).get_input_details()[0]
"""

import argparse
import csv
import io
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
import numpy as np
from pathlib import Path

from PIL import Image

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_PORT      = "/dev/ttyUSB1"
DEFAULT_BAUD      = 4_000_000
DEFAULT_OPENOCD   = 3333
#INFERENCE_TIMEOUT = 90        # seconds for one inference's UART output
INFERENCE_TIMEOUT = 240        # seconds for one inference's UART output
GDB_TIMEOUT       = 120

#PROJECT_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = Path.home() / "workdir" / "ztachip"
ELF_DEFAULT  = PROJECT_ROOT / "SW" / "build" / "ztachip.elf"
END_MARKER   = "TEST_MODEL_PASS_END"


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_imagenet(img_path: str) -> bytes:
    """
    Direct resize to 224x224, matching what the keras
    `tf.keras.preprocessing.image.load_img(target_size=(224,224))` flow does
    in your PC reference. Pixel data is written as standard uint8 BMP; the
    firmware's CreateWithBitmap converts to int8 (subtract 128) when called
    with TensorDataTypeInt8.

    Output is exactly 150582 bytes (14 + 40 + 224*672), matching
    sizeof(classifier_input[]).
    """
    #img = Image.open(img_path).convert("RGB").resize(
    #    (224, 224), Image.NEAREST
    #)

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    resize_shorter_side = 256
    input_h = 224
    input_w = 224

    if h < w:
        new_h = resize_shorter_side
        new_w = int(round(w * resize_shorter_side / h))
    else:
        new_w = resize_shorter_side
        new_h = int(round(h * resize_shorter_side / w))

    img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    left = (new_w - input_w) // 2
    top = (new_h - input_h) // 2
    right = left + input_w
    bottom = top + input_h

    img = img.crop((left, top, right, bottom))

    input_scale = 0.007843137718737125
    input_zero_point = -1
    x = np.asarray(img).astype(np.float32)
    x = x / 127.5 - 1.0
    q = np.round(x / input_scale + input_zero_point)
    q = np.clip(q, -128, 127).astype(np.int8)

    #buf = io.BytesIO()
    #img.save(buf, format="BMP")
    #return buf.getvalue()
    arr_q = np.transpose(q, (2, 0, 1))   # HWC -> CHW
    raw = arr_q.tobytes()

    return raw


# ─── Serial helpers (no pyserial.read) ────────────────────────────────────────

def stty_configure(port: str, baud: int) -> None:
    """Equivalent to:  stty -F <port> <baud> raw -echo"""
    subprocess.run(
        ["stty", "-F", port, str(baud), "raw", "-echo"],
        check=True,
    )


def flush_serial_input(port: str) -> None:
    """
    Discard any bytes already queued in the OS input buffer for `port`.
    Uses termios.tcflush — does not read into Python, so cannot drop bytes.
    Equivalent to the kernel-level tcflush(TCIFLUSH).
    """
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        termios.tcflush(fd, termios.TCIFLUSH)
    finally:
        os.close(fd)


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
    Long-running GDB process for the entire batch.

    Lifecycle
    ---------
      __init__:  spawn GDB inside a PTY (so its stdout is line-buffered),
                 then do all one-time setup:
                     set pagination off / set confirm off
                     file <elf>
                     target extended-remote localhost:<port>
                     monitor reset halt
                     load
                     break <bp_symbol>
                     continue            ← blocks until first test() break hits
                 → returns with CPU halted at first test() entry
      restore(bmp_path): send `restore <bmp_path> binary &classifier_input`
      cont(timeout):     send `continue`, block until next break is hit
      close():           send `quit`, reap process

    Synchronization
    ---------------
    GDB executes commands sequentially. Each command we send is followed by
    `echo \\n<MARKER>\\n` where MARKER is unique per call. We read GDB's stdout
    until MARKER appears — that means GDB finished the previous command. For a
    blocking command like `continue`, the echo only runs after the breakpoint
    is hit, so we naturally block until inference is done.

    Why a PTY
    ---------
    With plain pipes, GDB block-buffers its stdout (4 KB at a time), so we'd
    sit waiting for output forever. A PTY makes GDB think it's interactive,
    and it line-buffers as expected.
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

    def restore(self, bmp_path: str, timeout: float = 30) -> str:
        return self._cmd(f"restore {bmp_path} binary &ILSVRC2012_val_00000001_raw",
                          timeout=timeout)
        #return self._cmd(f"restore {bmp_path} binary &classifier_input",
        #                  timeout=timeout)

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
    rb"CreateWithBitmap\s*:\s*([0-9.]+)\s*s.*?"
    rb"Model Parse\s*:\s*([0-9.]+)\s*s.*?"
    rb"Inference\s*:\s*([0-9.]+)\s*s.*?"
    rb"Total\s*:\s*([0-9.]+)\s*s",
    re.DOTALL
)

TIMING_FIELDS = ["create_bitmap", "model_parse", "inference", "total"]

def parse_timing(path: Path):
    m = _TIMING_RE.search(path.read_bytes())
    if not m:
        return None
    return {
        "create_bitmap": float(m.group(1)),
        "model_parse":   float(m.group(2)),
        "inference":     float(m.group(3)),
        "total":         float(m.group(4)),
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
                         "between iterations for clean BMP swap (default: %(default)s)")
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
                             "fw_bitmap_s", "fw_parse_s",
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

                # 1. preprocess
                try:
                    bmp = preprocess_imagenet(str(p))
                except Exception as e:
                    print(f"IMG_ERR {e}")
                    continue

                # 2. write BMP to a temp file (GDB `restore` reads from disk)
                #with tempfile.NamedTemporaryFile(suffix=".bmp",
                #                                  delete=False) as bf:
                with tempfile.NamedTemporaryFile(suffix=".bin",
                                                  delete=False) as bf:
                    bf.write(bmp)
                    bmp_path = bf.name
                t_restore = t_infer = 0.0
                try:
                    # 3. restore via the persistent GDB session
                    #    (CPU is halted at break test() from the previous
                    #    iteration's `continue`)
                    try:
                        t0 = time.monotonic()
                        gdb.restore(bmp_path, timeout=30)
                        t_restore = time.monotonic() - t0
                    except (GdbTimeout, RuntimeError) as e:
                        # GDB session is in an undefined state — abort batch
                        print(f"GDB_RESTORE_FAIL — aborting batch: {e}")
                        break

                    # 4. drain serial buffer (kernel-level)
                    #try:
                    #    flush_serial_input(args.port)
                    #except Exception as e:
                    #    print(f"FLUSH_ERR {e}")
                    #    continue

                    # 5. start sed BEFORE we let the CPU run
                    cap_path = cap_dir / f"capture_{gidx:06d}.bin"
                    sed_proc = start_sed_capture(args.port, END_MARKER,
                                                  cap_path)
                    time.sleep(args.sed_startup_delay)

                    # 6. continue → CPU runs one test() iteration, prints
                    #    output (sed captures), returns, hits break at next
                    #    iteration → GDB's `continue` returns
                    try:
                        t0 = time.monotonic()
                        gdb.cont(timeout=args.inference_timeout + 30)
                        t_infer = time.monotonic() - t0
                    except (GdbTimeout, RuntimeError) as e:
                        sed_proc.kill()
                        print(f"GDB_CONT_FAIL — aborting batch: {e}")
                        break

                    # 7. sed should already have exited (PASS_END came before
                    #    the next iteration's break)
                    if not wait_sed(sed_proc, 10):
                        print(f"SED_TIMEOUT (cap={cap_path.name})")
                        continue

                    # accumulate timing stats only on a successful inference
                    sum_t_restore += t_restore
                    sum_t_infer   += t_infer
                    n_timed       += 1
                finally:
                    try:
                        os.unlink(bmp_path)
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
                    t_str += (f"  fw[bitmap={timing['create_bitmap']:.3f}"
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
        labels = {"create_bitmap": "CreateWithBitmap",
                  "model_parse":   "Model Parse",
                  "inference":     "Inference",
                  "total":         "Total"}
        print(f"  {'Phase':<20s} {'Avg':>10s} {'Min':>10s} {'Max':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        for k in TIMING_FIELDS:
            avg_s = sum_fw[k] / n_fw
            min_s = min_fw[k]
            max_s = max_fw[k]
            print(f"  {labels[k]:<20s} {avg_s:>10.4f} {min_s:>10.4f} {max_s:>10.4f}")


if __name__ == "__main__":
    main()
