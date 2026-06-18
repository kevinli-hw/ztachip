"""
Batch MNIST accuracy test for ztachip LeNet (Linux only).

Same GDB + sed architecture as batch_imagenet_test.py — see that file for
details on the serial/GDB synchronization strategy.

Preprocessing
-------------
MNIST test images are 28x28 grayscale PNGs organized as:
    test_images/0/test_00000.png
    test_images/1/test_00001.png
    ...
Label file format:  <subfolder/filename> <label>
    7/test_00000.png 7
    2/test_00001.png 2

Each image is converted to grayscale, quantized to int8 using the model's
quantization parameters (scale ≈ 1/255, zero_point = -128), and saved as
a raw 784-byte bin. GDB restores this directly to &classifier_input.
The firmware reads the pre-quantized int8 data with no further processing.
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
INFERENCE_TIMEOUT = 120
GDB_TIMEOUT       = 120

PROJECT_ROOT = Path.home() / "workdir" / "ztachip"
ELF_DEFAULT  = PROJECT_ROOT / "SW" / "build" / "ztachip.elf"
END_MARKER   = "TEST_MODEL_PASS_END"


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_mnist(img_path: str) -> bytes:
    """
    Convert a MNIST test image to pre-quantized int8 raw data (784 bytes).

    Quantization uses the model's input parameters:
        scale  = 0.003921568859368563  (≈ 1/255)
        zero_point = -128
        q = round(pixel/255 / scale) + zero_point = pixel - 128

    The output is 784 bytes of int8 data in row-major order (1×28×28),
    written directly to &classifier_input via GDB restore.
    """
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)       # [0, 255]
    arr_real = arr / 255.0                          # [0, 1]
    scale = 0.003921568859368563
    zero_point = -128
    arr_q = np.round(arr_real / scale + zero_point)
    arr_q = np.clip(arr_q, -128, 127).astype(np.int8)
    return arr_q.tobytes()                          # 784 bytes


# ─── Serial helpers (no pyserial.read) ────────────────────────────────────────

def stty_configure(port: str, baud: int) -> None:
    subprocess.run(
        ["stty", "-F", port, str(baud), "raw", "-echo"],
        check=True,
    )


def flush_serial_input(port: str) -> None:
    fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        termios.tcflush(fd, termios.TCIFLUSH)
    finally:
        os.close(fd)


def start_sed_capture(port: str, marker: str, out_path: Path) -> subprocess.Popen:
    out_f = open(out_path, "wb")
    return subprocess.Popen(
        ["sed", f"/{marker}/q", port],
        stdout=out_f,
        stderr=subprocess.DEVNULL,
    )


def wait_sed(proc: subprocess.Popen, timeout: float) -> bool:
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
    def __init__(self, gdb_path: str, elf: str, openocd_port: int,
                  bp_symbol: str = "test", load_timeout: float = 180,
                  log_path: str = ""):
        parent_fd, child_fd = pty.openpty()
        attrs = termios.tcgetattr(child_fd)
        attrs[3] &= ~(termios.ECHO | termios.ECHOCTL | termios.ECHONL)
        termios.tcsetattr(child_fd, termios.TCSANOW, attrs)

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
        self._log = open(log_path, "w") if log_path else None

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

            time.sleep(0.2)
            self._cmd(f"break {bp_symbol}", timeout=10)
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
        self._send(f"echo \\n{marker}\\n\n")
        return self._read_until(marker, timeout)

    # ── public API ────────────────────────────────────────────────────────────

    def restore(self, bin_path: str, timeout: float = 30) -> str:
        return self._cmd(f"restore {bin_path} binary &mnist_test_label_0",
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


# ─── Result parser ───────────────────────────────────────────────────────────

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
        "load_input":    float(m.group(1)),
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
    ap = argparse.ArgumentParser(
        description="Batch MNIST accuracy test for ztachip LeNet")
    ap.add_argument("--images", required=True,
                    help="Root directory containing test_images/0..9/ subfolders")
    ap.add_argument("--labels", default="",
                    help="Label file: 'subfolder/filename label' per line "
                         "(used with --start/--count)")
    ap.add_argument("--test-list", default="",
                    help="Test list file (overrides --labels/--start/--count)")
    ap.add_argument("--output", default="results_lenet.csv")
    ap.add_argument("--port",   default=DEFAULT_PORT)
    ap.add_argument("--baud",   type=int, default=DEFAULT_BAUD)
    ap.add_argument("--gdb",    default="riscv32-unknown-elf-gdb")
    ap.add_argument("--elf",    default=str(ELF_DEFAULT))
    ap.add_argument("--openocd-port", type=int, default=DEFAULT_OPENOCD)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--keep-captures", action="store_true")
    ap.add_argument("--captures-dir", default="captures_lenet")
    ap.add_argument("--no-stty", action="store_true")
    ap.add_argument("--sed-startup-delay", type=float, default=0.1)
    ap.add_argument("--inference-timeout", type=float, default=INFERENCE_TIMEOUT)
    ap.add_argument("--break-symbol", default="test")
    ap.add_argument("--gdb-log", default="")
    args = ap.parse_args()

    if not Path(args.elf).exists():
        sys.exit(f"ELF not found: {args.elf}")
    if not shutil.which("sed"):
        sys.exit("sed not found in PATH")
    if not shutil.which(args.gdb):
        sys.exit(f"GDB not found: {args.gdb}")

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

    try:
        port_guard = os.open(args.port, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError as e:
        sys.exit(f"failed to open guard fd on {args.port}: {e}")
    print(f"Held guard fd on {args.port} (stabilizes USB before GDB starts)")

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

    total = correct_top1 = correct_top5 = 0
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
                w.writerow(["idx", "image", "gt", "pred",
                             "ok_top1", "ok_top5", "bytes",
                             "fw_load_s", "fw_parse_s",
                             "fw_infer_s", "fw_total_s"])

            for i, (fname, gt) in enumerate(manifest):
                gidx = args.start + i
                p = Path(args.images) / fname
                tag = f"[{gidx+1:>6}]"

                if not p.exists():
                    print(f"{tag} SKIP missing {fname}")
                    continue
                print(f"{tag} {fname:<45} gt={gt:>2}",
                      end="  ", flush=True)

                try:
                    bin_data = preprocess_mnist(str(p))
                except Exception as e:
                    print(f"IMG_ERR {e}")
                    continue

                with tempfile.NamedTemporaryFile(suffix=".bin",
                                                  delete=False) as bf:
                    bf.write(bin_data)
                    bin_path = bf.name
                t_restore = t_infer = 0.0
                try:
                    try:
                        t0 = time.monotonic()
                        gdb.restore(bin_path, timeout=30)
                        t_restore = time.monotonic() - t0
                    except (GdbTimeout, RuntimeError) as e:
                        print(f"GDB_RESTORE_FAIL — aborting batch: {e}")
                        break

                    cap_path = cap_dir / f"capture_{gidx:06d}.bin"
                    sed_proc = start_sed_capture(args.port, END_MARKER,
                                                  cap_path)
                    time.sleep(args.sed_startup_delay)

                    try:
                        t0 = time.monotonic()
                        gdb.cont(timeout=args.inference_timeout + 30)
                        t_infer = time.monotonic() - t0
                    except (GdbTimeout, RuntimeError) as e:
                        sed_proc.kill()
                        print(f"GDB_CONT_FAIL — aborting batch: {e}")
                        break

                    if not wait_sed(sed_proc, 10):
                        print(f"SED_TIMEOUT (cap={cap_path.name})")
                        continue

                    sum_t_restore += t_restore
                    sum_t_infer   += t_infer
                    n_timed       += 1
                finally:
                    try:
                        os.unlink(bin_path)
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

                t_str = f"[restore={t_restore:.2f}s infer={t_infer:.1f}s]"
                if timing:
                    t_str += (f"  fw[load={timing['load_input']:.3f}"
                              f" parse={timing['model_parse']:.3f}"
                              f" infer={timing['inference']:.3f}"
                              f" total={timing['total']:.3f}]s")

                parsed = parse_top5(cap_path)
                if parsed is None:
                    print(f"NO_RESULT ({n}B)  {t_str}")
                    continue
                pred, top5 = parsed
                ok1 = int(pred == gt)
                ok5 = int(gt in top5)
                total += 1
                correct_top1 += ok1
                correct_top5 += ok5

                tick = "OK  " if ok1 else "FAIL"
                print(f"pred={pred:>2}  {tick}  "
                      f"Top1={correct_top1/total*100:5.1f}%  "
                      f"Top5={correct_top5/total*100:5.1f}%  "
                      f"({n}B)  {t_str}")

                w.writerow([gidx, fname, gt, pred, ok1, ok5, n] + fw_cols)
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
              f"Top1={correct_top1/total:.4f} ({correct_top1}/{total})  "
              f"Top5={correct_top5/total:.4f} ({correct_top5}/{total})")
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
        labels = {"load_input":   "Load Input",
                  "model_parse":  "Model Parse",
                  "inference":    "Inference",
                  "total":        "Total"}
        print(f"  {'Phase':<20s} {'Avg':>10s} {'Min':>10s} {'Max':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        for k in TIMING_FIELDS:
            avg_s = sum_fw[k] / n_fw
            min_s = min_fw[k]
            max_s = max_fw[k]
            print(f"  {labels[k]:<20s} {avg_s:>10.4f} {min_s:>10.4f} {max_s:>10.4f}")


if __name__ == "__main__":
    main()
