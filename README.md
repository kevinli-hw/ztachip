# ztachip — Per-Channel Quantization Extension (`feature/ZTA-Q`)

This repository is a **fork of [ztachip](https://github.com/ztachip/ztachip)**, the multicore,
data-aware, embedded RISC-V AI accelerator for edge inferencing on low-end FPGAs/ASICs.

Upstream ztachip runs INT8 CNNs using a single **per-tensor** requantization scale. This fork
**extends ztachip's quantization scheme to full per-channel (per-output-channel) requantization**,
matching the format produced by TensorFlow Lite's full-integer post-training quantization. With
per-channel requant in place, the accelerator reaches **TFLite-level accuracy** on a set of
standard image-classification models that previously could not be run faithfully:

- **LeNet-5** (MNIST)
- **MobileNetV1** (ImageNet-1k)
- **MobileNetV2** (ImageNet-1k, both `uint8` and `int8` quantized variants)
- **ResNet-18** (ImageNet-1k)

All of the work described here lives on the **`feature/ZTA-Q`** branch.

## Accuracy

End-to-end Top-1 / Top-5 accuracy, evaluated on the ImageNet-1k validation set (MobileNet, ResNet)
and the MNIST test set (LeNet). For each model three numbers are reported:

- **TF Float** — the original floating-point model (reference upper bound).
- **TFLite** — the full-integer INT8 `.tflite` model run with the TensorFlow Lite interpreter on a PC (the target we want to match).
- **FPGA** — the same INT8 model run on ztachip on an Arty A7-100T board.

| Model                    | Top-1 TF Float | Top-1 TFLite | Top-1 FPGA | Top-5 TF Float | Top-5 TFLite | Top-5 FPGA | Inference time / image |
| ------------------------ | :------------: | :----------: | :--------: | :------------: | :----------: | :--------: | :--------------------: |
| MobileNetV2 `[uint8-q]`  |       –¹       |    71.12%    |   71.10%   |       –¹       |    89.96%    |   89.88%   |        170.0 ms        |
| MobileNetV2 `[int8-q]`   |     71.48%     |    70.72%    |   70.68%   |     90.27%     |    89.88%    |   89.90%   |        170.2 ms        |
| MobileNetV1              |     70.32%     |    68.95%    |   68.95%   |     89.44%     |    88.56%    |   88.57%   |        256.4 ms        |
| LeNet-5                  |     99.20%     |    99.19%    |   99.21%   |      100%      |     100%     |    100%    |         22.4 ms        |
| ResNet-18                |     69.81%     |    69.57%    |   69.50%   |     89.11%     |    88.93%    |   88.91%   |         1.22 s         |

> ¹ The `uint8` MobileNetV2 is consumed as a pre-quantized TFLite model, so no separate
> floating-point baseline was measured for it.

The **FPGA column tracks the TFLite column to within a few hundredths of a percent** across all
models, which is the goal of this work: the hardware/software implementation of per-channel
requantization is numerically faithful to the reference TFLite kernels.

---

## Implementation — hardware/software co-design (`feature/ZTA-Q`)

Per-channel requantization means every output channel of a conv / depthwise-conv / fully-connected
layer carries its **own `(multiplier, shift)` pair** (TFLite's
`MultiplyByQuantizedMultiplier`). Upstream ztachip applied a single per-tensor scale, so the
extension touches the whole stack — the tensor-engine ALU, the DSL compiler, the kernel library,
and the stream/scalar processor activation path.

### 1. Two new tensor-engine micro-instructions

| Opcode | Name        | Semantics | Role |
| -----: | ----------- | --------- | ---- |
| 13     | `QUANT_MUL` | `Y = round((x1 × x2) >> 15)` — saturating-rounding-doubling-high-multiply | The **multiplier** half of requantization, evaluated at full accumulator width in one ALU pass. |
| 14     | `SHRA_V`    | Arithmetic shift-right where the **shift count comes from a vector register** (per lane) | The **shift** half of requantization; because the count is per-lane, each output channel can shift by a different amount. |

Together they let a Pcore apply a *distinct* `(multiplier, shift)` per output channel, in place on
the 32-bit accumulator — replacing the previous single per-tensor scale.

- **ALU hardware** — [`HW/quant_mul/src/alu/alu.vhd`](HW/quant_mul/src/alu/alu.vhd)
  - `QUANT_MUL` datapath: reuses the shared multiplier at `2×` accumulator width, applies the
    rounding *nudge* (`quant_mul_nudge_pos/neg`) and an arithmetic right shift by
    `quant_mul_shift_distance`, matching TFLite's `SaturatingRoundingDoublingHighMul` rounding.
  - `SHRA_V` vector-shift path.
- **HW package** — [`HW/quant_mul/src/ztachip_pkg.vhd`](HW/quant_mul/src/ztachip_pkg.vhd): opcode
  constants (`mu_opcode_quant_mul_c`, `mu_opcode_shra_v_c`) and the shift-distance constant.

### 2. DSL compiler support

The new opcodes are taught to the ztachip C-like DSL compiler:

- [`SW/compiler/config.h`](SW/compiler/config.h) — `OPCODE_QUANT_MUL = 13`, `OPCODE_SHRA_V = 14`.
- [`SW/compiler/config.cpp`](SW/compiler/config.cpp) — opcode table entries (operand kinds / data types).
- [`SW/compiler/gen.cpp`](SW/compiler/gen.cpp) — code generation emits `QUANT_MUL`.
- [`SW/compiler/instruction.cpp`](SW/compiler/instruction.cpp) — instruction encoding / scheduling for the new ALU ops.

### 3. Kernel-library extension

New per-channel activation kernels were added to ztachip's neural-net kernel library, using the
`QUANT_MUL` + `SHRA_V` idiom (`_A = _A * multiplier; top = _A >> shift;`, with `multiplier`/`shift`
as per-channel vectors):

- [`SW/apps/nn/kernels/conv.p`](SW/apps/nn/kernels/conv.p)
  - `convolution::activate_per_channel`
  - `convolution1x1::activate_per_channel`
  - `convolution_depthwise::exe3x3_per_channel`
- [`SW/apps/nn/kernels/fcn.p`](SW/apps/nn/kernels/fcn.p)
  - `inner_product::activate_per_channel` — per-channel fully-connected layer.

### 4. Stream/scalar-processor activation path

The per-channel multiplier/shift tables are loaded from the model and the post-multiply stage
(adding the output zero-point and clamping to INT8) is handled by
`SpuEvalActivation_Per_Channel`:

- [`SW/apps/nn/nn_conv2d.cpp`](SW/apps/nn/nn_conv2d.cpp) — `SpuEvalActivation_Per_Channel`
  ([line 355](SW/apps/nn/nn_conv2d.cpp:355)) wired into the conv layer
  ([line 166](SW/apps/nn/nn_conv2d.cpp:166)).

---

## Testing

Accuracy is measured by running the **same INT8 `.tflite` model** in three places and comparing:
the float reference, the PC TFLite interpreter, and ztachip on the FPGA. The FPGA runs are driven
by the batch test harness in the repository root:

| Script | Model(s) | Dataset |
| ------ | -------- | ------- |
| [`batch_lenet_test.py`](batch_lenet_test.py)        | LeNet-5            | MNIST test |
| [`batch_imagenet_test.py`](batch_imagenet_test.py)  | MobileNetV1 / V2  | ImageNet-1k val |
| [`batch_resnet18_test.py`](batch_resnet18_test.py)  | ResNet-18         | ImageNet-1k val |

### How the harness works (common to all three scripts)

The scripts drive the FPGA over JTAG (OpenOCD + GDB) and capture results over UART. One persistent
GDB process drives the whole batch:

1. **Startup (once):** `set pagination off` → `file ztachip.elf` → `target extended-remote
   localhost:3333` → `monitor reset halt` → `load` → `break test` → `continue` (runs `crt0` +
   `main()`, halts at the first call to `test()`).
2. **Per image:**
   - preprocess the image in Python (see per-model details below),
   - `restore` the preprocessed bytes directly into target memory while the CPU is halted,
   - flush the serial buffer and start a `sed '/TEST_MODEL_PASS_END/q' /dev/ttyUSBx > capture.bin`
     (UART bytes are captured by `sed` reading the tty directly — more reliable than `pyserial` at 4 Mbaud),
   - `continue` — the firmware runs one inference, prints `RESULT top1=… … top5=…`, loops back to
     `while(1)` and calls `test()` again, hitting the breakpoint,
   - parse the top-5 ids from the capture and accumulate Top-1 / Top-5 hit counts.
3. **Output:** a `results.csv` with the per-image predictions, from which Top-1/Top-5 accuracy is computed.

The firmware entry point is [`SW/src/test.cpp`](SW/src/test.cpp): `test()` dispatches to
`test_lenet()`, `test_mobilenet_v2()` or `test_resnet18()`, each of which calls
`TF2.Create("<model>.tflite", …)`, runs inference, and prints the `RESULT … / TEST_MODEL_PASS_END`
markers the harness keys on.

> **Selecting a model:** edit `test()` in [`SW/src/test.cpp`](SW/src/test.cpp) to call the desired
> `test_*()`, make sure the matching `.tflite` is embedded in the firmware filesystem (`SW/fs`),
> and rebuild (see *Build & run on FPGA* below).

### Per-model preparation

For each model the steps are: **download → quantize (INT8 TFLite) → embed → preprocess → run.**
The download + quantization steps are done offline on a PC; the resulting `.tflite` is embedded
into the firmware. Full-integer post-training quantization is the standard TFLite recipe and is
what produces the **per-channel** weight scales this fork consumes:

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)   # or from_saved_model(...)
converter.optimizations          = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen     # ~100–500 calibration images
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type   = tf.int8     # tf.uint8 for the uint8 MobileNetV2 variant
converter.inference_output_type  = tf.int8
open("model_int8.tflite", "wb").write(converter.convert())
```

#### LeNet-5 — MNIST  →  `lenet_mnist_int8.tflite`

- **Download / train:** a small LeNet-5 CNN trained on MNIST.
- **Quantize:** full-integer INT8 (input `scale ≈ 1/255`, `zero_point = -128`).
- **Preprocess** (in [`batch_lenet_test.py`](batch_lenet_test.py)): grayscale, resize to 28×28
  (bilinear), quantize `q = round(pixel/255 / scale) + zp = pixel − 128`, emit a raw **784-byte**
  INT8 buffer restored directly to `&classifier_input`.
- **Run:**
  ```bash
  python3 batch_lenet_test.py \
      --images  <mnist_test_images_dir> \
      --labels  label_test.txt \
      --elf     SW/build/ztachip.elf \
      --port    /dev/ttyUSB1 --baud 4000000 \
      --output  lenet_results.csv
  ```
  Image layout: `test_images/<digit>/test_NNNNN.png`; label file lines: `<subfolder/file> <label>`.

#### MobileNetV1 / MobileNetV2 — ImageNet  →  `mobilenet_v1_int8_bs1.tflite`, `mobilenet_v2_int8_bs1.tflite`, `mobilenet_v2_1_0_224_quant.tflite`

- **Download:**
  - `uint8` MobileNetV2 — the official pre-quantized model
    [`mobilenet_v2_1.0_224_quant.tflite`](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz).
  - `int8` MobileNetV1/V2 — start from `tf.keras.applications.MobileNet` / `MobileNetV2`
    (ImageNet weights) and quantize with the recipe above.
- **Preprocess** (in [`batch_imagenet_test.py`](batch_imagenet_test.py)): resize shorter side to
  256 (bilinear), center-crop 224×224, write a 24-bit **uint8 BMP** (150582 bytes = 14+40+224·672)
  restored to `&classifier_input`. No `−128` / normalization in Python:
  - for the `uint8` model (`scale = 0.0078125`, `zp = 128`) the `(x−128)·scale` happens inside the model;
  - for the `int8`-input model the firmware's `CreateWithBitmap(..., TensorDataTypeInt8)` subtracts 128.
- **Run:**
  ```bash
  python3 batch_imagenet_test.py \
      --images  <imagenet_val_dir> \
      --test-list <manifest.txt> \      # or --labels <file> --start N --count M
      --elf     SW/build/ztachip.elf \
      --port    /dev/ttyUSB1 --baud 4000000 \
      --output  mobilenet_results.csv
  ```
  > MobileNet outputs **1001** classes (background at id 0); use a 1001-class label manifest.

#### ResNet-18 — ImageNet  →  `resnet18_full_integer_quant.tflite`

- **Download:** a ResNet-18 trained on ImageNet (e.g. a Keras ResNet-18 or torchvision
  `resnet18` exported to TF/ONNX→TF), then quantized full-integer INT8.
- **Preprocess** (in [`batch_resnet18_test.py`](batch_resnet18_test.py)), matching the golden
  PyTorch-style flow: resize shorter side to 256 (bilinear) → 224×224 center crop → `float/255` →
  normalize with ImageNet mean/std (stays RGB) → quantize `q = round(x/scale + zp)`
  (`scale = 0.01865844801068306`, `zp = -14`, clip INT8) → **HWC→CHW** → raw **150528-byte** buffer.
  The buffer is `restore`d over the `ILSVRC2012_val_00000001_raw` array (mapped by the firmware's
  fake filesystem in [`SW/base/simplelib.c`](SW/base/simplelib.c)).
- **Run:**
  ```bash
  python3 batch_resnet18_test.py \
      --images  <imagenet_val_dir> \
      --test-list <manifest.txt> \
      --elf     SW/build/ztachip.elf \
      --port    /dev/ttyUSB1 --baud 4000000 \
      --output  resnet18_results.csv
  ```
  > ResNet-18 outputs **1000** classes (no background); the manifest's label ids must use 0..999 indexing.

### Build & run on the Arty A7-100T FPGA

The batch scripts above assume a programmed board and a running OpenOCD/GDB + UART session. End to end:

1. **Build the FPGA bitstream** (Xilinx Vivado, Arty A7-100T) from the RTL in `HW/`, including the
   per-channel ALU in [`HW/quant_mul/src`](HW/quant_mul/src). Program the board; confirm the green
   "done" LED. See [`Documentation/Vivado.md`](Documentation/Vivado.md).

2. **Build the firmware** with the desired model. Select it in `test()`
   ([`SW/src/test.cpp`](SW/src/test.cpp)), place the matching `.tflite` in `SW/fs`, then:
   ```bash
   export PATH=/opt/riscv/bin:$PATH
   cd SW/compiler && make clean all          # build the DSL compiler (incl. QUANT_MUL/SHRA_V)
   cd ../fs       && python3 bin2c.py         # embed models/inputs into the firmware filesystem
   cd ..          && make clean all -f makefile.kernels
   make clean all                            # produces SW/build/ztachip.elf
   ```

3. **Start JTAG (OpenOCD)** on the Linux host so GDB can load the ELF on port 3333:
   ```bash
   cd <openocd_riscv installation folder>
   sudo src/openocd -f usb_connect.cfg -c 'set MURAX_CPU0_YAML cpu0.yaml' -f soc_init.cfg
   ```

4. **Connect UART** — Arty A7 exposes the serial port over USB (e.g. `/dev/ttyUSB1`) at **4 Mbaud**,
   flow control disabled. This is the channel the harness captures `RESULT …` lines from.

5. **Run a batch test** — invoke the matching `batch_*_test.py` from the table above. The script
   loads `ztachip.elf` via GDB, streams preprocessed images into target memory, captures each
   inference's UART output, and writes Top-1/Top-5 results to its `--output` CSV.

---

## Upstream documentation

This fork keeps ztachip's original architecture, programming model, and build system. For the
accelerator internals and the DSL, see upstream:

1. [Technical overview](Documentation/Overview.md)
2. [Hardware Architecture](Documentation/HardwareDesign.md)
3. [Programmer's Guide](https://github.com/ztachip/ztachip/raw/master/Documentation/ztachip_programmer_guide.pdf)
4. [VisionAI Stack Programmer's Guide](https://github.com/ztachip/ztachip/raw/master/Documentation/visionai_programmer_guide.pdf)

Original project: **https://github.com/ztachip/ztachip** — licensed under Apache-2.0.
