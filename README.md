# ztachip — Per-Channel Quantization Extension (`feature/ZTA-Q`)

This fork adds per-channel weight quantization to ztachip, plus a batch test
harness that runs INT8 TFLite models on the FPGA and reports Top-1/Top-5
accuracy on MNIST / ImageNet.

## Models

All models are post-training quantized to **full-integer INT8 TFLite with
per-channel weight scales** (what this fork consumes). The resulting `.tflite`
is embedded into the firmware filesystem at build time (see
[Build the firmware](#2-build-the-firmware)).

| Model | Source | Quantization calibration set |
| ----- | ------ | ---------------------------- |
| LeNet-5 | Rebuilt and trained from [TensorFlow-Slim](https://github.com/tensorflow/models/blob/master/research/slim/nets/lenet.py) | MNIST — 100 images/class × 10 classes (1000 total) |
| MobileNetV1 / V2 | Imported from `tensorflow.keras.applications` | ImageNet — 1 image/class × 1000 classes (1000 total) |
| ResNet-18 | PyTorch → ONNX → TF/TFLite via [onnx2tf](https://github.com/PINTO0309/onnx2tf) | ImageNet — 1 image/class × 1000 classes (1000 total) |

### ResNet-18: PyTorch → TFLite via onnx2tf

Export the PyTorch model to ONNX, then let `onnx2tf` produce both the float
TensorFlow model and the full-integer INT8 TFLite in one pass:

```bash
# 1. PyTorch → ONNX (in your training/export script):
#    torch.onnx.export(model, dummy, "resnet18.onnx", opset_version=13,
#                      input_names=["input"], output_names=["output"])

# 2. ONNX → TF float + INT8 TFLite (per-channel), calibrated on a 1000-image
#    ImageNet set saved as calib_data.npy (NHWC, float32, pixels in [0, 1]):
onnx2tf -i resnet18.onnx -oiqt -qt per-channel \
    -cind input calib_data.npy "[[[[0.485,0.456,0.406]]]]" "[[[[0.229,0.224,0.225]]]]"
```

`-oiqt` emits several quantized variants; this fork uses
`resnet18_full_integer_quant.tflite`. `calib_data.npy` is produced by
[generate_calib.py](Batch_Test/generate_calib.py) (resize/crop to 224×224 and
scale to `[0, 1]` for the 1000-image ImageNet calibration set). The `-cind`
mean/std applied during calibration match the torch-style preprocessing below
(`input` must match the ONNX graph's input name).

## Image preprocessing

The firmware consumes pre-quantized INT8 input directly — no image decode or
normalization happens on the target. All preprocessing runs on the host inside
the batch scripts, which write a raw blob that GDB `restore`s into the firmware
input array.

### ImageNet (MobileNetV1/V2 and ResNet-18)

Aspect-preserving resize of the shorter side to 256 (bilinear), then center crop
to 224×224. The two model families differ only in normalization / quantization:

- **MobileNetV1/V2** — [batch_imagenet_test.py](Batch_Test/batch_imagenet_test.py):
  scale pixels to `[-1, 1]` (`x = pixel/127.5 − 1`), then quantize with
  `scale = 0.007843137718737125`, `zero_point = −1`.
- **ResNet-18** — [batch_resnet18_test.py](Batch_Test/batch_resnet18_test.py):
  torch-style — scale to `[0, 1]`, normalize with ImageNet
  `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`, then quantize
  with `scale = 0.01865844801068306`, `zero_point = −14`.

Both then clip to int8 and reorder HWC→CHW → 3×224×224 = **150528 bytes**.

### MNIST (LeNet-5)

Grayscale, resize to 28×28 — [batch_lenet_test.py](Batch_Test/batch_lenet_test.py):
quantize with `scale ≈ 1/255`, `zero_point = −128` (i.e. `q = pixel − 128`),
giving 1×28×28 = **784 bytes**.

## Test harness

The FPGA runs are driven by the batch scripts under [`Batch_Test/`](Batch_Test):

| Script | Model(s) | Dataset |
| ------ | -------- | ------- |
| [batch_lenet_test.py](Batch_Test/batch_lenet_test.py) | LeNet-5 | MNIST test |
| [batch_imagenet_test.py](Batch_Test/batch_imagenet_test.py) | MobileNetV1 / V2 | ImageNet-50k val |
| [batch_resnet18_test.py](Batch_Test/batch_resnet18_test.py) | ResNet-18 | ImageNet-50k val |

Each script loads `ztachip.elf` via GDB, streams preprocessed images into target
memory, captures each inference's UART output, and writes Top-1/Top-5 results to
its `--output` CSV. The scripts are **Linux-only** (they use `pty`, `termios`,
`stty`, and `sed`).

Sample inputs and label files live in
[`Batch_Test/test_data/`](Batch_Test/test_data) — one sample image per dataset,
plus the full label lists (`imagenet/label.txt` = 50k val GT,
`imagenet/label_test.txt` = 1000-image subset, `mnist/test_labels.txt` = 10k).
Point `--images` at your own MNIST / ImageNet-val directory for a full run.

## Step-by-step FPGA setup

### 1. Build the FPGA bitstream
See [Documentation/Vivado.md](Documentation/Vivado.md).

### 2. Build the firmware
```bash
export PATH=/opt/riscv/bin:$PATH
cd SW/compiler
make clean all                       # build the DSL compiler
cd ../fs
python3 bin2c.py                     # embed models/inputs into the firmware filesystem
cd ..
make clean all -f makefile.kernels   # build the computation kernels
make clean all                       # produces SW/build/ztachip.elf
```

### 3. Start JTAG (OpenOCD)
```bash
cd <openocd_riscv install folder>
sudo src/openocd -f usb_connect.cfg -c 'set MURAX_CPU0_YAML cpu0.yaml' -f soc_init.cfg
```

### 4. Run a batch test
Out of the box the firmware runs **ResNet-18** — to test another model first see
[Switching models](#switching-models). There are two modes:

**Accuracy run** over a validation set:
```bash
python3 Batch_Test/batch_resnet18_test.py \
  --images <imagenet_val_dir> \
  --labels Batch_Test/test_data/imagenet/label.txt --count 1000 \
  --elf    SW/build/ztachip.elf \
  --port   /dev/ttyUSB1 --baud 4000000 \
  --output resnet18_results.csv
```
Use `--test-list <file>` (a `filename label` list) instead of `--labels/--count`
to run an explicit set of images, e.g.
`--test-list Batch_Test/test_data/imagenet/label_test.txt`.

**Step mode** — per-layer dumps for fine-grained comparison. Enable
`STEP_MODE=yes` in [`SW/makefile`](SW/makefile) and rebuild, then:
```bash
python3 Batch_Test/batch_resnet18_test.py \
  --images <imagenet_val_dir> \
  --labels Batch_Test/test_data/imagenet/label.txt --count 1 \
  --elf    SW/build/ztachip.elf \
  --port   /dev/ttyUSB1 --baud 4000000 \
  --parser both --external-parser Batch_Test/parse_capture.py \
  --keep-captures --gdb-log /tmp/gdb.log
```

## Switching models

The committed firmware is wired for **ResNet-18 only**. To test another model,
edit two files and rebuild:

1. **Select the test in [`SW/src/test.cpp`](SW/src/test.cpp)** — inside `test()`,
   comment out `test_resnet18()` and uncomment the one you want
   (`test_lenet()`, `test_mobilenet_v2()`, …).

2. **Wire the model + input into the firmware filesystem in
   [`SW/base/simplelib.c`](SW/base/simplelib.c)** — uncomment the matching
   `#include "../fs/gen/<name>.c"` at the top *and* the matching
   `strcmp(name, "<file>")` block in `_open()`:
   - **LeNet-5**: enable `lenet_mnist_int8` and `mnist_test_label_0`.
   - **MobileNet**: enable `mobilenet_v1_int8_bs1` (or `mobilenet_v2_int8_bs1`);
     the ImageNet input array `ILSVRC2012_val_00000001_raw` is already enabled.

3. **Rebuild** the firmware (step 2 above) and run the matching batch script. The
   script's GDB `restore` target must exist as a symbol in the ELF — e.g. the
   LeNet script restores into `&mnist_test_label_0`, so that array's `#include`
   must be enabled.

## Upstream

This fork keeps ztachip's original architecture, programming model, and build
system. For the accelerator internals and the DSL:

1. [Technical overview](Documentation/Overview.md)
2. [Hardware Architecture](Documentation/HardwareDesign.md)
3. [Programmer's Guide](https://github.com/ztachip/ztachip/raw/master/Documentation/ztachip_programmer_guide.pdf)
4. [VisionAI Stack Programmer's Guide](https://github.com/ztachip/ztachip/raw/master/Documentation/visionai_programmer_guide.pdf)

Original project: **https://github.com/ztachip/ztachip** — licensed under Apache-2.0.
