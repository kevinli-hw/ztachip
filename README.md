# ztachip — Per-Channel Quantization Extension (`feature/ZTA-Q`)

This repository is a fork of [ztachip](https://github.com/ztachip/ztachip), the multicore,
data-aware, embedded RISC-V AI accelerator for edge inferencing on low-end FPGAs/ASICs.

Upstream ztachip runs INT8 CNNs using a single per-tensor requantization scale. This fork
extends ztachip's quantization scheme to both per-tensor and per-channel quantization,
matching the format produced by TensorFlow Lite's full-integer post-training quantization. 
The accelerator reaches **TFLite-level accuracy** on a set of standard image-classification models that previously could not be run faithfully:

- **LeNet-5** (MNIST)
- **MobileNetV1** (ImageNet-50k)
- **MobileNetV2** (ImageNet-50k, both `uint8` and `int8` quantized variants)
- **ResNet-18** (ImageNet-50k)

All of the work described here lives on the **`feature/ZTA-Q`** branch.

## Accuracy

End-to-end Top-1 / Top-5 accuracy, evaluated on the ImageNet-50k validation set (MobileNet, ResNet)
and the MNIST test set (LeNet). For each model, three numbers are reported:

- **TF Float** — the original floating-point model.
- **TFLite** — the full-integer `.tflite` model run with the TensorFlow Lite interpreter on CPU.
- **FPGA** — the same INT8 model run on ztachip on an Arty A7-100T board.

| Model                    | Top-1 TF Float | Top-1 TFLite | Top-1 FPGA | Top-5 TF Float | Top-5 TFLite | Top-5 FPGA | Inference time / image |
| ------------------------ | :------------: | :----------: | :--------: | :------------: | :----------: | :--------: | :--------------------: |
| MobileNetV2 `[uint8]`  |       –¹       |    71.12%    |   71.10%   |       –¹       |    89.96%    |   89.88%   |        170.0 ms        |
| MobileNetV2 `[int8]`   |     71.48%     |    70.72%    |   70.68%   |     90.27%     |    89.88%    |   89.90%   |        170.2 ms        |
| MobileNetV1              |     70.32%     |    68.95%    |   68.95%   |     89.44%     |    88.56%    |   88.57%   |        256.4 ms        |
| LeNet-5                  |     99.20%     |    99.19%    |   99.21%   |      100%      |     100%     |    100%    |         22.4 ms        |
| ResNet-18                |     69.81%     |    69.57%    |   69.50%   |     89.11%     |    88.93%    |   88.91%   |         1.22 s         |

> ¹ The `uint8` MobileNetV2 is consumed as a pre-quantized TFLite model provided by [ztachip](https://github.com/ztachip/ztachip) repository, so no separate
> floating-point baseline was measured for it.

---

## Implementation Details (`feature/ZTA-Q`)

### 1. Two new tensor-engine micro-instructions

| Opcode | Name        | Semantics | Role |
| -----: | ----------- | --------- | ---- |
| 13     | `QUANT_MUL` | `Y = round((x1 × x2) >> 15)` — saturating-rounding-doubling-high-multiply | The **multiplier** half of quantization flow|
| 14     | `SHRA_V`    | Arithmetic shift-right where the shift count comes from a vector register (per lane) | The **shift** half of quantization flow|

Together they let a Pcore apply a distinct `(multiplier, shift)` per output channel, in place on
the 32-bit accumulator — replacing the previous single per-tensor scale. New opcodes are taught to the ztachip C-like DSL compiler.

### 2. Kernel-library extension

New per-channel activation kernels are added to ztachip's neural-net kernel library, using the
`QUANT_MUL` + `SHRA_V` idiom in the pcore program. New operators like `FC`, `MEAN`, and `MaxPool` are also added to the kernel library.

### 3. Stream/scalar-processor activation path

The per-channel multiplier/shift tables are loaded from the model, and the post-multiply stage
(adding the output zero-point and clamping to INT8) is handled by
`SpuEvalActivation_Per_Channel`

## Upstream documentation

This fork keeps ztachip's original architecture, programming model, and build system. For the
accelerator internals and the DSL, see upstream:

1. [Technical overview](Documentation/Overview.md)
2. [Hardware Architecture](Documentation/HardwareDesign.md)
3. [Programmer's Guide](https://github.com/ztachip/ztachip/raw/master/Documentation/ztachip_programmer_guide.pdf)
4. [VisionAI Stack Programmer's Guide](https://github.com/ztachip/ztachip/raw/master/Documentation/visionai_programmer_guide.pdf)

Original project: **https://github.com/ztachip/ztachip** — licensed under Apache-2.0.
