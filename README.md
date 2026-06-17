# ztachip — Per-Channel Quantization Extension (`feature/ZTA-Q`)

# Testing

The FPGA runs are driven by the batch test harness in the repository root:

| Script | Model(s) | Dataset |
| ------ | -------- | ------- |
| [`batch_lenet_test.py`](batch_lenet_test.py)        | LeNet-5            | MNIST test |
| [`batch_imagenet_test.py`](batch_imagenet_test.py)  | MobileNetV1 / V2  | ImageNet-50k val |
| [`batch_resnet18_test.py`](batch_resnet18_test.py)  | ResNet-18         | ImageNet-50k val |

# Model preparation

For each model, the steps are: **download → quantize (INT8 TFLite) → embed → preprocess → run.**
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

# Step-by-step FPGA Setup

### Build the FPGA bitstream 
See [`Documentation/Vivado.md`](Documentation/Vivado.md).

### **Build the firmware**
```
export PATH=/opt/riscv/bin:$PATH
cd SW/compiler
make clean all                       # build the DSL compiler
cd ../fs      
python3 bin2c.py                     # embed models/inputs into the firmware filesystem
cd ..
make clean all -f makefile.kernels   # prepare computation kernel program    
make clean all                       # produces SW/build/ztachip.elf
```

### Start JTAG (OpenOCD)
```
cd <openocd_riscv installation folder>
sudo src/openocd -f usb_connect.cfg -c 'set MURAX_CPU0_YAML cpu0.yaml' -f soc_init.cfg
```

### Run a batch test
The script loads `ztachip.elf` via GDB, streams preprocessed images into target memory, captures each 
inference's UART output, and writes Top-1/Top-5 results to its `--output` CSV.
  1) step mode: print outputs layer by layer for fine-grained comparison. Need to enable STEP mode in the Makefile.
```
python3 batch_resnet18_test.py \
  --images <imagenet_val_dir> \
  --labels <label.txt> --count 1 \
  --elf     SW/build/ztachip.elf \
  --port    /dev/ttyUSB1 --baud 4000000 \
  --parser both --external-parser parse_capture.py \
  --keep-captures --gdb-log /tmp/gdb2.log

```
  3) batch mode: accuracy test on the validation dataset.
```
python3 batch_resnet18_test.py \
  --images  <imagenet_val_dir> \
  --test-list <label.txt> \ or [--labels <label.txt> --count N]
  --elf     SW/build/ztachip.elf \
  --port    /dev/ttyUSB1 --baud 4000000 \
  --output  resnet18_results.csv
```


## Upstream documentation

This fork keeps ztachip's original architecture, programming model, and build system. For the
accelerator internals and the DSL, see upstream:

1. [Technical overview](Documentation/Overview.md)
2. [Hardware Architecture](Documentation/HardwareDesign.md)
3. [Programmer's Guide](https://github.com/ztachip/ztachip/raw/master/Documentation/ztachip_programmer_guide.pdf)
4. [VisionAI Stack Programmer's Guide](https://github.com/ztachip/ztachip/raw/master/Documentation/visionai_programmer_guide.pdf)

Original project: **https://github.com/ztachip/ztachip** — licensed under Apache-2.0.
