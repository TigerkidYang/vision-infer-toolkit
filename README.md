# Vision-Infer-Toolkit

A comprehensive project to benchmark, optimize, and accelerate a ResNet-50 computer vision model. This repository documents the end-to-end workflow from a baseline PyTorch implementation to a highly optimized TensorRT engine, including the development of a custom operator plugin.

-----

## 🚀 Project Workflow

This project is structured into several distinct phases, each building upon the last to achieve maximum inference performance.

### **Phase 0: Environment Setup**

The foundation of the project is a reproducible, isolated environment to ensure consistency and avoid conflicts with the host system.

  * **Objective**: Configure the development environment using Docker.
  * **Key Tasks**:
      * Install NVIDIA Driver, Docker, and the NVIDIA Container Toolkit on the host machine.
      * Write a `Dockerfile` based on the official NVIDIA TensorRT image (`nvcr.io/nvidia/tensorrt:23.10-py3`).
      * The Dockerfile will install essential packages like `git`, `cmake`, and Python libraries including `torch`, `onnx`, `onnxruntime-gpu`, and `pycuda`.
      * Build the Docker image and launch a container with GPU support.
  * **Tech Stack**: `Docker`, `NVIDIA Container Toolkit`, `Bash`.

### **Phase 1: Establish Baseline - PyTorch Native Inference**

Before any optimization, we must obtain the "golden standard" performance metrics.

  * **Objective**: Measure the performance of the unoptimized ResNet-50 model in PyTorch.
  * **Key Tasks**:
      * **Model Preparation**: Write `baseline_pytorch.py` to load a pretrained `resnet50` from `torchvision.models` and set it to evaluation mode (`.eval()`).
      * **Benchmarking**:
          * **Latency**: Measure average inference time on a single image over 100 runs, preceded by warm-up iterations.
          * **Throughput**: Calculate images per second (FPS) by running inference on batches of images (e.g., batch size 32).
          * **Accuracy**: Evaluate Top-1 accuracy on a representative subset of the ImageNet validation set.
          * **Model Size**: Record the size of the saved model state dictionary (`.pth` file).
      * **Operator Profiling**: Use `torch.profiler` to identify the most time-consuming operators (e.g., Conv, BatchNorm) in the PyTorch backend.
  * **Tech Stack**: `PyTorch`, `torch.profiler`, `torchvision`.

### **Phase 2: Model Format Conversion - ONNX**

The next step is to convert the model to a standardized intermediate representation.

  * **Objective**: Convert the PyTorch model to the ONNX (Open Neural Network Exchange) format.
  * **Key Tasks**:
      * Write a script `export_onnx.py` using the `torch.onnx.export()` function.
      * Set `dynamic_axes` to allow the ONNX model to accept variable batch sizes during inference.
      * Validate the exported model using `onnx.checker.check_model()`.
      * Benchmark the ONNX model using `onnxruntime-gpu` to ensure the conversion process has not introduced significant errors in accuracy or performance.
  * **Tech Stack**: `torch.onnx`, `onnx`, `onnxruntime-gpu`.

### **Phase 3 & 4: TensorRT Core Optimization (FP16/INT8)**

This is the core optimization phase where we leverage TensorRT to build highly optimized inference engines.

  * **Objective**: Build optimized engines in FP32, FP16, and INT8 precisions.
  * **Key Tasks**:
      * **Build Script**: Write `build_engine.py` using the TensorRT Python API.
      * **FP32/FP16 Engines**: Parse the ONNX model and build the engines. For FP16, enable the `FP16` builder flag.
      * **INT8 Engine (Post-Training Quantization)**:
          * Prepare a calibration dataset by sampling a few hundred images from the validation set.
          * Implement a calibrator class (`ImageCalibrator`) that inherits from TensorRT's `IInt8MinMaxCalibrator` to feed calibration data to the builder.
          * Enable the `INT8` builder flag and assign the calibrator instance to the builder configuration.
      * **Inference Script**: Write `infer_tensorrt.py` to load the serialized `.engine` files, create execution contexts, manage GPU memory buffers (`pycuda`), and run inference.
      * **Full Benchmarking**: Re-evaluate Latency, Throughput, and Accuracy for the FP32, FP16, and INT8 engines.
  * **Tech Stack**: `TensorRT Python API`, `PyCUDA`, PTQ.

### **Phase 5: Custom Operator - IPluginV2 Development (Advanced)**

To demonstrate deep customization capabilities, we will extend TensorRT with a custom operator.

  * **Objective**: Implement a custom activation function (HardSwish) as a C++ TensorRT plugin.
  * **Key Tasks**:
      * **Model Modification**: In PyTorch, replace a standard `nn.ReLU` layer in the ResNet model with a custom HardSwish module.
      * **C++ Plugin Implementation**:
          * Create `HardSwishPlugin.h/.cpp` inheriting from `IPluginV2DynamicExt`, implementing virtual functions, especially `enqueue()` which will launch the custom CUDA kernel.
          * Write `kernel.cu` containing the CUDA kernel for the HardSwish computation.
          * Create a `HardSwishPluginCreator` class to register the plugin with TensorRT's registry.
          * Write a `CMakeLists.txt` to compile the C++ code into a shared library (`.so` file).
      * **End-to-End Integration**:
          * Export the modified PyTorch model to ONNX, which will contain a custom node for the unknown HardSwish operator.
          * In `build_engine.py`, load the compiled plugin library before parsing the ONNX model. TensorRT will then match the custom node to the registered C++ plugin.
          * Verify that the resulting engine works correctly and produces results consistent with the PyTorch model.
  * **Tech Stack**: `C++`, `CUDA C`, `TensorRT IPluginV2 API`, `CMake`.

### **Phase 6: Synthesis & Reporting**

The final phase is to professionally present the project outcomes.

  * **Objective**: Consolidate all benchmark results and provide a clear, data-driven analysis of the project.
  * **Key Tasks**:
      * **Comparison Table**: Create a summary table (in Markdown or using `matplotlib`) comparing all engines (PyTorch, ONNX-RT, TRT-FP32/FP16/INT8) across all metrics: Latency, Throughput, Accuracy, and Model/Engine Size.
      * **Trade-off Plot**: Generate a scatter plot with Throughput (X-axis) vs. Accuracy (Y-axis) to visualize the performance/accuracy trade-off of different quantization levels.
      * **Profiling Analysis**: Use Nsight Systems to profile the PyTorch and TensorRT inference processes. Capture and compare timeline views to visually demonstrate the effect of operator fusion.
      * **Write `README.md`**: Compose a comprehensive project report detailing the project's goals, setup instructions, methodology, results, and analysis, with a special focus on the custom plugin implementation.
  * **Tech Stack**: `matplotlib`, `Markdown`, `NVIDIA Nsight Systems`.

-----

## 📊 Phase 1: Baseline Performance (PyTorch)

The first step is to establish a reliable performance baseline using the native PyTorch framework. This provides the reference point against which all subsequent optimizations will be measured.

### Environment

  * **Base Image**: `nvcr.io/nvidia/tensorrt:23.10-py3`
  * **PyTorch Version**: `2.1.0`
  * **CUDA Version (in container)**: `12.2`
  * **GPU**: `NVIDIA GeForce RTX 3060 Laptop`

### Benchmark Results

The model was benchmarked across 10 independent runs to ensure stable and reliable metrics. The following table summarizes the aggregated results.

| Metric | Value |
| :--- | :--- |
| **Model** | `ResNet50-PyTorch` |
| **Accuracy (Top-1)** | `9.50%` |
| **Latency (ms)** | `7.00 ± 0.74` |
| **Throughput (FPS)** | `462.58 ± 5.30` |
| **Model Size (MB)** | `97.80` |

*Latency is reported as the average of medians over 10 runs. Throughput is the average FPS. Both are reported with their standard deviation.*

### Operator Profiling

We used `torch.profiler` to analyze the operator-level performance on the CPU and CUDA backends. The table below shows the top CUDA operators sorted by total time consumption.

| Operator Name | Self CPU Time | CPU Total Time | \# of Calls |
| :--- | :--- | :--- | :--- |
| **model\_inference** | 57.061ms | 308.229ms | 1 |
| **aten::cudnn\_convolution** | 51.935ms | 85.166ms | 1060 |
| **cudaLaunchKernel** | 53.175ms | 53.175ms | 4020 |
| **aten::batch\_norm** | 9.286ms | 86.628ms | 1060 |
| **aten::convolution** | 8.831ms | 99.304ms | 1060 |

**Analysis**: As expected, the vast majority of the execution time is spent in **`aten::cudnn_convolution`**. This confirms that convolution operations are the primary performance bottleneck in a standard ResNet model. Subsequent optimizations with TensorRT will focus heavily on fusing these and other operations into more efficient kernels.

-----

## 🛠️ How to Run

### 1\. Build the Docker Image

```bash
docker build -t vision-infer-toolkit .
```

### 2\. Run the Docker Container

This command starts an interactive session inside the container, mounting your current project directory into the `/workspace` directory.

```bash
docker run --gpus all -it -v $(pwd):/workspace vision-infer-toolkit /bin/bash
```

### 3\. Execute the Baseline Benchmark

From within the container's shell, run the script:

```bash
python src/baseline_pytorch_en.py
```