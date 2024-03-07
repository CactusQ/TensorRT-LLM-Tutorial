
# TensorRT-LLM: A Tutorial On Getting Started

Beginner-friendly tutorial for Tensor-RT-LLM using BLOOM-560M as an example model.

Video walkthrough and explanation:

[![Youtube video link](https://img.youtube.com/vi/TwWqPnuNHV8/0.jpg)](https://youtu.be/TwWqPnuNHV8)

### Accelerating BLOOM 560M Inference with TensorRT-LLM

This Jupyter notebook demonstrates the optimization of the BLOOM 560M model, a large language model, for faster inference using NVIDIA's TensorRT-LLM. The guide covers the installation of necessary tools, downloading and preparing the BLOOM model, and the steps to convert and optimize the model using TensorRT-LLM for both FP16 and INT8 quantization. It also includes a comparison of inference speed results between the baseline model from Huggingface, the optimized FP16 model, and the INT8 quantized model.

## Prerequisites
- NVIDIA GPU with CUDA support
- Docker and NVIDIA Container Toolkit installed (will be installed in the notebook as well)
- Python 3.10, pip, and necessary Python libraries
- Jupyter or Google Colab

Or run the docker container and install Jupyter there:
```bash
docker run --rm --runtime=nvidia --gpus all --entrypoint /bin/bash -it nvidia/cuda:12.1.0-devel-ubuntu22.04
```


## Overview
This notebook provides a detailed walkthrough for:

- **Installing the NVIDIA Container Toolkit**: Ensures that Docker containers can utilize the full power of NVIDIA GPUs.
- **Installing TensorRT-LLM**: Steps to clone the NVIDIA TensorRT-LLM repository and install the required Python packages.
- **Downloading BLOOM**: Instructions to download the BLOOM 560M model from Huggingface.
- **Converting and Building the BLOOM Model**: Processes to convert the BLOOM model from its original Huggingface format to a format compatible with TensorRT-LLM and optimize it for faster inference using FP16 and INT8 quantization.
- **Benchmarking**: Compares execution time and ROUGE metrics for summarization tasks between the baseline Huggingface model and the optimized TensorRT-LLM models.

## Key Steps

- **Model Loading and Conversion**: Load the BLOOM 560M model and convert it to the TensorRT-LLM optimized format.
- **Accelerating Inference with TensorRT**: The notebook demonstrates converting the BLOOM model to a TensorRT-optimized model, significantly reducing inference times.
- **Applying INT8 Quantization**: Further optimization using INT8 quantization to reduce model size and accelerate inference speed, with a comparative analysis of performance impact.
- **Benchmarking and Results Analysis**: In-depth comparison of inference speeds and performance metrics (like ROUGE scores) across the baseline, TensorRT-optimized, and INT8-quantized models. Visualizations included showcase the performance improvements.

## Results
The notebook concludes with a comparative analysis showcasing the inference speed improvements and performance metrics. It provides a clear visualization of the speed-ups achieved through TensorRT optimization and INT8 quantization, highlighting the substantial decrease in inference time while maintaining or improving model performance.

## Conclusion
This guide demonstrates the effectiveness of TensorRT-LLM in optimizing the BLOOM 560M model for faster inference. It serves as a valuable resource for AI practitioners looking to enhance the performance of large language models for real-world applications, making it especially useful for tasks requiring high throughput and low latency.
