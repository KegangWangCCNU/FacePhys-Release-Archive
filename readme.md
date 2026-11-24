# FacePhys Model Release

> This repository is an archive of FacePhys. For the maintained release, see [FacePhys/FacePhys-Release](https://github.com/FacePhys/FacePhys-Release). The current code, weights, and derivatives are covered by the [FacePhys Academic Community License](LICENSE), aligned with the maintained repository.

FacePhys is a lightweight rPPG algorithm that uses State Space Models (SSMs) to model heart states. With a memory usage of **only 4MB**, it is ideal for on-device inference. 
<img width="1258" height="896" alt="image" src="https://github.com/user-attachments/assets/1f1b6bc8-39f4-49a6-98be-4dd234006549" />

## Benchmark

<img width="1303" height="418" alt="image" src="https://github.com/user-attachments/assets/9ce011be-e003-440b-b853-6e4c71fb5053" />

## Python SDK 

FacePhys has been added to [Open-rPPG](https://github.com/KegangWangCCNU/open-rppg). Open-rPPG is a JAX-based high-performance rPPG algorithm inference toolkit. 

## FacePhys Inference 

### Inference Performance

FacePhys features a custom State Space Model (SSM) relying on frequent small tensor ops and dimension reshaping. While the model is lightweight, **the overhead from numerous fragmented operators limits inference speed.**

We tested this on multiple engines with a **batch size of 1** to simulate real-time scenarios.

#### 1. GPU Performance
**Device:** NVIDIA Tesla V100 SXM2 (32GB)

| Runtime | Inference FPS | Backend |
| :--- | ---: | :---: |
| **JAX JIT** | **1002** | CUDA |
| **TensorRT** | 778 | CUDA |
| **ONNX-Runtime** | 101 | CUDA |
| **TensorFlow JIT** | 84 | CUDA |
| **Torch Baseline** | 11 | CUDA |
| **Torch Compile** | 7 | CUDA |

> **Note:** PyTorch full graph compilation failed due to graph breaks.

#### 2. CPU Performance
**Device:** Intel Xeon Gold 6138 (Single Core)

| Runtime | Inference FPS | Backend |
| :--- | ---: | :---: |
| **LiteRT** | **337** | CPU |
| **ONNX-Runtime** | 180 | CPU |
| **JAX JIT** | 167 | CPU |
| **TensorFlow JIT** | 96 | CPU |
| **Torch Compile** | 8 | CPU |
| **Torch Baseline** | 7 | CPU |

#### Deployment Strategy

**The Bottleneck: Kernel Launch Overhead**
As shown in the GPU benchmarks, standard runtimes struggle with the SSM's fragmented operators. While GPUs possess high compute power, the latency from launching thousands of tiny kernels creates a bottleneck. **JAX-XLA and TensorRT** overcome this by performing **kernel fusion**—compiling the graph into fewer, monolithic kernels—drastically reducing launch overhead.

**Why LiteRT?**
Since rPPG targets **on-device usage** rather than cloud deployment, we prioritized CPU efficiency over raw GPU throughput. Unlike `mamba-ssm`, we avoided writing custom CUDA kernels to maintain portability. Instead, we utilized the [LiteRT](https://github.com/google-ai-edge/LiteRT) cross-platform runtime, which provided the best balance of performance and portability on CPU backends.  

#### Inference Example 

```python
from inference import * 
import matplotlib.pyplot as plt 

ipt   = np.load('weights/input.npy') # (1, 1600, 36, 36, 3)
state = load_state('weights/state.gz')

model = load_model_onnx('weights/model.onnx')
wave, final_state = inference(ipt[0], model, state)

plt.plot(wave)
```

For more examples, see `examples.ipynb`. 

### LiteRT Web Demo 

We utilized `LiteRT` to construct [a highly efficient rPPG demo capable of on-device inference.](https://github.com/KegangWangCCNU/FacePhys-Demo) 

### Paper 

Please cite **FacePhys: State of the Heart Learning, NeurIPS 2026** when using these materials in academic work.

[Official paper page and citation information](https://neurips.cc/virtual/2026/poster/150724)

## License

Copyright (c) 2025–2026 FacePhys Team.

FacePhys code, model weights, model architecture implementations, and derivatives are available under the [FacePhys Academic Community License](LICENSE).

- **Academic use only:** Use and modification are permitted solely for non-commercial research related to academic papers, with citation of the FacePhys paper.
- **Authorization for deployment:** Integrating FacePhys into applications and services through [Open-rPPG](https://github.com/KegangWangCCNU/open-rppg) framework or [FacePhys Demo](https://github.com/KegangWangCCNU/FacePhys-Demo) requires prior written authorization.
- **Commercial licensing:** Commercial use requires a separate license. A commercial license must be required when cumulative users exceed **1,000**, or combined monthly revenue of the operating entity, its parent company, and affiliates exceeds **RMB 10,000** (approximately **USD 1,428**).
- **Logo attribution:** Any use of the weights or their derivatives must visibly display the official FacePhys logo.
  
<img width="21752" height="5552" alt="image" src="https://github.com/user-attachments/assets/f5f7529c-6242-4e48-94a2-b48b2fba34f2" />  

 See [LICENSE](LICENSE) for the full terms and apply at [facephys.com](https://facephys.com).

FacePhys was trained on the [RLAP dataset](https://github.com/KegangWangCCNU/RLAP-dataset), which is licensed separately under CC BY-NC-ND 4.0 and requires a signed Data Usage Agreement. This license grants no rights to the dataset itself.

FacePhys is a research model, **not a medical device**.
