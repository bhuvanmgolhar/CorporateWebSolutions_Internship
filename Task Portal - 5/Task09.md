# Task 09 — Model Training at Scale, Distributed ML (DeepSpeed, Megatron, FSDP), Hyperparameter Optimization (Optuna, Ray Tune) & Experiment Tracking (MLflow, WandB)

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal V |
| Task Number | 09 |
| Topic | Model Training Engineering — Distributed Training (FSDP, DeepSpeed ZeRO, Megatron-LM), Distributed HPO (Optuna, Ray Tune) & Enterprise Experiment Tracking (MLflow, WandB) |
| Task Type | Technical Core & Deep Learning Systems Engineering |
| Status | Completed |
| Repository Section | `tasks/portal-05/task-09/` |

---

## 2. Objective

The objective of this task is to design, implement, and benchmark enterprise-grade **Distributed Model Training Systems, Automated Hyperparameter Optimization (HPO) Frameworks, and Experiment Governance Architectures** for large-scale machine learning models and Multi-Billion Parameter Deep Learning / Foundation Models.
This task focuses on:
- Architecting distributed training paradigms: Data Parallelism (DDP), Tensor Parallelism (TP), Pipeline Parallelism (PP), and Zero Redundancy Optimizer (ZeRO-1, ZeRO-2, ZeRO-3 via DeepSpeed & PyTorch FSDP).
- Scaling hyperparameter search pipelines using Ray Tune, Optuna (Asynchronous Successive Halving Algorithm - ASHA, Tree-structured Parzen Estimator - TPE), and Bayesian Optimization across multi-GPU clusters.
- Establishing enterprise-grade ML experiment tracking, artifact lineage, and metadata logging architectures using MLflow, Weights & Biases (WandB), and Neptune.ai.
- Managing mixed-precision computation (FP32, FP16, BF16, FP8) and memory footprint optimizations (Gradient Checkpointing, Activation Offloading, FlashAttention-2).
- Benchmarking GPU utilization (TFLOPS efficiency, Model Flops Utilization - MFU), memory allocation, and communication overhead (All-Reduce, All-Gather, Reduce-Scatter) across distributed cluster topologies.

---

## 3. Introduction

As machine learning models scale to billions of parameters, single-GPU training memory limits become a primary bottleneck. Large Language Models (LLMs) and deep neural networks require vast amounts of memory for parameters, gradients, optimizer states, and activations.

```text
                  Distributed ML & Scaling Ecosystem
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Distributed Data │ ───► │ Distributed      │ ───► │ Model Training   │
│ & Multi-GPU Pool │      │ Engine (FSDP/ZeRO│      │ Loop & Mixed Prec│
└──────────────────┘      └──────────────────┘      └────────┬─────────┘
                                                             │
┌──────────────────┐      ┌──────────────────┐               │
│ Enterprise ML    │ ◄─── │ Distributed HPO  │ ◄─────────────┘
│ Registry (MLflow)│      │ Engine (Ray Tune)│
└──────────────────┘      └──────────────────┘

```

Training these architectures requires distributed deep learning frameworks that partition parameters and computations across compute nodes. At the same time, maintaining control over training runs demands systematic hyperparameter search algorithms and detailed experiment tracking systems.
The core operating principle for large-scale model training is:

> **Distributed training scales efficiently only when memory partitioning, compute-to-communication overlaps, and systematic hyperparameter tuning eliminate resource bottlenecks.**

---

## 4. Distributed Parallelism Paradigms & ZeRO Taxonomy

Training models exceeding single-GPU VRAM capacity requires combining data, tensor, and pipeline parallelism methods.

```text
                  Distributed Training Paradigms & Parallelism
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ Parallelism Strategy                  │ Execution & Partitioning Mechanics    │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Distributed Data Parallel (DDP)       │ Replicates full model across GPUs;   │
│                                       │ splits micro-batches. Synchronizes    │
│                                       │ gradients via Ring All-Reduce.        │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Tensor Parallelism (Megatron-LM TP)   │ Splits individual layer weight matrices│
│                                       │ (e.g., Column/Row Parallel Linear)    │
│                                       │ across GPUs within a single node.     │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Pipeline Parallelism (PP)             │ Partitions sequential layers across   │
│                                       │ nodes; pipelines micro-batches via    │
│                                       │ 1F1B (One Forward, One Backward) schedule.│
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Fully Sharded Data Parallel (FSDP) /  │ Shards Optimizer States (ZeRO-1),     │
│ DeepSpeed ZeRO Stage 1-3              │ Gradients (ZeRO-2), and Parameters    │
│                                       │ (ZeRO-3) across data parallel workers.│
└───────────────────────────────────────┴───────────────────────────────────────┘

```

---

## 5. Mathematical & Algorithmic Foundations

Understanding GPU memory consumption and distributed communication primitives is essential for sizing training clusters.

### 5.1 Deep Learning Memory Allocation Analysis

For a model with $\Psi$ parameters trained using Adam optimizer in Mixed-Precision (FP16/BF16 weights, FP32 master weights/optimizer states), static memory consumption scales as:

$$M_{\text{static}} = M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer}}$$

$$M_{\text{static}} = 2\Psi \, \text{bytes (FP16 weights)} + 2\Psi \, \text{bytes (FP16 gradients)} + \underbrace{\left(4\Psi + 4\Psi + 4\Psi\right)}_{\text{FP32 Master Weights + Momentum + Variance}} \, \text{bytes}$$

$$M_{\text{static}} = 16\Psi \, \text{bytes}$$

*Example:* A $10\text{B}$ parameter model requires $160\text{GB}$ of static VRAM before factoring in activations ($M_{\text{activations}}$) or temporary buffers.

---

### 5.2 DeepSpeed ZeRO Memory Reduction Formulations

ZeRO distributes static memory across $N_{\text{d}}$ data-parallel processes:

$$\text{Memory}_{\text{ZeRO-1}} = 2\Psi + 2\Psi + \frac{12\Psi}{N_{\text{d}}} \quad \text{(Shards Optimizer States)}$$

$$\text{Memory}_{\text{ZeRO-2}} = 2\Psi + \frac{2\Psi + 12\Psi}{N_{\text{d}}} = 2\Psi + \frac{14\Psi}{N_{\text{d}}} \quad \text{(Shards Gradients \& Optimizer)}$$

$$\text{Memory}_{\text{ZeRO-3}} = \frac{2\Psi + 2\Psi + 12\Psi}{N_{\text{d}}} = \frac{16\Psi}{N_{\text{d}}} \quad \text{(Shards Parameters, Gradients \& Optimizer)}$$

```text
                        ZeRO Memory Footprint Comparison
Standard DDP:  [ Weights (2Ψ) ][ Gradients (2Ψ) ][ Adam States (12Ψ) ] = 16Ψ / GPU
ZeRO-1:        [ Weights (2Ψ) ][ Gradients (2Ψ) ][ Adam States / Nd  ]
ZeRO-2:        [ Weights (2Ψ) ][ Gradients / Nd ][ Adam States / Nd  ]
ZeRO-3:        [ Weights / Nd ][ Gradients / Nd ][ Adam States / Nd  ] = 16Ψ / (Nd * GPU)

```

---

### 5.3 Asynchronous Successive Halving Algorithm (ASHA) for HPO

ASHA speeds up hyperparameter searches by stopping underperforming trials early using a promotion rule based on reduction factor $\eta$ and minimum resource $r_{\text{min}}$:

$$r_k = r_{\text{min}} \cdot \eta^k$$

At evaluation rung $k$, only the top $1/\eta$ fraction of active trials are promoted to run for $r_{k+1}$ resource steps, freeing cluster resources for promising hyperparameter configurations.

---

## 6. Enterprise Model Training & Experiment Lineage Architecture

An enterprise ML lineage architecture records model artifacts, metrics, and parameters alongside distributed training loops.

```text
                     Enterprise ML Training Pipeline
┌─────────────────────────────────────────────────────────────────────────────┐
│ HYPERPARAMETER OPTIMIZATION ENGINE (Ray Tune / Optuna ASHA Scheduler)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DISTRIBUTED MODEL TRAINING CLUSTER                                         │
│ - PyTorch FSDP / DeepSpeed ZeRO-3 Engine                                    │
│ - Mixed Precision (BF16 / FP8) & FlashAttention-2 Execution                 │
│ - Activation Checkpointing & TensorBoard Logging                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────┴──────────────────────────────────────┐
│                                                                             │
▼                                                                             ▼
┌──────────────────────────────────────┐     ┌────────────────────────────────┐
│ EXPERIMENT METRIC TRACKER (WandB)    │     │ ENTERPRISE MODEL REGISTRY      │
│ - GPU utilization, loss, gradients   │     │ - MLflow Model Registry / S3   │
│ - Real-time training dashboards      │     │ - Lineage, dependencies, code  │
└──────────────────────────────────────┘     └────────────────────────────────┘

```

---

## 7. Mixed-Precision & Memory Efficiency Optimization Matrix

Optimizing memory allocation allows fitting larger batch sizes and accelerating GPU throughput.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MEMORY & COMPUTE EFFICIENCY ENGINE                      │
├──────────────────────────────┬──────────────────────────────┬───────────────┤
│ Precision Formats            │ Memory Offloading            │ Attention     │
│ - FP32 / FP16 / BF16 / FP8   │ - Gradient Checkpointing     │ - FlashAttention│
│ - Dynamic Loss Scaling       │ - CPU/NVMe Activation Offload│ - Memory-Aware│
└──────────────┬───────────────┴──────────────┬───────────────┴───────┬───────┘
               │                              │                       │
               └──────────────────────┬───────┴───────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MAXIMUM MFU & TFLOPS EFFICIENCY                        │
└─────────────────────────────────────────────────────────────────────────────┘

```

| Memory Technique | Primary Mechanics | VRAM Savings | Computational Impact |
| --- | --- | --- | --- |
| **Gradient Checkpointing** | Discards intermediate forward activations; recalculates them during backward pass. | $60\% - 80\%$ Activation VRAM Reduction | $\approx 20\% - 30\%$ Extra Compute Overhead |
| **BF16 Mixed Precision** | Uses 16-bit Brain Floating Point ($1$ sign, $8$ exponent, $7$ mantissa). | $50\%$ Weight/Gradient Memory | $2\times - 4\times$ Speedup on Tensor Cores |
| **FlashAttention-2** | Reorders attention math into fused GPU SRAM tiles without writing full $N \times N$ matrices to HBM. | $\mathcal{O}(N^2) \rightarrow \mathcal{O}(N)$ Memory | $2\times - 3\times$ Speedup in Attention Layers |

---

## 8. Technology & Integration Matrix

| Functional Area | Industry Standard Tooling | Primary Technical Function |
| --- | --- | --- |
| **Distributed ML Engines** | PyTorch FSDP, DeepSpeed, Megatron-LM | Executes data, tensor, pipeline, and sharded parameter parallelism across GPU clusters. |
| **Distributed HPO Frameworks** | Ray Tune, Optuna | Automates hyperparameter exploration using ASHA and TPE samplers across multi-GPU pools. |
| **Experiment Tracking** | Weights & Biases (WandB), MLflow, Neptune | Logs real-time metrics, system GPU utilization, loss curves, and artifact lineage. |
| **Model Registry & Governance** | MLflow Model Registry, AWS SageMaker Registry | Manages versioned model artifacts, staging-to-production transitions, and model governance. |

---

## 9. Personal Understanding

Task 09 highlighted that scaling large machine learning models requires treating compute resources, memory limits, and networking as an integrated system.
I now see that distributed training involves making trade-offs between memory footprints, GPU floating-point performance, and inter-node communication overhead, rather than simply pooling hardware.
Using tools like PyTorch FSDP, DeepSpeed ZeRO-3, Ray Tune, and MLflow makes it possible to train multi-billion parameter models with traceable metrics and reproducible artifacts.
The central principle remains:

> **Distributed training scales efficiently only when memory partitioning, compute-to-communication overlaps, and systematic hyperparameter tuning eliminate resource bottlenecks.**

---

## 10. Interview / Viva Questions

### Q1. What is the fundamental difference between PyTorch DDP and PyTorch FSDP?

**Answer:**

PyTorch DDP replicates the entire model, gradients, and optimizer states across every worker GPU, which limits maximum model size to single-GPU VRAM capacity. FSDP (Fully Sharded Data Parallel) shards model parameters, gradients, and optimizer states across data-parallel GPUs, reducing per-GPU memory consumption and allowing training of much larger models.

### Q2. How do DeepSpeed ZeRO Stage 1, Stage 2, and Stage 3 differ in VRAM memory management?

**Answer:**

* **ZeRO-1:** Shards Adam optimizer states across data-parallel workers ($4\times$ memory reduction).
* **ZeRO-2:** Shards both optimizer states and gradients ($8\times$ memory reduction).
* **ZeRO-3:** Shards optimizer states, gradients, and model parameters across all workers, gathering parameters on demand during forward/backward passes.

### Q3. Why is BF16 preferred over FP16 when training large deep learning architectures?

**Answer:**

BF16 (Brain Floating Point 16) allocates 8 exponent bits—the same as FP32—providing a wider dynamic range than FP16 (which has 5 exponent bits). This wider range prevents underflow and overflow issues during training, eliminating the need for dynamic loss scaling.

### Q4. How does Tensor Parallelism (Megatron-LM TP) split matrix operations within a linear layer?

**Answer:**

Tensor Parallelism splits individual weight matrices across GPUs within a node. In Column Parallel Linear layers, the weight matrix $W$ is split along columns ($W = [W_1, W_2]$) to calculate outputs in parallel. Row Parallel Linear layers split weights along rows and combine partial sums using an All-Reduce sum operation.

### Q5. What is the core mechanism of the Asynchronous Successive Halving Algorithm (ASHA) in Ray Tune?

**Answer:**

ASHA evaluates trials asynchronously at defined progress checkpoints (rungs). It stops underperforming hyperparameter configurations early and promotes only the top $1/\eta$ fraction of trials to higher rungs, saving GPU compute cycles.

### Q6. How does FlashAttention-2 speed up attention computations while saving VRAM?

**Answer:**

Standard attention materializes intermediate $N \times N$ attention matrix maps in GPU High Bandwidth Memory (HBM). FlashAttention-2 tiles softmax calculations directly inside fast GPU SRAM cache, avoiding intermediate HBM reads/writes and reducing memory complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$.

### Q7. What causes pipeline bubble overhead in Pipeline Parallelism, and how does the 1F1B schedule help?

**Answer:**

Pipeline bubbles are idle GPU wait states caused by sequential dependencies across pipeline stages. The 1F1B (One Forward, One Backward) schedule alternates forward and backward micro-batch passes on each stage once filled, keeping GPUs active and reducing memory footprint.

### Q8. What is the primary function of an MLflow Model Registry in production MLOps pipelines?

**Answer:**

The MLflow Model Registry provides a centralized store for managing a model's lifecycle. It tracks version history, records training parameters and code commits, manages stage transitions (e.g., Staging to Production), and enforces approval workflows before deployment.

### Q9. How does Activation Checkpointing (Gradient Checkpointing) reduce GPU VRAM usage during training?

**Answer:**

Activation Checkpointing discards intermediate layer activations during the forward pass instead of storing them all in VRAM. During the backward pass, it recomputes those activations on demand, trading a $\approx 20\%$ increase in compute time for a $60\%-80\%$ reduction in activation memory.

### Q10. What is Model FLOPs Utilization (MFU), and why is it a useful efficiency metric for LLM training?

**Answer:**

MFU measures the ratio of observed floating-point throughput to the theoretical peak TFLOPS of the target GPU hardware. It quantifies how efficiently a distributed training setup utilizes available GPU compute, taking into account communication and memory overheads.

### Q11. How does Tree-structured Parzen Estimator (TPE) differ from random hyperparameter search in Optuna?

**Answer:**

Random search samples hyperparameter values independently from prior distributions. TPE builds a probabilistic model using historical trial evaluations, splitting results into good and bad groups to sample parameter configurations that are more likely to improve model performance.

### Q12. What communication primitive is used to synchronize gradients in standard Distributed Data Parallel (DDP)?

**Answer:**

DDP uses the **Ring All-Reduce** primitive. GPUs are organized in a logical ring, passing gradient chunks sequentially around the ring to calculate average gradients across all workers with minimal network overhead.

### Q13. Why are FP32 master weights maintained when training models in mixed precision (FP16/BF16)?

**Answer:**

Gradient updates computed in lower precision (FP16/BF16) can be smaller than what 16-bit floating-point numbers can represent. Accumulating updates into an FP32 master weight copy prevents small gradient steps from rounding to zero, preserving convergence stability.

### Q14. What is the operational difference between parameter logging and artifact logging in Weights & Biases (WandB)?

**Answer:**

Parameter logging records scalar values, hyperparameter configs, and evaluation metrics (e.g., loss, learning rate, GPU temperature) over training steps. Artifact logging tracks versioned binary assets, such as dataset snapshots, model weights, and evaluation visualizations.

### Q15. How does CPU/NVMe offloading in DeepSpeed ZeRO-Offload enable training larger models on limited GPU resources?

**Answer:**

ZeRO-Offload moves Adam optimizer states and parameter update computations from GPU VRAM to system host RAM or NVMe storage. The GPU processes forward/backward compute passes, while host CPU memory handles optimizer steps, extending memory capacity beyond physical GPU limits.

---

## 11. Conclusion

Task 09 outlines an architecture for running large-scale model training pipelines, distributed hyperparameter optimization, and experiment tracking systems.
The complete operational flow is summarized below:

```text
Enterprise Large-Scale Training Lifecycle
      ↓
Cluster Compute Initialization & Mixed-Precision Config (BF16 / FP8)
      ↓
Distributed Execution Engine (PyTorch FSDP / DeepSpeed ZeRO-3)
      ↓
Automated Distributed HPO Search (Ray Tune ASHA / Optuna TPE)
      ↓
Experiment Metric & Lineage Tracking (WandB / MLflow Registry)
      ↓
Production Model Artifact Registration & Staging Validation

```

The core pillars of distributed model training include:

```text
Model Training & Engineering Framework
├── Distributed Parallelism (FSDP, ZeRO-3, Tensor Parallelism, DDP)
├── Compute & Memory Efficiency (FlashAttention-2, BF16, Checkpointing)
├── Distributed HPO (Ray Tune ASHA, Optuna TPE Bayesian Samplers)
└── Experiment Governance (MLflow Registry, WandB Tracking & Lineage)

```

Core tools and operational frameworks:

```text
PyTorch FSDP / DeepSpeed / Megatron-LM
Ray Tune / Optuna
Weights & Biases (WandB) / MLflow
FlashAttention-2 / NVIDIA NCCL

```

By using distributed parallelism frameworks, memory optimizations, and structured experiment tracking, teams can efficiently train multi-billion parameter models on modern GPU clusters.
The central principle remains:

> **Distributed training scales efficiently only when memory partitioning, compute-to-communication overlaps, and systematic hyperparameter tuning eliminate resource bottlenecks.**

---

## 12. Key Takeaways

1. Distributed model training scales across multi-GPU setups by balancing compute operations, VRAM capacity, and network communication overhead.
2. **PyTorch DDP** replicates the full model across all workers, limiting maximum model size to single-GPU VRAM limits.
3. **PyTorch FSDP** and **DeepSpeed ZeRO-3** shard optimizer states, gradients, and parameters across data-parallel workers, enabling training of multi-billion parameter models.
4. Static Adam optimizer memory requires $16\times$ model parameter count ($16\Psi$ bytes) in standard FP32/FP16 mixed precision.
5. **BF16 mixed precision** provides a wide dynamic range (8 exponent bits) that prevents numerical underflow without requiring dynamic loss scaling.
6. **Tensor Parallelism (Megatron-LM)** splits individual weight matrices across GPUs within a node to parallelize large layer computations.
7. **Pipeline Parallelism** divides sequential model layers across nodes, using 1F1B micro-batch schedules to minimize pipeline bubbles.
8. **FlashAttention-2** computes attention inside fast GPU SRAM cache, saving VRAM and speeding up long-sequence attention layers.
9. **Activation Checkpointing** trades a $\approx 20\%$ increase in recomputation time for a $60\%-80\%$ reduction in activation VRAM usage.
10. **Ray Tune** and **Optuna** automate hyperparameter searches using algorithms like ASHA to prune underperforming runs early.
11. **ASHA** accelerates HPO searches by stopping low-performing configurations early at predefined resource rungs.
12. **Tree-structured Parzen Estimator (TPE)** builds probabilistic models of hyperparameter performance to focus sampling on promising parameter spaces.
13. **MLflow** and **Weights & Biases (WandB)** track real-time training metrics, hardware utilization, loss curves, and artifact lineage.
14. The **MLflow Model Registry** manages stage transitions, model versioning, and approval workflows before production deployment.
15. **Ring All-Reduce** provides linear scaling for gradient synchronization in standard data-parallel training topologies.
16. **DeepSpeed ZeRO-Offload** moves optimizer states to host CPU RAM or NVMe storage to fit larger models on limited GPU resources.
17. Combining distributed training frameworks, memory-efficient operations, and systematic experiment tracking builds a solid foundation for large-scale enterprise AI workflows.
