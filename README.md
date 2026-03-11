# GPU Parallelism Roadmap

A hands-on learning roadmap for mastering distributed GPU training — from fundamentals to full parallelism implementations in PyTorch.

## How to use this repo

- **Code examples** are in numbered directories (`1_data_parallelism/`, `2_tensor_seq_parallelism/`, etc.) with runnable PyTorch scripts.
- **Theory & diagrams** use `.excalidraw` files for visual explanations and `.ipynb` notebooks for interactive derivations.
- **Baseline model**: All examples use the same tiny 2-layer Transformer defined in [single_worker_run.py](single_worker_run.py) (520 parameters) so you can compare across parallelism strategies.

### Viewing Excalidraw diagrams

The `.excalidraw` files contain visual diagrams for theoretical concepts. To view and edit them:

1. Install the [Excalidraw extension](https://marketplace.visualstudio.com/items?itemName=pomdtr.excalidraw-editor) in VS Code
2. Open any `.excalidraw` file — it renders as an interactive whiteboard
3. Exported `.png` versions are in the `images/` directory for quick reference

## Roadmap

### Fundamentals

1. [Parameter counting](1_parameter_counting/) — derive formula by hand ([diagram](1_parameter_counting/1_parameter_counting.excalidraw) | [notebook](1_parameter_counting/parameter_counting.ipynb))
2. Memory breakdown — params, gradients, optimizer states, activations
3. Collective communication operations — all-reduce, all-gather, reduce-scatter, broadcast, all-to-all
4. Mixed precision training
5. Activation recomputation and gradient accumulation
6. Memory math with ZeRO — calculate per-GPU memory for each stage

### Parallelism strategies

7. [Data parallelism](1_data_parallelism/) — DDP, ZeRO-1/2/3, FSDP — implement in PyTorch
8. [Tensor parallelism and sequence parallelism](2_tensor_seq_parallelism/) — implement in PyTorch
9. [Context parallelism](3_context_parallelism/) and ring attention — implement in PyTorch
10. [Pipeline parallelism](4_pipeline_parallelism/) — implement in PyTorch
11. [Expert / MoE parallelism](5_expert_parallelism/) — implement in PyTorch

### Putting it together

12. Understand at which level each parallelism is applied
13. Train a small language model and apply each type of parallelism
14. Learn frameworks — DeepSpeed, Megatron-LM, HuggingFace Accelerate

## Project setup

```bash
# Requires Python 3.12+
uv sync        # install dependencies
uv run python single_worker_run.py   # run the baseline model
```

## Repository structure

```
├── single_worker_run.py          # Baseline single-process training (no parallelism)
├── 1_parameter_counting/         # Theory: parameter counting derivation
│   ├── 1_parameter_counting.excalidraw
│   └── parameter_counting.ipynb
├── 1_data_parallelism/           # DDP, ZeRO-1/2/3 implementations
│   ├── dp_no_zero.py
│   ├── dp_zero1.py
│   ├── dp_zero2.py
│   └── dp_zero3.py
├── 2_tensor_seq_parallelism/     # Tensor parallelism implementation
│   ├── tp_sq_implementation.py
│   ├── learnings.md
│   └── flow_diagram.md
├── 3_context_parallelism/        # (coming soon)
├── 4_pipeline_parallelism/       # (coming soon)
├── 5_expert_parallelism/         # (coming soon)
└── images/                       # Exported diagram PNGs
```
