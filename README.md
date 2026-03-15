# GPU Parallelism Learning Project

A hands-on learning project to understand GPU parallelism with simple examples and detailed concept explorations.

## Setup

```bash
# Requires Python 3.12+
uv sync
uv run python single_worker_run.py
```

## Topics Covered

**Fundamentals**
- Parameter counting
- Memory breakdown
- Collective communication operations
- Mixed precision training
- Activation recomputation and gradient accumulation
- Memory math with ZeRO

**Parallelism Strategies**
- Data parallelism (DDP, ZeRO-1/2/3, FSDP)
- Tensor and sequence parallelism
- Context parallelism and ring attention
- Pipeline parallelism
- Expert / MoE parallelism

## Viewing Notes

Each concept has detailed notes and diagrams in `.excalidraw` format. To view them:

1. Install the [Excalidraw VS Code extension](https://marketplace.visualstudio.com/items?itemName=pomdtr.excalidraw-editor)
2. Open any `.excalidraw` file in the project to view interactive diagrams

## Acknowledgments

Many theoretical concepts are taken from **Raj Abhijit Dandekar's parallelism notes**. Thank you!
