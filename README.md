# Bridging the Pose-Semantic Gap: A Cascade Framework for Text-Based Person Anomaly Search

Official codebase for paper implementation of text-based person anomaly search.

Accepted by ACL 2026.

## Overview

SSDC addresses the pose-semantic gap in text-based person anomaly search with a two-stage pipeline:

1. Structure-aware coarse retrieval (SSDC stage-1 retriever).
2. Multi-round semantic verification and re-ranking with MLLMs.

This repository includes training, evaluation, and vLLM-based semantic refinement scripts.

## Highlights

- Coarse-to-fine retrieval for large-scale anomaly search.
- Pose-aware cross-modal backbone for fast candidate filtering.
- LLM-based detective-style multi-round semantic verification.
- Supports standard and variant scripts for SSDC + vLLM evaluation.

## Repository Structure

- `Search.py`: main training and evaluation entry for the SSDC stage-1 retriever.
- `run.py`: distributed launch wrapper for `Search.py`.
- `train.py`: training loop.
- `eval.py`: retrieval metrics and ITC/ITM evaluation.
- `evaluate.py`: standalone evaluation script.
- `vllm_infer_SSDC.py`: stage-2 semantic verification and re-ranking.
- `configs/ssdc.yaml`: SSDC stage-1 retriever configuration.
- `run.sh`, `evaluate.sh`, `run_stage2.sh`: example shell launch scripts.

## Environment Setup

### 1) Create environment

```bash
conda create -n ssdc python=3.10 -y
conda activate ssdc
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

For vLLM-based stage-2 inference, additionally install:

```bash
pip install vllm easydict ftfy qwen_vl_utils
```

## Data Preparation

The default config file is [configs/ssdc.yaml](configs/ssdc.yaml). Before running, update these paths to your local environment:

- `image_root`
- `train_file`
- `test_file`
- `vision_config`
- `text_config`

Dataset source note: the PAB dataset setup and related resources in this project are adapted with reference to the CMP repository: https://github.com/Shuyu-XJTU/CMP.

Current defaults in [configs/ssdc.yaml](configs/ssdc.yaml) point to local absolute paths from the original training machine.


## Training (Stage 1: SSDC Retriever)

Use [run.py](run.py) as the launcher:

```bash
python run.py \
  --task ssdc \
  --dist f4 \
  --output_dir output/ssdc
```

Notes:

- `--dist` options are defined in [run.py](run.py) (`f4`, `f2`, or `gpuX`).
- The default checkpoint in [run.py](run.py) is `checkpoint/16m_base_model_state_step_199999.th`.
- Training logs and checkpoints are saved under `--output_dir`.

## Evaluation (Stage 1)

Note: old `cmp` arguments are still accepted for backward compatibility.

### Option A: evaluate via launcher

```bash
python run.py \
  --task ssdc \
  --dist gpu0 \
  --output_dir evaluation_results/ssdc_eval \
  --checkpoint checkpoint/best.pth \
  --evaluate
```

### Option B: standalone evaluator

```bash
python evaluate.py \
  --config configs/ssdc.yaml \
  --task ssdc \
  --output_dir evaluation_results/run_01 \
  --checkpoint checkpoint/best.pth
```

Evaluation metrics include R1, R5, R10, mAP, and mINP.

## Semantic Verification and Re-ranking (Stage 2)

After SSDC Stage-1 retriever is ready, run vLLM-based re-ranking:

```bash
python vllm_infer_SSDC.py \
  --xi 0.1 \
  --lambda 0.4 \
  --embed_model_path checkpoint \
  --source PAB \
  --target PAB \
  --ssdc_checkpoint checkpoint/best.pth \
  --ssdc_config configs/ssdc.yaml \
  --model_dir /path/to/Qwen3-VL-8B-sft \
  --base_model RDE \
  --root_dir /path/to/data \
  --tag test
```

You can also use the stage-2 shell template:

- [run_stage2.sh](run_stage2.sh)

## Visuals

- Framework: [assets/framework.jpg](assets/framework.jpg)
- Dataset examples: [assets/dataset.jpg](assets/dataset.jpg)
- Qualitative examples: [assets/example.jpg](assets/example.jpg)
- Comparison figure: [assets/comparison.JPG](assets/comparison.JPG)

## Acknowledgment

The dataset organization and baseline preparation in this repository reference the public CMP project: https://github.com/Shuyu-XJTU/CMP.

## Reproducibility Notes

- Set all dataset/model paths before running.
- Make sure GPU visibility matches your scripts (`CUDA_VISIBLE_DEVICES`).
- For distributed launch, ensure `MASTER_PORT` in [run.py](run.py) is available.
- Stage-2 scripts may cache intermediate CMP features to speed up repeated runs.

## Citation

If you use this repository, please cite the corresponding paper:

```bibtex
@inproceedings{ssdc2026,
  title={Bridging the Pose-Semantic Gap: A Cascade Framework for Text-Based Person Anomaly Search},
  author={Anonymous},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}
```

## License

This project is released under the license in [LICENSE](LICENSE).
