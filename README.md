# Medical-Factory

**Medical-Factory** is an adaptation of the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository, tailored specifically for medical domain training workflows, including continued pretraining, supervised fine-tuning, and reinforcement learning methods.

## Key Features
- Adapted for medical language model training.
- Seamless integration with LoRA, QLoRA, and other efficient finetuning methods.
- Preconfigured for workflows like continued pretraining, supervised fine-tuning (SFT), and direct preference optimization (DPO).

---

## Requirements

### Software

| Dependency      | Minimum | Recommended |
|------------------|---------|-------------|
| **Python**       | 3.8     | 3.10        |
| **Torch**        | 1.13.1  | 2.2.0       |
| **Transformers** | 4.37.2  | 4.39.1      |
| **Datasets**     | 2.14.3  | 2.17.1      |
| **Accelerate**   | 0.27.2  | 0.28.0      |
| **PEFT**         | 0.9.0   | 0.10.0      |
| **TRL**          | 0.8.1   | 0.8.1       |

### Optional

| Dependency      | Minimum | Recommended |
|------------------|---------|-------------|
| **CUDA**         | 11.6    | 12.2        |
| **DeepSpeed**    | 0.10.0  | 0.14.0      |
| **BitsAndBytes** | 0.39.0  | 0.43.0      |
| **Flash-Attn**   | 2.3.0   | 2.5.6       |

---

## Hardware Requirements (Estimated)

| Method | Bits | 7B   | 13B  | 30B  | 70B   | 8x7B  |
|--------|------|-------|------|------|-------|-------|
| **Full**   | 16   | 60GB  | 120GB | 300GB | 600GB  | 400GB |
| **LoRA**   | 16   | 16GB  | 32GB  | 64GB  | 160GB  | 120GB |
| **QLoRA**  | 8    | 10GB  | 20GB  | 40GB  | 80GB   | 60GB  |
| **QLoRA**  | 4    | 6GB   | 12GB  | 24GB  | 48GB   | 30GB  |

---

## Getting Started

### Installation

```bash
git clone https://github.com/emrecanacikgoz/Medical-Factory
conda create -n medical_factory python=3.10
conda activate medical_factory
cd Medical-Factory
pip install -r requirements.txt
```

### Datasets

Datasets used in continued pretraining, fine-tuning, and preference learning are available upon request. Please request access via [this link](https://drive.google.com/drive/folders/11tX_nv8dxkZzt9NAMr2t7kiJ1YPvOZUl?usp=sharing).

---

## Training Workflows

### Continued Pretraining

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage pt \
    --do_train \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --dataset path_to_dataset \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_pt_checkpoint \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --fp16
```

### Supervised Fine-Tuning

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --dataset path_to_dataset \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_sft_checkpoint \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --fp16
```

### Direct Preference Optimization (DPO)

```bash
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --adapter_name_or_path path_to_sft_checkpoint \
    --create_new_adapter \
    --dataset path_to_dataset \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_dpo_checkpoint \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --fp16
```

### Exporting Model

```bash
CUDA_VISIBLE_DEVICES= python src/export_model.py \
    --model_name_or_path path_to_llama_model \
    --adapter_name_or_path path_to_checkpoint \
    --template default \
    --finetuning_type lora \
    --export_dir path_to_export \
    --export_size 2 \
    --export_legacy_format False
```

---

## Citation

If you find this repository helpful, please cite:

```bibtex
@article{acikgoz2024hippocrates,
  title={Hippocrates: An Open-Source Framework for Advancing Large Language Models in Healthcare},
  author={Acikgoz, Emre Can and {\.I}nce, Osman Batur and Bench, Rayene and Boz, Arda An{\i}l and Kesen, {\.I}lker and Erdem, Aykut and Erdem, Erkut},
  journal={arXiv preprint arXiv:2404.16621},
  year={2024}
}
```

---

## Acknowledgment

This repository is adapted from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). We appreciate their contribution to the open science community.
