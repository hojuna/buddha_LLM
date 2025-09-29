"""
Qwen/Qwen3-8B-Base를 HF Trainer로 학습하는 스크립트
- 입력: JSONL 한 줄당 {"text": "..."}
- 데이터: /home/work/paper/buddha/crawling/output/kabc_ABC_IT.translation.jsonl
- 로깅: Weights & Biases

예시 실행
python -m script.train.post_traing \
  --data_path /home/work/paper/buddha/crawling/output/kabc_ABC_IT.jsonl \
  --model_name Qwen/Qwen3-8B-Base \
  --output_dir /home/work/paper/buddha/outputs/qwen3-8b-base-kabc \
  --project buddha-train \
  --run_name qwen3-8b-base-kabc \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --block_size 2048


/home/work/miniconda3/envs/buddha/bin/python /home/work/paper/buddha/script/train/post_traing.py \
  --dataset_dir /home/work/paper/buddha/datasets/qwen3-8b-base-kabc \
  --model_name Qwen/Qwen3-8B-Base \
  --output_dir /home/work/paper/buddha/outputs/qwen3-8b-base-kabc \
  --project buddha-train \
  --run_name qwen3-8b-base-kabc \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --block_size 2048 \
  --bf16

"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

try:
    import wandb  # noqa: F401
except Exception:
    wandb = None  # type: ignore

import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

import bitsandbytes as bnb
from peft import LoraConfig, TaskType, get_peft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen/Qwen3-8B-Base on JSONL text")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/work/paper/buddha/crawling/output/kabc_ABC_IT.translation.jsonl",
        help="입력 JSONL 경로 (한 줄당 {'text': ...})",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B-Base",
        help="사전학습 언어모델 경로 또는 HF 허브 상의 모델 이름",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/work/paper/buddha/outputs/qwen3-8b-base-kabc",
        help="학습 산출물 저장 경로",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing",
                        action="store_true", default=True)
    parser.add_argument(
        "--project", type=str, default="buddha-train", help="W&B 프로젝트 이름"
    )
    parser.add_argument(
        "--run_name", type=str, default="qwen3-8b-base-kabc", help="W&B 런 이름"
    )
    parser.add_argument(
        "--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"],
    )
    parser.add_argument(
        "--eval_strategy", type=str, default="no", choices=["no", "steps", "epoch"],
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1, help=">0 이면 epochs 대신 max steps 우선"
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="체크포인트 경로"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="사전 전처리된 데이터셋(`save_to_disk`) 경로. 지정 시 해당 경로에서 로드",
    )
    # LoRA options
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="LoRA 어댑터를 사용하여 미세조정")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank r")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="콤마로 구분된 LoRA 대상 모듈 이름들"
    )
    return parser.parse_args()


def ensure_tokenizer_padding_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # 최후 수단: 특수 토큰 추가
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,

    )
    ensure_tokenizer_padding_token(tokenizer)
    # Causal LM 학습에서는 대체로 좌/우 패딩 모두 가능. 안전하게 우측 패딩.
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass
    return tokenizer


def load_model(model_name: str, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    # 가능하면 Flash-Attention2 사용
    # if hasattr(model.config, "use_flash_attention_2"):
    #     try:
    #         model.config.use_flash_attention_2 = True
    #     except Exception:
    #         pass
    return model


def build_datasets(
    data_path: str,
    tokenizer: AutoTokenizer,
    block_size: int,
    num_proc: int | None = None,
):
    # JSONL: 각 줄은 {"text": "..."}
    dataset = load_dataset("json", data_files=data_path, split="train")

    def filter_valid(example: Dict[str, str]) -> bool:
        txt = example.get("translation")
        if txt is None:
            return False
        if not isinstance(txt, str):
            return False
        return len(txt.strip()) > 0

    dataset = dataset.filter(filter_valid, num_proc=num_proc)

    def tokenize_function(examples: Dict[str, List[str]]):
        return tokenizer(
            examples["translation"],
            add_special_tokens=True,
            return_attention_mask=True,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        # 원본(예: 'translation')을 포함한 모든 기존 컬럼 제거 → 토크나이즈 결과만 유지
        remove_columns=list(dataset.column_names),
        num_proc=num_proc,
        desc="Tokenizing",
    )

    def group_texts(examples: Dict[str, List[List[int]]]):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        return result

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        desc=f"Grouping into blocks of {block_size}",
    )

    return lm_dataset


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # Tokenizer / Model
    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, tokenizer)

    # Apply LoRA if requested
    if getattr(args, "use_lora", False):
        target_modules = [m.strip()
                          for m in args.lora_target_modules.split(",") if m.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Dataset
    if args.dataset_dir is not None and os.path.isdir(args.dataset_dir):
        print(f"[data] Loading preprocessed dataset from: {args.dataset_dir}")
        train_dataset = load_from_disk(args.dataset_dir)
    else:
        train_dataset = build_datasets(
            data_path=args.data_path,
            tokenizer=tokenizer,
            block_size=args.block_size,
            num_proc=max(1, args.dataloader_num_workers //
                         2) if args.dataloader_num_workers else None,
        )

    # Collator: Causal LM
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # W&B
    report_to = ["wandb"]
    if wandb is None:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = []

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs if args.max_steps <= 0 else 1.0,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=report_to,
        run_name=args.run_name,
        # evaluation_strategy=args.eval_strategy,
        remove_unused_columns=False,
    )
    adam_optimizer = bnb.optim.Adam8bit(
        model.parameters(), lr=args.learning_rate)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        optimizers=(adam_optimizer, None),
    )

    # 학습
    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model()

    # Save LoRA adapters explicitly if used
    try:
        if getattr(args, "use_lora", False) and hasattr(model, "save_pretrained"):
            model.save_pretrained(args.output_dir)
    except Exception:
        pass

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 토크나이저도 함께 저장
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
