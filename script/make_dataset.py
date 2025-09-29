"""
JSONL 원본을 전처리 → 토크나이즈 → 블록 그루핑까지 수행하고
`datasets` 포맷으로 `save_to_disk` 저장하는 스크립트.

예시 실행
python /home/work/paper/buddha/script/make_dataset.py \
  --input_jsonl /home/work/paper/buddha/crawling/output/kabc_ABC_IT.jsonl \
  --text_column translation \
  --model_name Qwen/Qwen3-8B-Base \
  --block_size 2048 \
  --output_dir /home/work/paper/buddha/datasets/qwen3-8b-base-kabc
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess JSONL into tokenized/grouped dataset and save to disk"
    )
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="입력 JSON/JSONL 경로")
    parser.add_argument("--text_column", type=str, default="translation",
                        help="텍스트가 들어있는 컬럼명")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B-Base",
                        help="토크나이저 모델 이름/경로")
    parser.add_argument("--block_size", type=int, default=1024,
                        help="그루핑 블록 크기")
    parser.add_argument("--num_proc", type=int, default=4,
                        help="병렬 전처리 프로세스 수")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="`save_to_disk` 결과 저장 경로")
    return parser.parse_args()


def ensure_tokenizer_padding_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset("json", data_files=args.input_jsonl, split="train")

    # 1) 텍스트 정리: [내용영역] 제거
    def clean_text(example: Dict[str, str]) -> Dict[str, str]:
        text = example.get(args.text_column)
        if isinstance(text, str):
            text = text.replace("[내용영역]", "")
        example[args.text_column] = text
        return example

    dataset = dataset.map(clean_text, desc="Cleaning text: remove [내용영역]")

    # 2) 유효성 필터
    def filter_valid(example: Dict[str, str]) -> bool:
        txt = example.get(args.text_column)
        if txt is None:
            return False
        if not isinstance(txt, str):
            return False
        return len(txt.strip()) > 0

    dataset = dataset.filter(filter_valid, num_proc=args.num_proc,
                             desc="Filter valid rows")

    # 3) 토크나이저 준비
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True
    )
    ensure_tokenizer_padding_token(tokenizer)
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass

    # 4) 토크나이즈
    def tokenize_function(examples: Dict[str, List[str]]):
        return tokenizer(
            examples[args.text_column],
            add_special_tokens=True,
            return_attention_mask=True,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=list(dataset.column_names),
        num_proc=args.num_proc,
        desc="Tokenizing",
    )

    # 5) 그루핑
    def group_texts(examples: Dict[str, List[List[int]]]):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        if total_length >= args.block_size:
            total_length = (total_length // args.block_size) * args.block_size
        result = {
            k: [t[i: i + args.block_size]
                for i in range(0, total_length, args.block_size)]
            for k, t in concatenated.items()
        }
        return result

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        desc=f"Grouping into blocks of {args.block_size}",
    )

    # 6) 저장
    lm_dataset.save_to_disk(args.output_dir)
    print(f"Saved dataset to: {args.output_dir}")
    print(lm_dataset)


if __name__ == "__main__":
    main()
