# 🧘 Buddha LLM

불교 문헌 데이터 기반 한국어 대규모 언어모델 학습 프로젝트

## 📋 프로젝트 개요

Buddha LLM은 한국 불교 문헌 데이터베이스에서 텍스트를 수집하고, 이를 활용하여 Qwen3-8B 기반 언어모델을 학습시키는 프로젝트입니다. 동국대학교 한국불교전서(KABC) 데이터베이스에서 불교 문헌을 크롤링하여 고품질의 한국어 불교 전문 언어모델을 구축합니다.

### 주요 기능

- **데이터 수집**: 동국대학교 한국불교전서(KABC)에서 불교 문헌 크롤링
- **데이터 전처리**: JSONL 형식의 데이터를 토큰화 및 블록 그루핑 처리
- **모델 학습**: Qwen3-8B-Base 모델을 활용한 Causal Language Modeling
- **LoRA 지원**: 메모리 효율적인 파인튜닝을 위한 LoRA 어댑터 지원

## 🗂️ 프로젝트 구조

```
buddha_LLM/
├── crawling/
│   ├── crawling_IT.py      # 일체경 번역문(ABC_IT) 크롤링 스크립트
│   └── crawling_BJ.py      # 불교저술(ABC_BJ) 크롤링 스크립트
├── script/
│   ├── make_dataset.py     # 데이터셋 전처리 스크립트
│   └── train/
│       └── post_traing.py  # 모델 학습 스크립트
└── README.md
```

## 🔧 설치 방법

### 필요 라이브러리

```bash
pip install requests beautifulsoup4 urllib3
pip install datasets transformers torch
pip install wandb  # 학습 로깅용 (선택사항)
pip install peft bitsandbytes  # LoRA 학습용 (선택사항)
pip install flash-attn  # Flash Attention 2 지원용 (선택사항)
```

## 📖 사용 방법

### 1. 데이터 크롤링

#### 일체경 번역문(ABC_IT) 크롤링

```bash
python crawling/crawling_IT.py \
  --output kabc_ABC_IT.jsonl \
  --max_workers 8
```

#### 불교저술(ABC_BJ) 크롤링

```bash
python crawling/crawling_BJ.py \
  --output kabc_ABC_BJ.jsonl \
  --max_workers 8
```

**주요 옵션:**
- `--output`: 결과를 저장할 JSONL 파일 경로
- `--max_workers`: 병렬 처리 워커 수 (기본값: 8)
- `--delay`: 요청 간 대기 시간 (초)

### 2. 데이터셋 전처리

크롤링한 JSONL 파일을 토크나이즈하고 블록 단위로 그루핑합니다:

```bash
python script/make_dataset.py \
  --input_jsonl ./crawling/output/kabc_ABC_IT.jsonl \
  --text_column translation \
  --model_name Qwen/Qwen3-8B-Base \
  --block_size 2048 \
  --output_dir ./datasets/qwen3-8b-base-kabc
```

**주요 옵션:**
- `--input_jsonl`: 입력 JSONL 파일 경로
- `--text_column`: 텍스트가 포함된 컬럼명 (기본값: "translation")
- `--model_name`: 토크나이저 모델 이름/경로
- `--block_size`: 그루핑 블록 크기 (기본값: 1024)
- `--num_proc`: 병렬 전처리 프로세스 수 (기본값: 4)
- `--output_dir`: 전처리된 데이터셋 저장 경로

**처리 과정:**
1. `[내용영역]` 태그 제거
2. 유효하지 않은 텍스트 필터링
3. 토크나이제이션 (Qwen 토크나이저 사용)
4. 블록 단위로 그루핑 (효율적인 학습을 위해)

### 3. 모델 학습

#### 기본 학습 (Full Fine-tuning)

```bash
python script/train/post_traing.py \
  --dataset_dir ./datasets/qwen3-8b-base-kabc \
  --model_name Qwen/Qwen3-8B-Base \
  --output_dir ./outputs/qwen3-8b-base-kabc \
  --project buddha-train \
  --run_name qwen3-8b-base-kabc \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --block_size 2048 \
  --bf16
```

#### LoRA를 활용한 학습 (메모리 효율적)

```bash
python script/train/post_traing.py \
  --dataset_dir ./datasets/qwen3-8b-base-kabc \
  --model_name Qwen/Qwen3-8B-Base \
  --output_dir ./outputs/qwen3-8b-lora-kabc \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --bf16
```

**주요 학습 옵션:**
- `--dataset_dir`: 전처리된 데이터셋 경로
- `--model_name`: 기본 모델 이름/경로
- `--output_dir`: 학습 결과 저장 경로
- `--num_train_epochs`: 학습 에포크 수
- `--per_device_train_batch_size`: 배치 크기
- `--gradient_accumulation_steps`: 그래디언트 누적 스텝
- `--learning_rate`: 학습률 (기본값: 2e-5)
- `--bf16`: BFloat16 혼합 정밀도 학습 활성화

**LoRA 옵션:**
- `--use_lora`: LoRA 어댑터 사용
- `--lora_r`: LoRA rank (기본값: 16)
- `--lora_alpha`: LoRA alpha (기본값: 32)
- `--lora_dropout`: LoRA dropout (기본값: 0.05)
- `--lora_target_modules`: 대상 모듈 (기본값: q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj)

**Weights & Biases 로깅:**
- `--project`: W&B 프로젝트 이름
- `--run_name`: W&B 실행 이름

## 💡 코드 설명

### crawling_IT.py / crawling_BJ.py

동국대학교 한국불교전서(KABC) 웹사이트에서 불교 문헌을 자동으로 수집하는 크롤링 스크립트입니다.

**주요 기능:**
- HTTP 세션 관리 및 재시도 로직
- URL 정규화 및 중복 제거
- 병렬 크롤링으로 성능 최적화
- BeautifulSoup을 활용한 HTML 파싱
- JSONL 형식으로 데이터 저장

**데이터 소스:**
- `crawling_IT.py`: ABC_IT (일체경 번역문)
- `crawling_BJ.py`: ABC_BJ (불교저술)

### make_dataset.py

크롤링된 JSONL 데이터를 언어모델 학습에 적합한 형태로 전처리하는 스크립트입니다.

**처리 단계:**
1. **텍스트 정리**: 불필요한 태그(`[내용영역]`) 제거
2. **유효성 검증**: 빈 텍스트나 유효하지 않은 데이터 필터링
3. **토크나이제이션**: Qwen 토크나이저로 텍스트를 토큰으로 변환
4. **블록 그루핑**: 지정된 블록 크기로 토큰 시퀀스를 그룹화
5. **저장**: HuggingFace Datasets 포맷으로 디스크에 저장

### post_traing.py

Qwen3-8B-Base 모델을 불교 문헌 데이터로 파인튜닝하는 학습 스크립트입니다.

**주요 특징:**
- **Causal Language Modeling**: 다음 토큰 예측 방식의 언어모델 학습
- **Flash Attention 2**: 메모리 효율적인 어텐션 메커니즘 (선택사항)
- **Gradient Checkpointing**: 메모리 절약을 위한 체크포인팅
- **LoRA 지원**: Parameter-Efficient Fine-Tuning
- **8비트 Adam 옵티마이저**: 메모리 효율적인 최적화
- **W&B 통합**: 학습 과정 모니터링 및 로깅

**학습 흐름:**
1. 토크나이저 및 모델 로드
2. LoRA 설정 적용 (선택사항)
3. 전처리된 데이터셋 로드
4. DataCollatorForLanguageModeling 설정
5. HuggingFace Trainer로 학습 실행
6. 모델 및 메트릭 저장

## 📊 데이터 형식

### 크롤링 결과 (JSONL)

```json
{
  "url": "https://kabc.dongguk.edu/content/view?dataId=ABC_IT_K0223_T_001",
  "translation": "불교 문헌 번역 텍스트...",
  "metadata": {...}
}
```

### 전처리 후 데이터셋

HuggingFace Datasets 포맷:
```python
{
  "input_ids": [토큰_ID_배열],
  "attention_mask": [어텐션_마스크_배열]
}
```

## 🎯 활용 사례

- 불교 전문 용어 및 문맥 이해
- 한국 불교 문헌 번역 및 해석
- 불교 지식 기반 질의응답 시스템
- 불교 교리 학습 보조 도구

## ⚠️ 주의사항

- 크롤링 시 서버에 과부하를 주지 않도록 적절한 딜레이 설정 필요
- 학습 시 GPU 메모리가 부족한 경우 LoRA 사용 권장
- Flash Attention 2는 특정 GPU에서만 지원됨 (A100, H100 등)
- W&B 로깅을 사용하려면 `wandb login` 필요

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 사용됩니다.

## 🤝 기여

버그 리포트나 기능 제안은 GitHub Issues를 통해 제출해 주세요.

---

**개발자**: hojuna  
**프로젝트 저장소**: https://github.com/hojuna/buddha_LLM
