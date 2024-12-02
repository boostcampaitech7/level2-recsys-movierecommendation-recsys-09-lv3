# Movie Recommendation Baseline Code

영화 추천 대회를 위한 베이스라인 코드입니다. 다음 코드를 대회에 맞게 재구성 했습니다.

- 코드 출처: https://github.com/aHuiWang/CIKM2020-S3Rec

## Installation

```
pip install -r requirements.txt
```

## How to run

```bash
python3 main.py --config config/config.yaml
```
or
```bash
sh run_program.sh
```
## How to run memory_based recommendation

```bash
python3 run_memory_based.py
```
## WandB
1. 루트 디렉토리에 `.env` 파일을 생성(`touch ./.env`)
2. `.env` 파일에 `WANDB_API_KEY`를 입력
```
WANDB_API_KEY="****"
```
