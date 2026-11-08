# 03_model — Optuna 튜닝 완성 모델

Log-MMSE + 고정/추적 결합 잡음 PSD + 저주파 가중 DSP 모델입니다. 학습된 가중치는 없고, 파라미터가 `dsp.py`의 `PROFILES["optuna"]`에 들어 있습니다(기본값).

```bash
pip install -r requirements.txt
python enhance.py --input path/to/input --output path/to/output   # 기본 profile=optuna
python enhance.py --profile pesq                                  # 재우 원본 제출 파라미터
```

입력과 같은 파일명·샘플링 주파수·채널·샘플 수로 저장합니다. Validation PESQ-WB 2.0505 (원본 2.0037).
