# 04_test — 외부 데이터 검증 (VoiceBank-DEMAND)

학습·튜닝에 쓰지 않은 **외부 데이터**로 `03_model`을 평가합니다.

| 항목 | 내용 |
|---|---|
| 데이터 | [`JacobLinCool/VoiceBank-DEMAND-16k`](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k) test split (CC-BY-4.0) |
| 규모 | 824개 clean/noisy 쌍, 16 kHz, 1.2~9.8초(중앙값 2.3초) |
| 특징 | 챌린지 데이터와 화자·잡음(DEMAND)·SNR(2.5~17.5 dB) 구성이 다름 |
| 지표 | PESQ-WB, SNR (clean 대비) |

## 결과

| 입력 / 프로파일 | PESQ-WB | SNR (dB) |
|---|---:|---:|
| Noisy 입력 | 1.9707 | 8.45 |
| 원본 제출 (`pesq`) | 2.6468 | 8.49 |
| **Optuna** | **2.7887** | 4.87 |
| `balanced` | 2.4576 | 15.22 |

- Optuna가 원본보다 PESQ **+0.142** 높습니다(파일별 paired 차이의 bootstrap 95% CI 0.135~0.149, 824개 중 93.6%에서 우위).
- Noisy보다 PESQ가 오른 파일은 Optuna 99.6%, 원본 99.0%입니다.
- 챌린지 validation(+0.047)보다 개선폭이 큽니다. 단, 이 데이터는 잡음이 비교적 정상적이고 SNR이 높아 저주파 억제가 더 잘 맞는 영향일 수 있습니다.
- SNR은 챌린지 때와 같은 경향입니다. PESQ 중심으로 튜닝한 Optuna는 SNR이 noisy보다 낮고, `balanced`는 SNR이 크게 오릅니다.

## 실행

```bash
pip install -r ../01_code/requirements.txt huggingface_hub pyarrow
python evaluate_external.py        # 데이터 자동 다운로드(data/, git 제외) → result/
```

`result/summary.csv`에 요약, `result/external_<profile>_scores.csv`에 파일별 점수가 저장됩니다.
