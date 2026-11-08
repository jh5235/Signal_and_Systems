# 연세대학교 미래캠퍼스 신호및시스템 4조 프로젝트

- 조원 : 신준현, 이민혁, 이재우

딥러닝·머신러닝을 쓰지 않는 **전통적 신호 처리(DSP) 기반 음성 잡음 제거**입니다. Log-MMSE 기반 알고리즘의 파라미터를 Optuna(TPE)로 튜닝했습니다.

## 목차

1. [알고리즘 설명](#1-알고리즘-설명)
2. [검증 PESQ 결과](#2-검증-pesq-결과)
3. [테스트 데이터와 PESQ 결과](#3-테스트-데이터와-pesq-결과)

---

## 1. 알고리즘 설명

모든 처리는 STFT 영역에서 이득 G(k,m)을 noisy 스펙트럼에 곱하는 방식입니다. 위상은 noisy 위상을 그대로 씁니다. 학습된 가중치는 없고, 파라미터는 `03_model/dsp.py`에 상수로 들어 있습니다.

```
x[n] → STFT → 파워 P → 잡음 PSD λ 추정 → 사후 SNR γ · 사전 SNR ξ → Log-MMSE 이득 G → 저주파 가중 → G·Y → ISTFT → ŝ[n]
```

| 단계 | 내용 |
|---|---|
| ① STFT | Hann 창, 길이 `n_fft`=1024, 이동 `hop`=192 |
| ② 잡음 PSD 추정 | **고정 추정**(에너지가 가장 낮은 약 9% 프레임의 평균 파워)과 **추적 추정**(음성 존재 확률 SPP 기반 unbiased MMSE 추적기, Gerkmann & Hendriks 2012)을 기하평균으로 결합하고, 과대 추정 계수(`noise_scale`≈1.75)를 곱함 |
| ③ 사전 SNR | decision-directed 방식(Ephraim & Malah 1984). 이전 프레임의 추정 음성 파워와 현재 프레임을 `a`≈0.88로 섞음 |
| ④ 이득 | Log-MMSE 진폭 추정기(Ephraim & Malah 1985), G = ξ/(1+ξ)·exp(½E₁(v)), v = ξ/(1+ξ)·γ. 이득 하한 약 −31 dB |
| ⑤ 저주파 가중 | 6차 Butterworth형 고역통과 가중, 기준 주파수 약 384 Hz, 최대 감쇠 약 −21 dB. 차량·엔진·바람처럼 저역에 몰린 잡음을 억제 |
| ⑥ 복원 | ISTFT로 입력과 같은 길이의 파형을 만듦 |

**Optuna 튜닝**: 위 단계의 파라미터 13개(`n_fft`, `hop`, `frac`, `w`, `a_pow`, `xi_h1_db`, `noise_scale`, `a`, `xi_min_db`, `gmin_db`, `hp_hz`, `hp_order`, `hp_floor_db`)를 `TPESampler`로 299 trials 탐색했습니다. 목적함수는 train 400개의 평균 PESQ-WB이고, 알고리즘 구조는 바꾸지 않았습니다. 원본(이재우 제출물)의 coordinate descent 결과를 첫 trial로 넣어 시작했습니다.

> **PESQ–SNR trade-off**: 저주파를 강하게 줄이면 PESQ는 오르지만 음성 저역(기본주파수 대역)도 함께 줄어 파형 SNR은 낮아집니다. PESQ만 최적화한 Optuna 모델은 SNR이 noisy 입력보다 낮습니다.

실행:

```bash
pip install -r 03_model/requirements.txt
python 03_model/enhance.py --input path/to/input --output path/to/output
```

---

## 2. 검증 PESQ 결과

대회 공개 데이터(`noise_reduction_challenge`)의 **validation 164개**로 평가했습니다. 튜닝에는 train 400개만 썼습니다.

| | PESQ-WB | SNR (dB) |
|---|---:|---:|
| Noisy 입력 | 1.3947 | 9.63 |
| 원본 제출 (이재우, coordinate descent) | 2.0037 | 5.64 |
| **Optuna 튜닝 모델** | **2.0505** | 2.94 |

- Optuna가 원본보다 PESQ **+0.047** 높습니다.
- 탐색에 쓰지 않은 train 나머지 896개에서도 원본 2.0079 → Optuna 2.0799로 올라, 개선이 유지됩니다. 탐색에 쓴 train 400개에서는 1.9842 → 2.1006입니다(탐색 데이터에 일부 과적합).
- 타깃 SNR별 PESQ(Optuna): 0 dB 1.401, 5 dB 1.735, 10 dB 2.050, 15 dB 2.440.

재현:

```bash
pip install -r 01_code/requirements.txt
python 01_code/tune_optuna.py --trials 300 --subset 400 --jobs 14   # 튜닝, 약 35분
python 01_code/evaluate_model.py --profile optuna                   # validation 평가
```

데이터셋은 저장소에 없습니다. 기본 경로는 `<신호및시스템>/00_원자료/noise_reduction_challenge`이며, 다른 위치는 환경변수 `NR_DATA`로 지정합니다. 파일별 점수와 최적 파라미터는 `02_result/`에 있습니다.

---

## 3. 테스트 데이터와 PESQ 결과

튜닝에 쓰지 않은 **외부 데이터**로 일반화를 확인했습니다.

| 항목 | 내용 |
|---|---|
| 데이터 | [VoiceBank-DEMAND 16k](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k) (Hugging Face `JacobLinCool/VoiceBank-DEMAND-16k`) test split, CC-BY-4.0 |
| 규모 | clean/noisy 쌍 824개, 16 kHz, 1.2~9.8초 |
| 특징 | 대회 데이터와 화자·잡음(DEMAND)·SNR(2.5~17.5 dB) 구성이 다름 |

| | PESQ-WB | SNR (dB) |
|---|---:|---:|
| Noisy 입력 | 1.9707 | 8.45 |
| 원본 제출 (이재우) | 2.6468 | 8.49 |
| **Optuna 튜닝 모델** | **2.7887** | 4.87 |

- Optuna가 원본보다 PESQ **+0.142** 높습니다(파일별 차이의 bootstrap 95% 신뢰구간 0.135~0.149, 824개 중 93.6%에서 우위).
- 이 데이터는 잡음이 비교적 정상적이고 SNR이 높아, 대회 validation(+0.047)보다 개선폭이 크게 나왔을 수 있습니다.

재현:

```bash
python 04_test/evaluate_external.py     # 데이터 자동 다운로드(04_test/data, git 제외) → 04_test/result/
```

---

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `01_code/` | 튜닝(`tune_optuna.py`)·검증(`evaluate_model.py`) 코드 |
| `02_result/` | Optuna 결과(`best_params.json`, `trials.csv`, `optuna.db`)와 validation 점수 |
| `03_model/` | 완성 모델(`enhance.py`, `dsp.py`) |
| `04_test/` | 외부 데이터 검증 스크립트와 결과 |
| `99_docs/` | 실험 설계 및 결과 상세 문서 |
