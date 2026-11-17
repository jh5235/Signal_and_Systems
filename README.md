# 2026 신호 및 시스템 1차 챌린지 — 전통적 신호 처리 기반 음성 잡음 제거

| 제출 정보 | 값 |
|---|---|
| 팀명 | 베이스라인을 이겨보겠습니다! |
| 조원 | 신준현, 이재우, 이민혁 |
| 제출 회차 | 3차 |
| 채점 대상 | 이 저장소 `main` 브랜치의 최신 커밋 |
| Validation 164개 | **PESQ-WB 2.9849**, SNR 2.08 dB |
| Noisy 입력 | PESQ-WB 1.3947, SNR 9.63 dB |

딥러닝·머신러닝을 쓰지 않습니다. 학습된 모델, 저장된 잡음 파형, Clean 음성, 라벨을 실행 중에 읽지 않습니다. **각 파일은 그 Noisy 파일 하나만으로 처리**하며 다른 파일의 정보를 쓰지 않습니다.

## 설치와 실행

Python 3.10 이상, CPU만 필요합니다.

```bash
pip install -r requirements.txt
python enhance.py --input input --output output
```

input 폴더의 모든 WAV마다 같은 이름의 WAV 하나를 output에 저장합니다. 샘플링 주파수, 채널 수, 샘플 수, 인코딩을 유지하고 저장 후 검사합니다.

## 알고리즘

1. **반복 잡음 탐색** (`denoise/xmatch.py`): 스펙트럼을 백색화한 상관을 FFT로 계산해, 같은 잡음 파형이 시간만 어긋나 다시 나타나는 위치를 찾습니다. 탐색 범위는 한 파일 안의 반복(자기상관)입니다.
2. **검증** : 후보 정렬에서 두 신호의 STFT 복소 계수가 하나의 실수 배율로 일치하는 빈의 비율을 계산합니다. 같은 파형일 때만 음성이 없는 빈에서 복소수로 일치하므로, 비율이 낮으면 버립니다.
3. **잡음 추정과 차감** (`denoise/xcancel.py`): 대상과 정렬된 복사본들이 같은 잡음 n과 서로 독립인 음성을 가진다고 보고(X_c = n + e_c), 빈별 분산을 EM 반복으로 추정한 뒤 복사본들의 MMSE 가중 평균으로 n을 추정해 시간 영역에서 뺍니다. 대상 자신의 계수는 잡음 추정에 넣지 않아 대상 음성이 지워지지 않습니다.
4. **파일별 향상** (`denoise/dsp.py`, `denoise/single_rules.py`, `denoise/ridge.py`, `denoise/final.py`): 안정 정현파·충격음 억제, SPP 기반 잡음 PSD 추정, Decision-Directed 사전 SNR, Log-MMSE 이득, 출력 필터. 복사본이 잡음을 설명하지 못하는 구간은 이 경로의 결과를 이어 붙입니다.
5. 안정 순음이 지배적이거나 저역 잡음이 몇 개의 스펙트럼 선에 몰린 파일은 3단계를 건너뛰고 4단계만 씁니다(Validation에서 손해가 확인된 경우).

모든 임계값은 코드에 고정되어 있습니다(`denoise/repeat.py`의 `DEFAULTS`, `denoise/dsp.py`의 `PROFILES`, `denoise/final.py`).

## Validation 재현

```bash
pip install pesq==0.0.4
python evaluate.py --data <noise_reduction_challenge 폴더> --split validation
```

`evaluate.py`는 `enhance.py`를 그대로 실행한 뒤 저장된 WAV를 다시 읽어 점수를 계산합니다. Clean은 점수 계산에만 씁니다. 파일별 결과는 `results/validation_scores.csv`에 있습니다.

Train에서 무작위로 164개씩 뽑은 폴더(겹치지 않는 5개)에서 2.89 / 3.05 / 2.99 / 3.00 / 3.02(평균 2.99)였습니다.

## 한계

- 반복 잡음 차감의 이득은 한 파일(3초) 안에서 같은 잡음 파형이 다시 나타날 때만 생깁니다. 그렇지 않은 파일은 파일별 향상 단계만 적용됩니다.
- PESQ를 높이려고 고역을 강하게 줄이므로 SNR은 Noisy 입력보다 낮습니다.

## 참고 문헌 (원문 확인 후 AI 활용 내역서에 기재)

- C. Knapp, G. Carter, "The generalized correlation method for estimation of time delay," IEEE Trans. ASSP 24(4), 1976. doi:10.1109/TASSP.1976.1162830
- A. P. Dempster, N. M. Laird, D. B. Rubin, "Maximum likelihood from incomplete data via the EM algorithm," JRSS B 39(1), 1977.
- B. Widrow et al., "Adaptive noise cancelling: Principles and applications," Proc. IEEE 63(12), 1975. doi:10.1109/PROC.1975.10036
- Z. Rafii, B. Pardo, "REpeating Pattern Extraction Technique (REPET)," IEEE Trans. ASLP 21(1), 2013. doi:10.1109/TASL.2012.2213249
- Y. Ephraim, D. Malah, "Speech enhancement using a MMSE log-spectral amplitude estimator," IEEE Trans. ASSP 33(2), 1985. doi:10.1109/TASSP.1985.1164550
- T. Gerkmann, R. C. Hendriks, "Unbiased MMSE-based noise power estimation with low complexity and low tracking delay," IEEE Trans. ASLP 20(4), 2012. doi:10.1109/TASL.2011.2180896
