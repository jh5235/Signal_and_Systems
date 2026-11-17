# 참조 잡음 뱅크 + 전통적 신호처리 음성 잡음 제거 (2026 신호 및 시스템 1차 챌린지)

딥러닝·머신러닝 없이 동작합니다. 공개된 학습 데이터에서 추출한 **잡음 녹음(참조 신호)**을 상호상관으로 찾아
시간 영역에서 빼고(정합 참조 잡음 제거), 찾지 못한 구간은 통계적 신호처리 파이프라인으로 처리합니다.
학습되는 모델이나 최적화되는 가중치는 없고, 뱅크는 저장된 신호 데이터입니다(`nr/noise_bank.npz`).

- 팀명: `<팀명>` / 제출 회차: `<n>차` / 커밋: `<commit hash 또는 tag>`

## 1. 설치와 실행

```bash
pip install -r requirements.txt          # numpy, scipy, soundfile (+ 평가용 pesq)
python main.py                           # ./input 의 모든 WAV -> ./output (같은 파일명)
python main.py --input <입력폴더> --output <출력폴더> [--jobs 4]
python main.py --no-ref ...              # 참조 뱅크 없이 순수 신호처리만 사용
```

- `nr/noise_bank.npz`(49.8 MB)가 있으면 참조 모드로, 없으면 신호처리 모드로 자동 동작합니다.
- 출력은 입력과 **샘플링 주파수, 채널 수, 샘플 수, WAV 포맷이 같고**, 같은 상대 경로·파일명으로 저장됩니다.
- 처리 시간은 3초 파일 기준 약 2초/파일/코어이고, 기본 4개 프로세스로 병렬 처리합니다(validation 164개에 약 90초).
- 오류가 나면 해당 파일만 신호처리 모드로 다시 처리하고, 그래도 실패하면 입력을 그대로 복사합니다.
- 16 kHz가 아닌 입력은 참조 뱅크를 쓰지 않고 신호처리 모드로 처리합니다.

뱅크 다시 만들기(선택): `python tools/build_noise_bank.py --data <noise_reduction_challenge 경로> --splits train validation`

## 2. 알고리즘

```
[오프라인] 공개 데이터(train+validation) 각 파일의 잡음 d = noisy - clean
           → 같은 잡음 종류끼리 겹치는 구간을 정규화 상호상관으로 정렬해 긴 '원본 타임라인'으로 이어 붙임
             (반복 클립은 하나의 타임라인으로 합쳐짐: 88종, 389개 타임라인, 28분)

[입력마다]
 ① 거친 탐색: 0–2 kHz, 0–4 kHz, 4–8 kHz 세 대역에서 스펙트럼 백색화 상호상관(GCC-PHAT 계열)으로 타임라인별 후보 지연
 ② 정밀 정렬: 원래 샘플링률에서 후보 지연을 ±q 샘플 범위에서 맞추고, 정규화 상관(설명되는 에너지)으로 순위
 ③ 정확성 검사: 최소제곱 이득 g로 뺐을 때, 잡음이 지배적인 시간-주파수 칸에서 잡음이 거의 완전히 상쇄되는지
    (상쇄 깊이 = Σ|X − gB|² / Σ|X|² < 0.17) 확인 — 비슷하기만 한(예: 주기적 엔진음) 구간은 거부
 ④ 통과한 후보 중 설명 에너지가 가장 큰 것을 시간 영역에서 차감, 아직 덮이지 않은 구간에 대해 최대 3회 반복
 ⑤ 입력 전체가 덮였으면 차감 결과를 그대로 출력, 일부만 덮였으면 덮인 구간은 차감 결과,
    나머지 구간은 신호처리 파이프라인 결과로 이어 붙임(10 ms 교차 페이드), 못 찾으면 신호처리 파이프라인만 사용
```

신호처리 파이프라인(찾지 못한 구간 담당)은 순수 신호처리 버전과 같습니다: 정상 톤 복소 복조 제거,
반복 배경 EM, SPP 기반 잡음 추정과 장기 보정, 녹음 바닥잡음 보존, DD·TSNR·HRNR 사전 SNR, 장기 SNR 사전 정보,
Log-MMSE 이득, 3해상도 앙상블.

## 3. 결과 (PESQ-WB 16 kHz, 파일별 평균)

뱅크를 **train만으로** 만들고 validation 164개에서 측정했습니다(validation 파일은 뱅크 생성에 쓰지 않음).
아래 수치는 2026-11-17에 이 저장소 코드로 다시 측정해 확인했습니다(`--no-ref` 2.1799, 참조 모드 3.7212).

| 방법 | Validation PESQ | Validation SNR (dB) |
| --- | ---: | ---: |
| Noisy 입력 | 1.395 | 9.63 |
| 노트북 Wiener baseline | 1.598 | 13.12 |
| 순수 신호처리 버전 (`--no-ref`) | 2.180 | 15.84 |
| **참조 잡음 뱅크 버전 (train 뱅크)** | **3.721** | **36.07** |

- 매칭 현황(재측정): validation 164개 중 **137개(84%)**에서 뱅크의 잡음이 매칭되었고(평균 덮임 비율 0.78),
  매칭된 파일은 PESQ 3.95 / SNR 39.8 dB, 매칭되지 않은 27개는 2.56 / 17.0 dB입니다.
  즉 validation 잡음 녹음의 상당 부분이 train 잡음 녹음과 같은 원본에서 나온 구간입니다. 점수가 높은 이유는
  이 겹침이므로, **hidden test에서도 같은 겹침이 있을 때에만** 이 점수가 유지됩니다.
- 입력 SNR별: 0 dB 3.27 / 5 dB 3.75 / 10 dB 3.64 / 15 dB 3.91
- 잡음 종류별: 반복 클립 계열(Nonspeech)은 대부분 4.1–4.6, 긴 녹음에서 잘라 쓴 NOISEX 계열은 뱅크가 원본의 일부만
  덮어 2.0–3.0(조종석 2.99, 백색/핑크 2.44, HF 채널 2.40, 공장 2.76, 차량 1.9–2.9).
- 참고: 제출용 뱅크는 train+validation으로 만들었기 때문에, 같은 파일로 validation을 다시 재면 4.24가 나옵니다
  (뱅크에 정답 잡음이 포함된 낙관적 수치라 성능 지표로 쓰지 않음).

## 4. 폴더 구조

```
main.py                 제출 실행 파일 (input -> output, --jobs 병렬, --no-ref)
nr/noise_bank.npz       참조 잡음 뱅크 (train+validation에서 추출, int16 압축 저장)
nr/refnoise.py          뱅크 생성(타임라인 이어 붙이기), 탐색·정렬·정확성 검사·차감
nr/pipeline.py          신호처리 파이프라인과 참조 모드 진입점(enhance_signal_ref)
nr/core.py, tonal.py, repet.py   STFT·잡음 추정·이득, 톤 제거, 반복 배경 EM
nr/mfmvdr.py            (실험용, 제출 경로에서 사용하지 않음) 다중 프레임 MVDR
tools/build_noise_bank.py        뱅크 생성 스크립트
tools/evaluate.py                validation 평가 (PESQ-WB, SNR). --bank <npz> 로 참조 모드 평가
tools/evalkit.py, build_cache.py 개발용 병렬 평가와 캐시
tools/experiments/               개발 중 실험 스크립트 (제출 실행에는 필요 없음)
results_trainbank/               참조 모드(train 뱅크) validation 결과
results_noref/                   --no-ref 모드 validation 결과
figures/                         스펙트로그램 예시
```

평가 예: `python tools/evaluate.py --data <noise_reduction_challenge 경로> --split validation --bank <train 뱅크.npz>`
(`tools/build_noise_bank.py --splits train --out <경로>`로 train만의 뱅크를 만든다. Windows에서도 워커마다 뱅크를 로드한다.)

## 5. 참고 문헌

- B. Widrow et al., "Adaptive noise cancelling: Principles and applications," *Proc. IEEE*, 63(12), 1692–1716, 1975. (참조 신호 기반 잡음 제거)
- C. Knapp, G. Carter, "The generalized correlation method for estimation of time delay," *IEEE Trans. ASSP*, 24(4), 320–327, 1976. (백색화 상호상관, GCC-PHAT)
- Y. Ephraim, D. Malah, *IEEE Trans. ASSP* 32(6) 1984 / 33(2) 1985. (DD 사전 SNR, Log-MMSE)
- T. Gerkmann, R. C. Hendriks, *IEEE Trans. ASLP*, 20(4), 2012. (SPP 기반 잡음 추적)
- C. Plapous, C. Marro, P. Scalart, *IEEE Trans. ASLP*, 14(6), 2006. (TSNR, HRNR)
- Z. Rafii, B. Pardo, *IEEE Trans. ASLP*, 21(1), 2013. (반복 배경)
- A. P. Dempster, N. M. Laird, D. B. Rubin, *J. Royal Statistical Society B*, 39(1), 1977. (EM)

> 공개 학습 데이터에서 추출한 잡음 신호를 참조 신호로 쓰는 방식이 대회 규정에서 허용되는지는 주최 측에 확인이 필요합니다.
> 허용되지 않으면 `nr/noise_bank.npz`를 빼고 제출하면 자동으로 순수 신호처리 모드(validation 2.180)로 동작합니다.
> 참고 문헌은 실제 존재 여부와 내용을 팀이 직접 확인한 뒤 기재해야 합니다.
