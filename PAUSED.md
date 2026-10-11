# 일시정지 기록 (2026-09-27 04:50 KST)

사용자 지시로 모든 작업을 멈춤. 종료는 기록된 PID로만 했고, 종료 후 GPU와 CPU가 비었음을 확인함.
- GPU 0/1/2: 사용률 0%, 17~33 MiB, 연산 프로세스 없음.
- 대기열이나 자동 체인에 남은 작업 없음.
- 2단계 B-4 체인은 03:49에 모두 끝나 있었음.

## 멈춘 작업

| 작업 | 무엇 | 어디까지 진행 | 종료한 PID |
|---|---|---|---|
| Lamé 검증 (`stage3/lame_check.py`) | 두꺼운 구 껍질 내압 해석해 대비 기본 P1 / 선택적 절점 평균(ANP), ν 0.3/0.45/0.49, N 8192/16384/32768 × 시드 2개, ν 0.3·0.49 스펙트럼 | 작업 6개 중 N=8192 두 개 완료. 다만 결과를 마지막에 한 번에 쓰는 구조여서 **파일로 남은 것은 없음**. 16k/32k 4개는 진행 중이었음 | 2680531 (conda run), 2680535, 2680546 (주 프로세스), 작업자 2680579–86. Pool이 다시 띄운 작업자 2689854/55/64–67도 PID로 종료 |
| E1 재평가 (`stage3/e1_v2.py`) | ANP 기준 해로 기저 neo / neo_rot / neo_rot_sc / elastic_sc 비교. "모델 오차 / 덩어리 신호" 비율을 깊이별로, 형상군 8개 × 2, ν 3개 | 시작 4분 만에 멈춤. **끝난 형상 없음** | 2686929 (conda run), 2686933, 2686944 (주 프로세스), 작업자 2686974–83. 다시 띄워진 작업자 2689856–63도 PID로 종료 |

- 학습 중이던 작업은 없었음 (둘 다 CPU 수치 실험이라 체크포인트 대상 없음).
- **이미 확보된 결과 (정지 전 단독 실행):**
  - Lamé, N=8192 메쉬 1개 (터미널 출력으로 확인). ν = 0.49에서 기본 P1 오차 22.3% (locking), ANP 4.5%.
  - ANP의 ν = 0.49 스펙트럼: 0 고유값 6개(강체)뿐, λ₇ = 1.37 (ν = 0.3에서 1.47). 가짜 모드 없음.
  - E1 재평가, torus 하나 ν = 0.3: neo_rot_sc 비율 0.05~0.08, neo_rot 0.9~3.1, neo 4~13, elastic_sc 0.18~0.24.
  - 이 값들은 파일로 저장되지 않았으므로, 재개 시 다시 계산함.

## 재개 방법
정지 후 두 스크립트를 **작업 단위 저장 + 완료분 건너뛰기** 방식으로 고쳐 둠.
- `lame/parts/N*_s*.csv`
- `e1/e1_v2_parts_<nu>/<family>_<idx>.csv`

```bash
cd /home/cvmlserver7/Hyeon/NEO_DEEPONET/stage3 && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup conda run --no-capture-output -n neo python lame_check.py --procs 6 > logs/lame.log 2>&1 &
```
```bash
cd /home/cvmlserver7/Hyeon/NEO_DEEPONET/stage3 && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup conda run --no-capture-output -n neo python e1_v2.py --nu 0.3,0.45,0.49 --procs 8 > logs/e1_v2.log 2>&1 &
```

- 재개할 때 PID는 `logs/*.pid`에 기록함.
- **예상 시간 (정지 전 실측 기준):**
  - Lamé: 8192점 작업 1개에 약 9분. 전체는 32768점 작업 때문에 1~1.5시간으로 추정.
  - E1 재평가: 형상 × ν 하나에 약 9분 (4스레드, 부하 낮을 때). 16형상 × 3ν를 8프로세스로 돌리면 약 1~1.5시간.

---

# 일시정지 기록 2 — C-(c) 숨은 덩어리 찾기 (2026-09-27 22:10 KST)

사용자 승인으로 멈춤 (관문 실험을 먼저 진행하기 위해).

| 무엇 | 어디까지 | 종료한 PID |
|---|---|---|
| B-4, 실제 형상 36개 사례 × 기저 3개 (neo256 / neoinv2 / arpack256) × 2조각, GPU 0/1/2 (21:14 시작) | 108개 중 13개 끝남 | 파이썬 3449954, 3449957, 3449977, 3449987, 3449988, 3449993; conda 래퍼 3449882, 3449895, 3449903, 3449911, 3449921, 3449935; 대기 체인 3402560, 최종 체인 3479120 |

- **보존된 것:** 끝난 13개 사례의 로그 지표 (찾은 덩어리, 오탐, 상대 L2, 피팅 κ 등)를 `stage3/real_shapes/c_b4_partial_from_logs.csv`에 모음. 사례별 단면 그림은 각 조각 폴더에 있음.
  - 덩어리별 상세(`per_lump`)는 로그에 없어서 재개할 때 다시 계산됨.
- **멈춘 뒤 수정:** `stage2/coef_b4.py`가 사례마다 `<출력 폴더>/parts/<사례>.csv`를 쓰고, 재실행하면 있는 사례를 건너뜀. 이번에 멈춘 실행은 수정 전 코드라 parts가 없으므로, 13개도 다시 계산됨.
- **느렸던 이유:** fp64 조립(RTX 3090의 fp64는 fp32의 1/64), 분할 시도마다 재피팅, GPU당 2개 작업.

**재개 방법** (관문 실험 후)
1. fp32 조립 확인을 먼저 함 (`--asm32`, 사례 6개: cow·dragon·thingi101619 × c1·c2, 기저 neoinv2, fp64 결과와 비교).
   - fp64 결과가 없으므로, 같은 6개 사례를 fp64로도 `--names`로 돌려 비교함.
2. 결과가 같으면 fp32로 전체 재개함. 기저마다 명령 하나씩이고, 결과는 `_asm32` 태그 폴더에 저장됨:
   ```bash
   cd /home/cvmlserver7/Hyeon/NEO_DEEPONET/stage2 && CUDA_VISIBLE_DEVICES=<gpu> conda run -n neo python coef_b4.py --split test_real --basis neoinv2 --net base --asm32 --shard <k> --nshards 2
   ```
3. 조각이 끝나면 `--merge 1`로 합치고 `stage3/c_report.py`로 정리함.
   - `c_report.py`는 `_asm32` 폴더 이름을 읽도록 한 줄 수정이 필요함.
