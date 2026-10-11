# NEO 사용 가능 여부 확인

**요약: 사용 가능함.**
- 사전학습 가중치는 저장소에서 제공됨.
- README 설치 방식(pip)은 torch 빌드가 맞지 않아 모델이 import되지 않음. `neo` 환경 안에서 torch를 cu126 빌드로 재설치해 해결함(`uv.lock`과 같은 조합).
- 형상 3개(suzanne / sphere / torus, N=2048)에서 추론이 됨.
- Ours-Large, k=96 기준 ARPACK 대비 **span loss는 5.9e-4 ~ 2.2e-3**으로, README의 ShapeNet 수치 3.35e-3과 같거나 더 좋음. **고유값 상대오차는 0.76% ~ 2.1%**이고, 항상 과대추정됨.
- 추론 시간은 ARPACK 218 ms 대비 **24 ms**.

작업 공간은 `NEO_DEEPONET/neo_check/`이고, 저장소 `NEO/`의 파일은 하나도 수정하지 않음. GPU는 비어 있던 0번 하나만 사용함(`CUDA_VISIBLE_DEVICES=0`).

## 1. 저장소

- `NEO/`는 이미 clone되어 있었음. `origin = github.com/Adversarr/NEO`, `main @ d928c31`
- `git fetch` 결과 원격과 같고, 로컬 변경은 없음.

## 2. 환경 (`neo`, Python 3.12.14)

| 단계 | 명령 | 결과 |
|---|---|---|
| 1 | `cd NEO && ./setup_conda.sh -n neo` (README "Automatic") | exit 0. 로그: `setup_conda.log` |
| 2 | `import g2pt.training.pretrain` | **실패**: `torch_cluster/_version_cuda.so: undefined symbol …` → `pointnet2.py:13 quit("This example requires 'torch-cluster'")` |
| 3 | `conda run -n neo python -m pip install "torch==2.6.0+cu126" "torchvision==0.21.0+cu126" --index-url https://download.pytorch.org/whl/cu126` | 성공. 로그: `torch_cu126_install.log` |
| 4 | 재검증 | torch 2.6.0+cu126, CUDA 12.6, cxx11abi=True, RTX 3090. `torch_cluster.knn` GPU 실행됨, `g2pt.training.pretrain` import됨 |

**2단계 실패 원인:** README의 pip 경로와 저장소의 `pyproject.toml`/`uv.lock`이 다름.
- `requirements.txt`의 `torch==2.6.0`은 PyPI 빌드로 설치됨(cu124, cxx11abi=False).
- `requirements-pyg.txt`는 torch 2.6.0+cu126용 `torch_cluster` 휠을 받음.
- `uv.lock`은 torch를 `download.pytorch.org/whl/cu126`에서 받도록 고정해 두었으므로, 3단계는 저장소가 의도한 조합으로 맞춘 것임.
- 드라이버 550.54(CUDA 12.4)에서 cu126 빌드가 정상 동작함.
- `setup_conda.sh`의 검증 단계는 `import g2pt`만 확인하기 때문에 이 문제를 잡지 못함.

**README 오류:** Manual 절의 `conda env create --name neo "python=3.12"`는 conda 문법 오류임. `conda env create`는 yml 파일을 받는 명령이므로 `conda create`가 맞음.

## 3. 사전학습 가중치 — 있음

README의 Google Drive 폴더에서 받아 `neo_check/ckpts/`에 둠.

| 파일 | sha256 (앞 8자리) | 구성 |
|---|---|---|
| `Ours-Large.ckpt` (138 MB) | `06c9b845` | TransolverExp, d_model 256, 6층, RoPE, 16.9M 파라미터, targ_dim 96 / 출력 부분공간 256, epoch 399 |
| `Ours-Small-NoRoPE.ckpt` (44 MB) | `b745be5f` | d_model 192, 4층, RoPE 없음, 5.4M 파라미터, 출력 192, **epoch 1 (944 step)** |

가중치가 있으므로 학습은 하지 않음.

## 4. 추론

### 형상

- **Suzanne:** 저장소에 있는 유일한 메쉬 `pyfastspectrum/suzanne.obj`. 원본은 507 정점이라 `--subdiv 2`로 세분함.
- **Bunny/Armadillo는 받지 못함:** Open3D 예제 데이터를 `data_root=neo_check/data`로 받으려 했으나, GitHub release URL이 3번 연속 **HTTP 500**을 돌려줌. 다른 미러는 찾지 않음.
- **대체 형상:** trimesh로 로컬에서 생성한 **구**(icosphere subdiv 5, 10242 정점)와 **토러스**(R=1, r=0.4)를 사용함. 파일은 `neo_check/meshes/`에 있음.
  - 구는 해석해 l(l+1)/R²로 추가 검증이 가능함.

### 명령

`D=neo_check`이고, 매 명령 앞에 다음 환경변수를 **명령 단위로만** 붙임:

```
CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=$D/.cache/mpl XDG_CACHE_HOME=$D/.cache TRITON_CACHE_DIR=$D/.cache/triton
```

```bash
# 입력 생성 (저장소 스크립트, README에는 없음). suzanne는 --subdiv 2 추가
conda run -n neo python scripts/sample_points.py --mesh_path $D/meshes/<shape>.obj \
    --n_points 2048 --k 256 --normalize --seed 0 --out $D/samples/<shape>/sample_points.npy
# 추론 (README 4절)
conda run -n neo python exp/pretrain/infer.py --ckpt $D/ckpts/Ours-Large.ckpt \
    --data_dir $D/samples --glob "*" --device cuda [--k 96]
# 요약
conda run -n neo python neo_check/summarize.py
```

- 결과 JSON: `results/{large,small}_k{96,256}_<shape>.json`
- 요약 표: `results/summary.{md,csv}`
- 로그: `infer_*.log`

### 수치 (fp32, N=2048, ARPACK = `infer.py`의 `solve_gev_ground_truth`, 같은 point-cloud L/M)

| ckpt | k | 형상 | span loss | 고유값 상대오차 평균 / 최대 | i=1..10 평균 | cos(pc,net) 중앙값 | NEO / ARPACK 시간 |
|---|---|---|---|---|---|---|---|
| Large | 96 | sphere | **5.85e-4** | 0.76% / 0.86% | 0.78% | 0.991 | 24 / 218 ms |
| Large | 96 | suzanne | **2.23e-3** | 2.13% / 3.76% | 2.46% | 0.973 | 24 / 218 ms |
| Large | 96 | torus | **6.49e-4** | 0.77% / 0.87% | 0.79% | 0.999 | 24 / 211 ms |
| Small | 96 | sphere | 1.00e-3 | 0.96% / 1.18% | 0.98% | 0.986 | 15 / 219 ms |
| Small | 96 | suzanne | 4.29e-3 | 3.12% / 5.23% | 3.52% | 0.916 | 15 / 222 ms |
| Small | 96 | torus | 1.13e-3 | 0.99% / 1.22% | 1.03% | 0.997 | 15 / 215 ms |
| Large | 256 (기본값) | sphere / suzanne / torus | 0.069 / 0.169 / 0.083 | 2.6–5.9% / 25–46% | 0.8–2.5% | 0.75–0.98 | 24 / 660–730 ms |

- fp16 결과는 fp32와 거의 같음(span loss 차이 1% 미만, `summary.md`).
- **k 기본값에 주의:** `infer.py`에 `--k`를 주지 않으면 `targ_dim_model`(=256)개의 모드를 평가함. 모델은 96개 모드로 학습되었으므로, README의 3.35e-3과 비교할 수 있는 수치는 k=96 행임.

## 5. 해석과 주의점

- **고유값 오차는 부분공간에서 오는 것임(검증함).** 고유값 오차는 모든 모드에서 거의 일정하고(+0.64 ~ +0.86%), 항상 양수(Rayleigh–Ritz 상한)임.
  - `infer.py`의 Rayleigh–Ritz는 대각 SᵀMS만 사용함. 그래서 dense SᵀMS와 float64로 다시 계산해 비교함(`check_rr.py` → `results/check_rr.txt`).
  - 결과: S는 M-정규직교(비대각 성분 ≤ 3e-7)이고, 두 계산의 오차는 1e-4 이내로 같음. 따라서 평가 코드의 단순화 때문이 아니라 학습된 부분공간의 고유한 오차임.
- **NEO가 맞추는 대상은 point-cloud 라플라시안임.** NEO의 오차는 robust_laplacian의 ARPACK 해 대비 수치이고, 참 LBO 대비가 아님.
  - 같은 구(N=2048)에서 ARPACK 고유값 자체가 l(l+1)/R² 대비 **−5.0%**(i=1..3), **−12%**(i=80..95)임.
  - spec_heat/bias의 강성 편향 결론과 일치함. 즉 NEO를 쓰더라도 이 편향은 그대로 물려받음.
- **mesh↔pc / mesh↔net 코사인 점수(0.27–0.53)는 신뢰하기 어려움.**
  - `scripts/sample_points.py`가 mesh 고유벡터를 `axis=1`, 즉 점별로 정규화하는 것으로 보임(열별 정규화인 `axis=0`이어야 할 것 같음).
  - 구·토러스는 고유공간이 겹치는(degenerate) 모드가 많아 1:1 코사인 매칭 자체가 불안정함.
  - span loss와 고유값 오차는 이 문제와 무관함.
- **n_neighbors 불일치:** 학습 데이터는 `n_neighbors=20`(`g2pt/utils/mesh_feats.py:71`), `infer.py`는 기본값 30으로 라플라시안을 만듦.
- **Small 체크포인트:** epoch 1 체크포인트지만 span loss가 Large보다 1.7–1.9배 나쁜 정도에 그침.

## 6. 막힌 부분 / 서버 규칙 관련

- **해결한 것:** torch/torch_cluster ABI 불일치. `neo` 환경 내부 재설치로 해결함.
- **못 한 것:** Open3D 예제 메쉬(GitHub HTTP 500)와 ShapeNet(로컬에 없음)은 사용하지 못함.
- **새 규칙 이후 프로젝트 밖에 만든 파일:** 없음. 확인한 방법은 `find ~ -newermt`이며, 예외는 conda 환경 디렉터리, pip 캐시, Claude 세션 기록뿐임.
- **새 규칙 전에 프로젝트 밖에 남은 것:**
  - 세션 scratchpad(`/tmp/claude-…/scratchpad`)에 둔 임시 파일 2개 → 삭제함.
  - spec_heat FEM 실행 때 gmsh(FLTK)가 `~/.fltk/fltk.org/fltk.prefs`를 만듦. 삭제 여부는 결정 필요.
  - `spec_heat/run_all.py`를 다시 실행하면 이 파일이 다시 생길 수 있음.

## 7. [과제 1] Part 0 — N=8000 재계산과 곡면 강성 보정

### NEO, N=8000, k=96

`samples_n8000/`, `results/large-n8000_k96_*.json`, `infer_large_k96_n8000.log`

| 형상 | span loss N=2048 → 8000 | 고유값 상대오차 평균 N=2048 → 8000 | NEO / ARPACK 시간 (N=8000) |
|---|---|---|---|
| sphere | 5.85e-4 → **2.31e-4** | 0.76% → 0.89% | 33 / 531 ms |
| suzanne | 2.23e-3 → **1.40e-3** | 2.13% → 2.54% | 33 / 505 ms |
| torus | 6.49e-4 → **2.66e-4** | 0.77% → 0.92% | 33 / 500 ms |

- N=8000에서 span loss는 2–2.5배 좋아짐(학습 해상도 2k보다 큰 입력으로 zero-shot 전이됨).
- 고유값 오차는 조금 커짐.
- `infer.py`는 라플라시안을 n_neighbors=30으로 만듦.

### 곡면 보정 c = 2·sum(M)/(xᵀLx+yᵀLy+zᵀLz), 구의 l(l+1)/R² 대비

`sphere_calib.py` → `results/sphere_calib*.{txt,csv}`. 표의 값은 부호 있는 상대오차 평균임.

| nn | N | c | 보정 전 l=1 / l=9 | 보정 후 l=1 / l=9 |
|---|---|---|---|---|
| 30 | 2048 | 1.043 | −5.0% / −12.5% | **−1.0%** / −8.8% |
| 30 | 8000 | 1.042 | −4.9% / −6.8% | **−0.9%** / **−2.9%** |
| 20 | 2048 | 1.045 | −5.7% / −13.1% | −1.4% / −9.1% |
| 20 | 8000 | 1.116 | **−19.8% / −21.8%** | **−10.4% / −12.7%** |

**nn=30에서의 효과**
- 보정하면 저차 편향이 약 −5%에서 약 −1%로 줄어듦.
- 고차 모드(l=9)의 오차는 N=2048→8000에서 −8.8% → −2.9%로 줄어듦. 즉 고차 오차는 해상도로 줄어드는 성분이고, 저차의 −5%는 c로 없어지는 성분임.

**nn=20, N=8000은 비정상:** 보정 전 −20%, 질량합 0.73×면적. c로는 복구되지 않음(균일한 스케일 오차가 아니기 때문).

추가 점검으로 단위 구에서 점 배치와 N을 바꿔 봄. 값은 저차 8모드 평균 편향과 (질량합/면적).

| 점 배치 | N | nn=20 | nn=30 |
|---|---|---|---|
| 랜덤 균일 | 2048 / 4000 / 8000 / 16000 | −6.1% (0.97) / −6.2% (0.97) / **−20.9% (0.73)** / **−38.7% (0.48)** | −5.2% / −5.1% / −5.0% / −6.2% (≈1.0) |
| Fibonacci (준균일) | 2048 / 4000 / 8000 / 16000 | −0.1% / −0.1% / −3.7% / **−18.9% (0.83)** | −0.1% / −0.1% / −0.04% / −0.02% |

**결론**
1. robust_laplacian point-cloud 라플라시안은 **n_neighbors=20에서 N이 커지면 질량과 강성이 함께 무너짐**. nn=30은 안정적임.
2. 약 −5%의 저차 편향은 **랜덤(불규칙) 샘플링에서만** 나타남. 준균일 점에서는 거의 0임.
   - spec_heat/bias의 "강성 편향" 결론은 그대로 맞음. 다만 그 원인이 샘플의 불규칙성이라는 점이 추가로 확인됨.
3. NEO 학습 데이터의 nn=20은 2k 해상도에서는 문제가 없음. 하지만 더 큰 N에서 nn=20으로 만든 질량을 NEO 입력으로 쓰면 위험함.

### Part 0 결론 (명시)

> **robust_laplacian point-cloud Laplacian은 무작위 샘플에서 nn=20일 때 N이 커지면 불안정. 준균일 샘플링 또는 nn≥30 필요.**

근거는 단위 구에서 저차 8모드의 평균 편향이며, 괄호 안은 질량합/면적임.

- **무작위 균일 샘플, nn=20:** N이 커질수록 붕괴함.
  - N=2048: −6.1% (0.97), N=4000: −6.2% (0.97)
  - N=8000: **−20.9% (0.73)**, N=16000: **−38.7% (0.48)**
- **같은 무작위 샘플, nn=30:** N=2048→16000에서 −5.2% / −5.1% / −5.0% / −6.2%이고 질량합은 ≈1.0으로 안정적임. 남는 −5%는 곡면 보정 c로 약 −1%까지 줄어듦(N=8000에서 l=1 −0.9%).
- **Fibonacci 준균일 샘플, nn=20:** N ≤ 4000에서는 −0.1%로 정확함. 준균일이어도 N=8000에서 −3.7%, N=16000에서 **−18.9% (0.83)**로 붕괴가 시작됨.
- **Fibonacci 준균일 샘플, nn=30:** N=2048–16000 전 구간에서 |편향| ≤ 0.14%임.
- **NEO 샘플의 구 (메쉬 면에서 무작위 샘플), nn=20, N=8000:** 보정 전 −19.8%, 질량합 0.73. c로 보정해도 −10.4%임(균일한 스케일 오차가 아니므로 보정되지 않음).

**stage1에 주는 함의:** stage1은 nn=20을 쓰지만 점이 FPS 준균일 N=4000이므로 안전한 영역에 있음. 실제 600개 형상에서 확인한 값:
- pc 질량/면적: 평균 1.001 (범위 0.998–1.003)
- 보정 계수 c: 1.007–1.016
- 보정된 pc 고유값 vs 메쉬 고유값: i=1..10 평균 −0.15% (범위 −0.67% ~ −0.07%), i=80..95 평균 −0.12%
- 자세한 수치는 `stage1/results/tables.md`의 "Basis quality" 표.
