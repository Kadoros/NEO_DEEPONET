# 코드 출처 정리 (neo_check / stage1)

**요약**
- NEO 저장소(`NEO/`)는 수정하지 않았음. 쓴 것은 사전학습 가중치, 모델 로더·순전파, 입력 정규화, QR 직교화, 평가 함수(neo_check 진단용)와 CLI 스크립트 2개(`sample_points.py`, `infer.py`)뿐임.
- 그 밖의 코드(데이터 생성, 메쉬 cotan 라플라시안, 기저 계산 파이프라인, 모든 학습 모델, 학습·평가·보고)는 직접 구현했음.
- **`sample_points.py` 수정 사본은 만들지 않았음.** `axis=1` 정규화가 의심된다는 점은 NEO_CHECK.md에 적기만 했고 고치지는 않았음(§1.4).
- DON-base는 직접 구현했으며 파라미터는 256,065개임. 외부 구현을 가져오지 않았고, 코드나 보고서에 출처 표기도 없음. 구조는 DeepONet(branch·trunk 내적 + 스칼라 편향)과 PointNet식 집합 인코더를 따름(§2).
- stage1의 학습 모델은 모두 `torch.nn` 기본 층으로만 만들었음. **torch_cluster와 torch_geometric은 stage1 모델에 쓰이지 않았음.** NEO 패키지를 import할 때 필요한 의존성일 뿐임(§3).

이 문서를 만들면서 코드는 수정하지 않았음. 확인에 쓴 명령은 `grep`/`find`, `git status`, `importlib.metadata`, DON-base 파라미터 수 출력(scratchpad의 임시 스크립트)임.

---

## 1. NEO 관련

### 1.1 저장소와 설치 상태

- `NEO/` = `github.com/Adversarr/NEO`, `main @ d928c319d3f1` (2026-05-26 커밋). `git status` 결과 로컬 변경 없음.
- `neo` 환경에 `g2pt 0.1.0`이 **editable**로 설치되어 있음(위치: `NEO/`). 그래서 stage1의 `from g2pt ...`는 저장소 원본 코드를 그대로 실행함.
- 사전학습 가중치는 README의 Google Drive에서 받아 `neo_check/ckpts/`에 둔 원본 그대로임.
  - `Ours-Large.ckpt` sha256 `06c9b845…`
  - `Ours-Small-NoRoPE.ckpt` sha256 `b745be5f…`
  - stage1은 `Ours-Large`만 사용하며, 추론만 하고 학습이나 미세조정은 하지 않음.

### 1.2 NEO에서 가져다 쓴 부분 (수정 없음)

| NEO 쪽 심볼 (파일) | 사용한 곳 | 용도 |
|---|---|---|
| `PretrainTraining.load_from_checkpoint` (`src/g2pt/training/pretrain.py`, Lightning) | `stage1/neo_basis.py`, `neo_check/check_rr.py` | 체크포인트 로드 |
| `PretrainTraining.model` = `TransolverExpModel` (`src/g2pt/neuralop/model/transolver_experimental.py`, 하위 층 `neuralop/layers/attn/transolver_experimental.py`, `rope.py`, `mlps.py`, `norms.py`) | `stage1/neo_basis.py`: `model.model(x, x, mass)` 직접 호출 | NEO 원출력 F (N×256) |
| `PretrainTraining.forward` (`pretrain.py:123`) | `neo_check/check_rr.py` | F → QR (저장소 경로 그대로) |
| `normalize_pc` (`src/g2pt/data/transforms.py:6`) | `neo_basis.py`, `check_rr.py` | 중심 이동 + max\|coord\| = 1 |
| `qr_orthogonalization` (`src/g2pt/utils/ortho_operations.py:90`) | `neo_basis.py` | 질량 가중 QR → NEO span S |
| `solve_gev_from_subspace_cuda`, `solve_gev_ground_truth` (`src/g2pt/utils/gev.py`) | `check_rr.py`만 | 저장소 RR과 ARPACK 기준값 (비교 진단) |
| `scripts/sample_points.py` (CLI) | neo_check 입력 `samples*/<shape>/sample_points.npy` | 메쉬 → 점 샘플 |
| `exp/pretrain/infer.py` (CLI) | neo_check `results/*.json` | NEO 추론 + 평가 |

**stage1에서 NEO 순전파를 쓰는 방식.** `neo_basis.py`는 `PretrainTraining.forward`를 호출하지 않고, 그 내부 두 줄을 그대로 따라 씀.
- `pretrain.py:129`의 `self.model(x, x, mass)`
- `pretrain.py:144`의 `qr_orthogonalization(y, mass)`
- 이렇게 한 이유는 직교화 전의 원출력 F를 따로 저장하기 위해서임. F는 ε_θ의 trunk 입력으로 쓰는 F̂이 됨.

### 1.3 직접 구현한 부분

**neo_check/**

| 파일 | 함수/내용 |
|---|---|
| `summarize.py` | `infer.py` 결과 JSON을 모아 `results/summary.{md,csv}` 생성. span loss와 고유값 오차 집계 |
| `check_rr.py` | 저장소 RR(diag SᵀMS, float32)과 dense SᵀMS·float64 `scipy.linalg.eigh` RR 비교. NEO 모델 호출과 L, M은 `infer.py`와 같게 맞춤 |
| `sphere_calib.py` | 곡면 강성 보정 c = 2·ΣM / Σᵢ xᵢᵀLxᵢ, 구 해석해 l(l+1)/R² 대비 편향 |
| `meshes/sphere.obj`, `torus.obj` | trimesh로 생성 (`suzanne.obj`는 저장소의 `pyfastspectrum/suzanne.obj`) |

**stage1/**

| 파일 | 함수 | 비고 |
|---|---|---|
| `common.py` | 설정 상수, 캐시 경로 고정 | |
| `shapes.py` | `fps`, `_orient`, `check_mesh`, `angles`, `_real_sh`, `sh_sphere_radius`, `superquadric_radius`, `star_mesh`, `torus_mesh`, `face_areas`, `normalise`, `make_shape`, `cotan_laplacian` | 형상 3종 생성, 메쉬 생성, P1 cotan 강성과 lumped 질량. FPS는 numpy로 직접 구현했고 torch_cluster의 `fps`는 쓰지 않음. 구면조화는 `scipy.special.sph_harm_y` 사용 |
| `make_dataset.py` | `initial_conditions`, `build`, `main` | 정답 `expm_multiply`, 메쉬/pc ARPACK, 보정 c, 분할 |
| `neo_basis.py` | `main` | NEO 호출 파이프라인(§1.2의 함수 사용). RR, F̂·z 계산, span loss는 직접 구현 |
| `eps_precompute.py` | `work`, `main` | r = u0 − YYᵀMu0, r_p = (M + tL)⁻¹Mr (`splu`), PC-discrete 해 |
| `models.py` | `mlp`, `log_t_feature`, `Galerkin`, `SpectralMultiplier`, `Correction`, `LearnedSpectral`, `DeepONetBase`, `n_params`, `ResidualEps`, `LNeoResidual` | 모든 학습 모델 |
| `train.py`, `train_eps.py` | `canon_sign`, `load_all`, `batch`, `rel_err`, `evaluate`, `train`, `time_inference`, `continue_don` 등 | 학습과 평가 |
| `report.py`, `report_eps.py` | 표·그림 생성 | |

**`neo_basis.py`가 저장소 `infer.py`와 다른 점**

| 항목 | `infer.py` | `stage1/neo_basis.py` |
|---|---|---|
| 네트워크 입력 질량의 n_neighbors | 30 (기본값) | **20** (NEO 학습 설정, `mesh_feats.py:71`) |
| RR 연산자 | 정규화 좌표에서 만든 L, 보정 없음 | **보정한 c·L**을 면적 정규화 좌표 V에서 만듦(G-exact와 같은 연산자), M = M_pc |
| RR 계산 | diag(SᵀMS), float32, GPU | dense SᵀLS·SᵀMS, float64, `scipy.linalg.eigh` |
| 결과 | 부분공간 전체 | 256개 Ritz 쌍 중 하위 96개 (Φ_neo, λ_neo) |

### 1.4 `sample_points.py` 수정 사본 — **만들지 않았음**

- 근거
  - `find`로 찾은 `sample_points*.py`는 `NEO/scripts/sample_points.py` 하나뿐임.
  - `neo_check/`의 `.py`는 `summarize.py`, `check_rr.py`, `sphere_calib.py` 3개임.
  - `NEO/`는 `git status`가 깨끗함.
- neo_check의 샘플(`samples/`, `samples_n8000/`)은 모두 **저장소 원본 스크립트**로 만들었음. 명령은 NEO_CHECK.md §4에 있음.
- NEO_CHECK.md §5의 의심 사항은 기록만 했고 고치지 않았음.
  - 대상 코드는 `NEO/scripts/sample_points.py:148–149`임: `norm = Σ_k φ_ik² m_i` (`axis=1`, 점별) → `evecs_on_points / sqrt(norm)`.
  - 영향은 neo_check의 mesh↔pc, mesh↔net 코사인 점수에 한정됨. span loss와 고유값 오차는 영향받지 않음.
- stage1은 `sample_points.py`를 쓰지 않음. 점은 메쉬 정점 그대로이고, 메쉬 고유벡터는 `make_dataset.py`에서 `eigsh`로 직접 구함. 따라서 이 문제는 stage1 결과와 무관함.

---

## 2. DON-base 구조 (`stage1/models.py:100` `DeepONetBase`)

### 2.1 입력

| 기호 | 형태 | 내용 |
|---|---|---|
| x | (B, N, 3) | 메쉬 정점 = 점 좌표. `shapes.normalise`로 면적 중심을 원점에 두고, 총면적 4π로 스케일하고, 무작위 회전함. NEO의 `normalize_pc`는 적용하지 않음 |
| u0 | (B, I, N) | 초기조건 원값(가우시안 합, 스케일 조정 없음) |
| m_pc | (B, N) | point-cloud lumped 질량 (branch 평균 풀링의 가중치) |
| t | (B, T) | 시간. 특징은 (log t + 3)/3 |

N = 4000, I = 4 (IC), T = 8 (τ). 연결 정보, 기저, NEO 특징은 쓰지 않음.

### 2.2 층 구성

모든 `nn.Linear`는 `bias=True`이고 활성함수는 SiLU임. `mlp()` 헬퍼는 **마지막 Linear 뒤에 활성함수를 붙이지 않음**.

**Branch — PointNet식 집합 인코더 (IC마다 하나의 벡터)**

| # | 층 | 입력 → 출력 | 활성 | 파라미터 |
|---|---|---|---|---|
| 1 | 점별 Linear (`point.0`) | [x, u0(x)] 4 → 64 | SiLU | 320 |
| 2 | 점별 Linear (`point.2`) | 64 → 128 | SiLU | 8,320 |
| 3 | 점별 Linear (`point.4`) | 128 → 256 | 없음 | 33,024 |
| – | 풀링 | N×256 → 512 = [Σ wᵢhᵢ ‖ maxᵢ hᵢ], wᵢ = mᵢ/Σm | – | 0 |
| 4 | Linear (`head.0`) | 512 → 256 | SiLU | 131,328 |
| 5 | Linear (`head.2`) | 256 → **p = 128** | 없음 | 32,896 |

**Trunk — 점·시간별 MLP**

| # | 층 | 입력 → 출력 | 활성 | 파라미터 |
|---|---|---|---|---|
| 1 | `trunk.0` | [x, (log t+3)/3] 4 → 128 | SiLU | 640 |
| 2 | `trunk.2` | 128 → 128 | SiLU | 16,512 |
| 3 | `trunk.4` | 128 → 128 | SiLU | 16,512 |
| 4 | `trunk.6` | 128 → **p = 128** | 없음 | 16,512 |

**출력:** û(t, x; u0) = Σ_{k=1..128} b_k(u0) · τ_k(x, t) + b₀.
- b₀는 학습되는 스칼라 편향 1개이며 0으로 초기화함.
- 출력은 (B, T, I, N).

### 2.3 요약 수치

| 항목 | 값 |
|---|---|
| 층 수 | branch Linear 5개 (점별 3 + head 2), trunk Linear 4개 |
| 폭 | branch 64-128-256 → (풀링 512) → 256 → 128, trunk 128×3 → 128 |
| 출력 차원 p | 128 |
| 편향 항 | 모든 Linear에 bias, 출력 스칼라 편향 b₀ 1개 |
| 정규화 층 | **없음** (BatchNorm·LayerNorm·Dropout 없음). 입력은 t에만 로그 스케일을 적용하고 x와 u0는 원값 |
| 초기화 | PyTorch 기본값 (Linear는 Kaiming-uniform). ε_θ와 달리 마지막 층을 0으로 초기화하지 않음 |
| 파라미터 수 | **256,065** = point 41,664 + head 164,224 + trunk 50,176 + b₀ 1 (실행해서 확인. REPORT.md 표와 같음) |

**학습**
- 손실: 메쉬 질량 가중 상대 L2.
- 1차: AdamW (lr 1e-3, wd 1e-5) + OneCycle (pct 0.05), 20k step, batch 8 형상 × 4 IC × 8 τ, grad clip 1, 250 step마다 val로 최선 가중치 선택 (`train.py`).
- 2차: 1차 최선 가중치에서 이어서 학습함 (`train_eps.py:continue_don`).
  - AdamW 3e-4, ReduceLROnPlateau (factor 0.5, patience 4, min 1e-6).
  - 1000 step마다 평가하고, 상대 개선이 0.5% 미만인 평가가 20회 이어지면 멈춤.
  - 총 172k step.

### 2.4 참고한 논문·코드

- **코드: 없음.** 외부 구현을 복사하거나 옮겨 오지 않았음. `models.py`의 모든 모델은 같은 파일의 `mlp()` 헬퍼와 `torch.nn` 기본 층으로만 만들었음. 코드 주석, REPORT.md, REPORT_eps.md 어디에도 출처 표기가 없음.
  - 쓰지 않은 것: DeepXDE, NEO의 `pointnet2.py`(PyG `PointNetConv` 기반), `torch_geometric`의 PointNet 계열 층.
- **개념상 대응하는 원 논문** (구조가 해당하는 표준 개념이며, 구현을 참고한 것은 아님)
  - DeepONet: Lu, Jin, Pang, Zhang, Karniadakis, "Learning nonlinear operators via DeepONet…", *Nature Machine Intelligence* 3, 218–229 (2021). unstacked branch/trunk 내적과 스칼라 편향 b₀.
  - PointNet: Qi, Su, Mo, Guibas, "PointNet: Deep Learning on Point Sets…", CVPR 2017. 점별 공유 MLP와 대칭 풀링.
- **원형과 다른 점**
  - DeepONet: branch 입력이 고정 센서 값 벡터가 아니라 점 집합 {(x, u0(x))}임. 형상마다 점 위치가 달라서 집합 인코더를 씀.
  - PointNet과 다른 점
    - T-Net(입력·특징 변환)이 없음.
    - BatchNorm이 없음.
    - ReLU 대신 SiLU를 씀.
    - 풀링 직전 층(128→256)에 활성함수가 없음.
    - max 풀링에 질량 가중 평균 풀링을 이어 붙임.
  - 따라서 "PointNet"은 PointNet 스타일 인코더라는 뜻으로 읽어야 함.

---

## 3. 1단계 외부 라이브러리 중 모델 구성 요소로 쓴 것

환경은 conda `neo` (Python 3.12.14)임. 버전은 `importlib.metadata`와 `pip show`로 확인함.

### 3.1 학습·추론 경로에 들어가는 것

| 라이브러리 | 버전 | 출처 (설치 경로) | 모델에서의 역할 |
|---|---|---|---|
| PyTorch | 2.6.0+cu126 (CUDA 12.6) | pip, `download.pytorch.org/whl/cu126` (NEO `uv.lock`과 같은 조합으로 재설치. NEO_CHECK.md §2) | 모든 stage1 모델의 층: `nn.Linear`, `nn.SiLU`, `F.softplus`, `F.scaled_dot_product_attention`(`ResidualEps` branch의 교차 어텐션), AdamW와 스케줄러 |
| NEO (`g2pt`) | 0.1.0, 커밋 `d928c31` | editable 설치, `github.com/Adversarr/NEO` | **고정된(학습하지 않는) 특징 추출기.** `TransolverExpModel` + `Ours-Large.ckpt` → F̂, z, NEO span. 내부 구조(Transolver 슬라이스 어텐션, RoPE 등)는 NEO 코드 그대로임 |
| Lightning | 2.6.6 | pip (PyPI) | NEO 체크포인트 로드(`LightningModule.load_from_checkpoint`)에만 사용. stage1 학습 루프는 직접 작성한 순수 PyTorch 코드 |
| robust_laplacian | 1.1.0 | pip (PyPI) 휠. 원 코드 `github.com/nmwsharp/robust-laplacians-py` (Sharp & Crane, "A Laplacian for Nonmanifold Triangle Meshes", SGP 2020) | 학습 파라미터는 없지만 모델 입력을 만듦: pc 라플라시안 L과 질량 M(n_neighbors=20) → NEO 질량 입력, Φ_ex, RR, r_p, PC-discrete, branch 풀링 가중치 m_pc |
| SciPy | 1.18.1 | pip (PyPI) | 모델 입력과 정답 계산: `eigsh`(ARPACK), `linalg.eigh`(RR), `splu`(r_p), `expm_multiply`(정답, PC-discrete). 형상 생성: `ConvexHull`, `Delaunay`, `sph_harm_y`, `Rotation` |

### 3.2 모델에 쓰이지 않는 것

| 라이브러리 | 버전 | 출처 | 실제 역할 |
|---|---|---|---|
| **torch_cluster** | 1.6.3+pt26cu126 | pip, `data.pyg.org/whl/torch-2.6.0+cu126.html` (NEO `requirements-pyg.txt`). 원 코드 `github.com/rusty1s/pytorch_cluster` | **stage1과 neo_check의 어떤 모델에도 쓰이지 않음.** `g2pt/neuralop/model/__init__.py`가 `pointnet2.py`를 import하고, `pointnet2.py:12–13`이 torch_cluster가 없으면 `quit()`함. 그래서 NEO를 import하려면 설치되어 있어야 함. `TransolverExpModel`의 순전파는 torch_cluster를 호출하지 않음 |
| torch_geometric | 2.7.0 | pip (PyPI) | 위와 같음. NEO import 때만 필요함 |
| PointNet 구현 | – | – | **외부 구현을 쓰지 않았음.** DON-base의 집합 인코더는 직접 구현(§2). NEO `pointnet2.py`는 import만 되고 사용하지 않음 |
| trimesh | 5.1.0 | pip (PyPI) | `shapes.py`에서 icosphere 방향 집합(40962점)을 만들 때만 사용. neo_check의 구·토러스 메쉬 생성 |
| NumPy / pandas / Matplotlib | 2.5.3 / 3.0.6 / 3.11.2 | pip (PyPI) | 데이터 처리, 표, 그림 |
