# Point-DeepONet 기준 모델 준비 (2026-09-27) — 준비만, 실험 없음

저장소: https://github.com/jangseop-park/point-deeponet (commit `a22c347`, 2026-01-19) → `baselines/point-deeponet/`
논문: Park & Kang, *Point-DeepONet*, Neural Networks 2026 (108560).

## 1. 설치 — 완료
- 새 conda 환경 **`pdon`** (python 3.10, conda-forge를 명령행 인자로만 지정, 서버 설정 변경 없음), 크기 6.0 GB.
- README의 `requirements.txt`에는 `pyvista` 한 줄뿐이라, 코드의 import에서 실제 의존성을 확인해 설치함:
  ```bash
  conda run -n pdon python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
  ```
  ```bash
  conda run -n pdon python -m pip install "numpy<2" scikit-learn matplotlib tqdm pyvista vtk deepxde
  ```
  결과: torch 2.3.1+cu121 (GPU 인식 확인), deepxde 1.15.0 (`dde.maps` 호환 확인), numpy 1.26.4, pyvista 0.49, vtk 9.7.

## 2. 논문 결과 재현 — **막힘 (우회하지 않음)**
- 예제 데이터가 저장소에 없고 Kaggle 데이터셋(`jangseop/point-deeponet-dataset`, CC BY-SA 4.0)에서 받아야 함.
  - **인증 필요:** Kaggle API 토큰(개인 계정의 `~/.kaggle/kaggle.json`) 없이는 다운로드가 404. 사용자 설정 파일 생성은 서버 규칙 밖이라 진행하지 않음.
  - **용량:** 전체 106.5 GB (파일 2141개, 대부분 VTK 메쉬). 학습에 최소로 필요한 `targets.npz` 21.2 GB + `xyzdmlc.npz` 47.7 GB = **약 69 GB** + `bracket_labels.csv`. 현재 디스크 여유 약 60 GB(93% 사용)라 **받을 수 없음**. 샘플링된 학습 파일(`Rpt0_N5000.npz` 등)은 이 두 파일에서 노트북으로 만들어야 해서 따로 받을 수도 없음.
  - 가능한 길: (a) 사용자가 Kaggle 토큰과 디스크(≥ 80 GB, 다른 볼륨 포함)를 마련, (b) 재현은 생략하고 논문 수치를 인용, (c) 저자에게 샘플링된 작은 파일 요청.
- **대신 한 것 — 동작 확인(재현 아님):** 저장소 형식과 같은 **인공 데이터**(사례 40개 × 500점, `data/sampled_SYNTHETIC/`, `data/npy/combined_40_split_random_train_valid.npz`)로 저장소 코드를 **수정 없이** 실행.
  - 학습 300스텝(GPU 0, 16초), 매개변수 251,936개(은닉 100), 검증 손실 0.166 → 0.0073, 부분 평가 R² 0.87–0.99 출력까지 정상.
  - 마지막 전체 해상도 평가에서 멈춤: 경로 `/root/Example3/data/npy/targets.npz`가 코드에 하드코딩되어 있고 VTK 메쉬가 필요함 (예상된 실패).
  - 로그 `baselines/pdon_smoke.log`, 출력 `point-deeponet/experiments_SYNTHETIC/`.

## 3. 우리 데이터를 넣으려면 바꿀 것

**저장소의 데이터 형식** (`Rpt{RUN}_N{N_pt}.npz`)
| 키 | 모양 | 내용 |
|---|---|---|
| `a` | (사례, N_pt, 9) | x, y, z, **d (SDF)**, m (질량), l (하중 크기), c (하중 방향 3성분) |
| `b` | (사례, N_pt, 4) | ux, uy, uz, von Mises |
| `c` | (사례,) | 이름 `"ver_101_428"` — 접두어(ver/hor/dia)가 **출력 클리핑 범위**를 고름 |
- 모델: branch = 전역 조건(m, l, c의 MLP) **+** PointNet(xyz만, 최대 풀링); trunk = xyz(사인 인코딩) + d; 출력 성분 수 **4로 하드코딩**; tanh 출력(MinMax −1~1 정규화 전제).

**필요한 변경 (저장소 코드는 그대로 두고 어댑터 스크립트에서 모델 클래스만 가져다 쓰는 방식 권장)**
1. **입력 형식:** 우리 문제의 입력은 전역 벡터가 아니라 **점마다의 함수**(원천 f(x), 초기조건, 계수장 κ(x))임.
   - 최소 변경: PointNet 입력을 `xyz` → `xyz + f(x)` 로 확장 (코드가 `process_branch_pointnet_input('xyz', ...)`로 xyz만 넣게 고정돼 있어 어댑터에서 교체). 전역 조건 branch에는 μ, ν, 시간 τ 같은 스칼라 매개변수를 넣음.
   - trunk 입력은 xyz + d 그대로 사용 가능.
2. **SDF(d):** 절차적 형상은 `neo_vol_data(2).make_sdf`의 해석적 SDF를 정규화 좌표에서 계산 (부호 규칙 확인 필요 — 저장소 데이터 없이 확인 불가, 문서화된 규칙도 없음). 실제 메쉬 형상(C 단계)은 메쉬에서 부호 있는 거리 계산(예: trimesh) 필요.
3. **출력 차원:** 스칼라 문제(확산·반응-확산·헬름홀츠)는 1, 탄성 변위는 3. `num_output_components = 4`와 ux/uy/uz/vm 전용 지표 함수 4개를 **출력 차원 인자로** 바꿔야 함.
4. **출력 클리핑 제거:** `process_output`이 이름 접두어별 고정 범위로 자름 → 우리 데이터에는 적용하면 안 됨.
5. **점 수:** 사례마다 같은 N_pt(부족하면 복원추출). 우리는 N = 8192(평가 16384); 메모리는 배치 × N_pt에 비례 (논문 설정 16 × 5000).
6. **분할·시드:** 분할 파일 경로가 `../data/npy/combined_{N}_split_…`로 고정, 시드가 `2024`로 고정 → D 단계(학습 샘플 수 N ∈ {10…500}, 시드 {0, 7, 17, 23, 42})를 위해 인자로 빼야 함.
7. **평가:** 하드코딩된 `/root/Example3/…`와 VTK 기반 평가를 우리 평가(같은 형상군/처음 보는 형상군/불규칙 샘플링)로 대체.
8. **학습 루프 주의:** `torch.autograd.set_detect_anomaly(True)`가 켜져 있어 느림(공정한 학습 시간 비교를 위해 끄는 것이 맞음 — 명시하고 끔); `ReduceLROnPlateau`가 학습 끝에 한 번만 호출되어 사실상 학습률 스케줄이 없음 → 원 코드 그대로 둘지 결정 필요(공정성: "원 논문 설정 그대로"가 기본).
9. **매개변수 수 맞추기(D):** 은닉 100에서 약 25만 개. 우리 구조와 맞추려면 은닉 폭 인자로 조정.

## 4. 현재 상태
- GPU: 동작 확인 때 GPU 0을 16초 사용, 지금은 비어 있음. 디스크 추가 사용: 환경 6.0 GB + 저장소·인공 데이터 48 MB.
- 다음 할 일(승인 후): 어댑터 `baselines/pdon_adapter.py` 작성(위 1–9 반영, 저장소 코드는 수정하지 않음) → 우리 확산 데이터로 소규모 확인.
