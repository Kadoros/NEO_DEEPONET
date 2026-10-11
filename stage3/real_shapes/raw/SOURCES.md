# 실제 형상 원본 (2026-09-27 다운로드)

| 모델 | 출처 (허용 경로) | 파일 | 정점 / 삼각형 | 닫힘 여부 |
|---|---|---|---|---|
| Stanford Bunny | Open3D 예제 데이터 (`o3d.data.BunnyMesh`, github isl-org/open3d_downloads) | `open3d/extract/BunnyMesh/BunnyMesh.ply` | 35,947 / 69,451 | 열린 모서리 223개 (바닥 구멍) |
| Armadillo | Open3D 예제 데이터 (`o3d.data.ArmadilloMesh`) | `open3d/extract/ArmadilloMesh/ArmadilloMesh.ply` | 172,974 / 345,944 | 닫힘 (다양체) |
| Stanford Dragon | pyvista 예제 데이터 (`pyvista.examples.download_dragon`, 캐시는 이 폴더) | `pyvista/dragon.ply` | 437,645 / 871,414 | 열린 모서리 4,270개 |
| **Spot (소)** | **못 받음.** Open3D·libigl·pyvista 예제 데이터에 없고, Thingi10k(3D 프린팅 모델)에도 없음 | — | — | — |

- 두 가지 유의 사항:
  - libigl 예제 데이터에도 bunny와 armadillo가 있음 (대체 원본으로 쓸 수 있음). 또 `cow.off`가 있는데, Spot과는 다른 모델임.
  - Open3D 다운로드 때 파일은 즉시 받아졌으나, 그 뒤 파이썬 프로세스가 끝나지 않아 10분 시간 초과로 종료됨. 파일은 크기와 읽기 모두 정상.
- 라이선스: Stanford 3D Scanning Repository 모델은 연구용 사용이 허용됨. 논문에 쓸 때 출처를 명시해야 함.

## Thingi10K 10개 (2026-09-27, `c_select_thingi.py`, SEED 20260927)

**출처:** HuggingFace `Thingi10K/Thingi10K` v1.5.0. 메타데이터 CSV 3개와 선택된 `npz/<file_id>.npz`만 받음 (전체 22 GB는 받지 않음).

**선택 기준:**
- 닫힘(경계 모서리 0), 정점·모서리 다양체, 방향 일관, 연결 성분 1개
- 자기 교차 없음, 퇴화·중복 면 없음
- 면 수 2k–200k, Creative Commons 라이선스
- 적격 1,750개 (genus 0: 682, genus 1: 359, genus 2–10: 493)
- genus 층별 무작위 추출: 0에서 4개, 1에서 3개, 2 이상에서 3개

| 층 | file_id | Thing ID | genus | 정점 | 면 | 라이선스 |
|---|---|---|---|---|---|---|
| g0 | 53405 | 16229 | 0 | 1403 | 2802 | Creative Commons - Attribution - Share Alike |
| g0 | 101619 | 35425 | 0 | 11546 | 23088 | Creative Commons - Attribution |
| g0 | 103141 | 36059 | 0 | 8391 | 16778 | Creative Commons - Attribution - Share Alike |
| g0 | 135364 | 51376 | 0 | 1308 | 2612 | Creative Commons - Attribution - Share Alike |
| g1 | 37881 | 11836 | 1 | 1700 | 3400 | Creative Commons - Attribution - Share Alike |
| g1 | 41732 | 12858 | 1 | 7587 | 15174 | Creative Commons - Attribution - Share Alike |
| g1 | 1146172 | 713815 | 1 | 10423 | 20846 | Creative Commons - Attribution - Non-Commercial |
| g2+ | 472052 | 259005 | 2 | 66152 | 132308 | Creative Commons - Attribution |
| g2+ | 702401 | 42198 | 5 | 3568 | 7152 | Creative Commons - Attribution - Share Alike |
| g2+ | 814677 | 492429 | 4 | 2410 | 4832 | Creative Commons - Attribution - Non-Commercial |

**부피화 결과:**
- thingi135364(genus 0)는 벽 두께가 약 0.1로, N = 8192의 점 간격보다 얇음. 표면 점만으로 N을 넘어서 부피 포인트 클라우드를 만들 수 없음 → 제외 (해상도 한계).
- 나머지 9개는 사용함.

**Spot 대체:** libigl 예제 데이터의 `cow.off`를 사용함 (사용자 결정).
