"""kappa-network input comparison (gate v2 data, N = 200, seeds 0/7, test-time refined basis)
-> gate/REPORT_kappa_inputs.md, gate/kappa_inputs.png

    python stage3/gate_feat_report.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "gate", "runs_v2")
VARIANTS = [("", "(i) 깊이 + NEO 특징"), ("_hks_d", "(ii) 깊이 + HKS"), ("_hks", "(iii) HKS만 (깊이 제외)")]
THIN = "thingi472052"


def main():
    L = ["# κ̂ 네트워크 입력 비교 (관문 v2 데이터, N = 200, 시드 0·7 평균, 시험 시 다듬은 NEO 기저)", "",
         "**정보량 비교:** 세 변형 모두 κ를 입력으로 받지 않음. (i)·(ii)는 깊이 d(경계까지 거리)를 받는데, 이는 Point-DeepONet이 "
         "trunk에서 받는 SDF 입력(우리 데이터 로더에서 같은 d)과 같은 정보임. 즉 (i)·(ii)와 Point-DeepONet은 같은 기하 정보로 비교됨. "
         "(iii)은 깊이를 빼고 HKS(열 커널 서명, 회전·이동 불변, 6개 척도)만 받으므로 Point-DeepONet보다 적은 명시적 정보를 받음.", ""]
    rows = []
    for suf, lab in VARIANTS:
        for s in (0, 7):
            d = os.path.join(RUNS, f"ours_N200_s{s}{suf}")
            for fname, basis in (("test.csv", "원시"), ("test_refined.csv", "다듬은")):
                p = os.path.join(d, fname)
                if not os.path.exists(p):
                    continue
                t = pd.read_csv(p)
                t = t[t.model == "ours"].groupby(["split", "shape"], as_index=False).rel.mean()
                t["variant"], t["seed"], t["basis"] = lab, s, basis
                rows.append(t)
    D = pd.concat(rows)
    for excl in (False, True):
        x = D[D["shape"] != THIN] if excl else D
        T = x.groupby(["variant", "basis", "seed", "split"]).rel.mean().groupby(["variant", "basis", "split"]).mean().unstack("split")
        L.append(f"## 평균 상대 L2 ({'thingi472052 제외' if excl else '전체'})\n")
        L.append(T[["test_in", "test_ood", "test_real"]].rename(columns={"test_in": "학습 형상군", "test_ood": "합성 OOD",
                                                                         "test_real": "실제 형상"}).to_markdown(floatfmt=".4f"))
        L.append("")
    L.append("참고 (같은 데이터): Point-DeepONet N=200 — 학습 형상군 17.0%, 합성 OOD 34.3%, 실제 형상 32.9%; "
             "G(κ=1) 다듬은 NEO — 34.6% / 18.2% / 12.2%; 정답 κ를 넣은 G(다듬은 NEO) 상한 — 1.04% / 0.44% / 0.33%.\n")
    L.append("## 형상별 (다듬은 기저, 시드 평균)\n")
    S = D[D.basis == "다듬은"].groupby(["split", "shape", "variant"]).rel.mean().unstack("variant")
    L.append(S.to_markdown(floatfmt=".4f"))
    txt = "\n".join(L) + "\n"
    open(os.path.join(ROOT, "gate", "REPORT_kappa_inputs.md"), "w").write(txt)
    print(txt)
    figure()


def figure():
    fig, ax = plt.subplots(3, 3, figsize=(16, 13), sharex=True, sharey=True)
    dd = np.linspace(0, 0.45, 200)
    col = {"test_in": "#2a78d6", "test_ood": "#eb6834", "test_real": "#1baf7a"}
    titles = {"test_in": "training families (unseen shapes)", "test_ood": "synthetic OOD", "test_real": "real shapes"}
    names = {"": "(i) depth + NEO features", "_hks_d": "(ii) depth + HKS", "_hks": "(iii) HKS only (no depth)"}
    for r, (suf, _) in enumerate(VARIANTS):
        p = os.path.join(RUNS, f"ours_N200_s0{suf}", "kappa_hat.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        for c, split in enumerate(("test_in", "test_ood", "test_real")):
            a = ax[r, c]
            for k in z.files:
                if k.startswith(split + "__"):
                    d, kh, _ = z[k]
                    sel = np.random.default_rng(0).choice(len(d), min(1500, len(d)), replace=False)
                    a.scatter(d[sel], kh[sel], s=2, alpha=0.25, color=col[split])
            a.plot(dd, 1 + 3 / (1 + np.exp(-(dd - 0.12) / 0.03)), color="#0b0b0b", lw=2)
            a.set_title(f"{names[suf]} — {titles[split]}", loc="left", fontsize=9)
            a.grid(True, color="#e4e3df"); a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            if r == 2:
                a.set_xlabel("depth d (distance to the boundary)")
            if c == 0:
                a.set_ylabel("κ̂  (black: true rule)")
    fig.suptitle("κ̂(x) recovered with three κ-network inputs (N=200, seed 0; points = cloud nodes; κ never given)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "gate", "kappa_inputs.png"), dpi=100, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
