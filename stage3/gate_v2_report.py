"""Gate v2 report (task B) -> stage3/gate/REPORT_gate_v2.md, gate/kappa_hat_v2.png

    python stage3/gate_v2_report.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "gate", "runs_v2")
DATA = os.path.join(ROOT, "gate", "data_v2")
THIN = "thingi472052"
SPLITS = ("test_in", "test_ood", "test_real")
CRIT = ("**판정 기준 (v2):** (1) 주 기준: 처음 보는 형상(합성 OOD / 실제 형상 각각)에서의 절대 오차 — 우리 vs Point-DeepONet vs G. "
        "(2) 배수 기준(우리 증가 ≤ 2배, Point-DeepONet ≥ 5배, 우리가 ≥ 3배 정확)은 해당 모델의 학습 형상군 오차가 5% 이하일 때만 적용. "
        "(3) 별도 항목: 우리 학습 모델이 G보다 나은가. (4) 극단적으로 얇은 형상 thingi472052는 포함 / 제외 둘 다 표기.")


def baselines():
    rows = []
    for split in SPLITS:
        for p in sorted(glob.glob(os.path.join(DATA, split, "*.npz"))):
            z = np.load(p)
            U, m = z["U"], z["m"][None, None]
            for key, lab in (("U_g_raw", "G (κ=1), 원시 NEO"), ("U_g_ref", "G (κ=1), 다듬은 NEO"),
                             ("U_g_oracle_raw", "G, 정답 κ, 원시 NEO [상한]"), ("U_g_oracle_ref", "G, 정답 κ, 다듬은 NEO [상한]")):
                if key in z:
                    e = np.sqrt((m * (z[key] - U) ** 2).sum(-1) / (m * U ** 2).sum(-1))
                    rows.append(dict(model=lab, N=0, seed=0, split=split, shape=os.path.basename(p)[:-4], rel=float(e.mean())))
    return pd.DataFrame(rows)


def learned():
    rows, runs = [], []
    for d in sorted(glob.glob(os.path.join(RUNS, "*_N*_s*"))):
        rp = os.path.join(d, "result.json")
        if not os.path.exists(rp):
            continue
        r = json.load(open(rp))
        t = pd.read_csv(os.path.join(d, "test.csv"))
        if "model" not in t:
            t["model"] = r["model"]
        t["model"] = t.model.replace({"pdon": "Point-DeepONet", "ours": "우리 (κ̂ + ε)", "ours-kappa": "우리 (κ̂만)"})
        t["N"], t["seed"] = r["N"], r["seed"]
        rows.append(t.groupby(["model", "N", "seed", "split", "shape"], as_index=False).rel.mean())
        runs.append(dict(model=r["model"], N=r["N"], params=r["params"], t_train_s=r["t_train_s"]))
    return pd.concat(rows), pd.DataFrame(runs)


def table(D, excl):
    x = D[D["shape"] != THIN] if excl else D
    per = x.groupby(["model", "N", "seed", "split"]).rel.mean().unstack("split")
    return per.groupby(["model", "N"]).mean()


def main():
    L, R = learned()
    B = baselines()
    D = pd.concat([L, B])
    out = ["# 관문 실험 v2 (과제 B: 숨은 열전도도, 공통 규칙)", "", CRIT, ""]
    for excl, lab in ((False, "thingi472052 포함"), (True, "thingi472052 제외")):
        T = table(D, excl)
        out.append(f"## 평균 상대 L2 오차 ({lab}; 학습 모델은 시드 2개 평균)\n")
        out.append("| 모델 | N | 학습 형상군 | 합성 OOD | 실제 형상 | 합성/학습 | 실제/학습 |")
        out.append("|---|---|---|---|---|---|---|")
        for (m, N), row in T.iterrows():
            out.append(f"| {m} | {N} | {row.test_in:.4f} | {row.test_ood:.4f} | {row.test_real:.4f} | "
                       f"{row.test_ood / row.test_in:.2f}× | {row.test_real / row.test_in:.2f}× |")
        out.append("")
    T = table(D, False)
    T2 = table(D, True)
    out.append("## 판정\n")
    out.append("### (1) 주 기준: 처음 보는 형상에서의 절대 오차 순위\n")
    for N in (50, 200):
        for split, lab in (("test_ood", "합성 OOD"), ("test_real", "실제 형상")):
            cand = {m: T.loc[(m, n), split] for (m, n) in T.index if n in (0, N) and "상한" not in m}
            rank = sorted(cand.items(), key=lambda kv: kv[1])
            out.append(f"- N = {N}, {lab}: " + " < ".join(f"{m} {v:.4f}" for m, v in rank))
    out.append("\n### (2) 배수 기준 (학습 형상군 오차 ≤ 5%인 모델에만 적용)\n")
    out.append("| N | 대상 | 우리 증가 | PDON 학습 오차 | PDON 증가 | PDON / 우리 | 결과 |")
    out.append("|---|---|---|---|---|---|---|")
    for N in (50, 200):
        if ("우리 (κ̂ + ε)", N) not in T.index or ("Point-DeepONet", N) not in T.index:
            continue
        o, p = T.loc[("우리 (κ̂ + ε)", N)], T.loc[("Point-DeepONet", N)]
        for split, lab in (("test_ood", "합성 OOD"), ("test_real", "실제 형상")):
            ox, px, ratio = o[split] / o.test_in, p[split] / p.test_in, p[split] / o[split]
            p_ok = p.test_in <= 0.05
            res = [ox <= 2 if o.test_in <= 0.05 else None, (px >= 5) if p_ok else None, ratio >= 3]
            txt = " / ".join("✓" if r else ("✗" if r is not None else "적용 안 함") for r in res)
            out.append(f"| {N} | {lab} | {ox:.2f}× | {p.test_in:.3f} | {px:.2f}× {'' if p_ok else '(적용 안 함)'} | {ratio:.1f}× | {txt} |")
    out.append("\n### (3) 우리 학습 모델이 G보다 나은가 (G = κ=1 Galerkin, 같은 원시 NEO 기저)\n")
    out.append("| N | 모델 | 학습 형상군 | 합성 OOD | 실제 형상 | 실제 형상 (thin 제외) |")
    out.append("|---|---|---|---|---|---|")
    g, g2 = T.loc[("G (κ=1), 원시 NEO", 0)], T2.loc[("G (κ=1), 원시 NEO", 0)]
    for N in (50, 200):
        for m in ("우리 (κ̂만)", "우리 (κ̂ + ε)"):
            if (m, N) in T.index:
                r, r2 = T.loc[(m, N)], T2.loc[(m, N)]
                cell = lambda a, b: f"{a:.4f} vs {b:.4f} {'✓' if a < b else '✗'}"
                out.append(f"| {N} | {m} | {cell(r.test_in, g.test_in)} | {cell(r.test_ood, g.test_ood)} | "
                           f"{cell(r.test_real, g.test_real)} | {cell(r2.test_real, g2.test_real)} |")
    out.append("\n## 형상별 (N = 200, 시드 평균)\n")
    S = D[D.N.isin([0, 200])].groupby(["split", "shape", "model"]).rel.mean().unstack("model")
    out.append(S.to_markdown(floatfmt=".4f"))
    if len(R):
        out.append("\n## 비용\n")
        out.append(R.groupby(["model", "N"]).agg(params=("params", "first"), t_train_s=("t_train_s", "mean")).to_markdown(floatfmt=".3g"))
    txt = "\n".join(out) + "\n"
    open(os.path.join(ROOT, "gate", "REPORT_gate_v2.md"), "w").write(txt)
    print(txt)
    kappa_figure()


def kappa_figure():
    runs = sorted(glob.glob(os.path.join(RUNS, "ours_N200_s*", "kappa_hat.npz")))
    if not runs:
        return
    z = np.load(runs[0])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    dd = np.linspace(0, 0.45, 200)
    colors = {"test_in": "#2a78d6", "test_ood": "#eb6834", "test_real": "#1baf7a"}
    titles = {"test_in": "training families (unseen shapes)", "test_ood": "synthetic OOD families", "test_real": "real shapes"}
    for a, split in zip(ax, ("test_in", "test_ood", "test_real")):
        for k in z.files:
            if k.startswith(split + "__"):
                d, kh, kt = z[k]
                sel = np.random.default_rng(0).choice(len(d), min(1500, len(d)), replace=False)
                a.scatter(d[sel], kh[sel], s=2, alpha=0.25, color=colors[split])
        a.plot(dd, 1 + 3 / (1 + np.exp(-(dd - 0.12) / 0.03)), color="#0b0b0b", lw=2, label="true rule κ(d)")
        a.set_title(f"{titles[split]}: recovered κ̂ vs depth", loc="left", fontsize=10)
        a.set_xlabel("depth d (distance to the boundary)")
        a.grid(True, color="#e4e3df"); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("conductivity κ")
    ax[0].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Gate v2: κ̂(x) recovered by our model ({os.path.basename(os.path.dirname(runs[0]))}; points = cloud nodes, "
                 f"never given κ) against the true rule", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "gate", "kappa_hat_v2.png"), dpi=110, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
