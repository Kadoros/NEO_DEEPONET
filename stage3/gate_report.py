"""Gate experiment report -> stage3/gate/REPORT_gate.md

    python stage3/gate_report.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "gate", "runs")
CRIT = ("**판정 기준 (사용자 지정):** 우리 OOD 증가 ≤ 2배, Point-DeepONet ≥ 5배, OOD에서 우리가 ≥ 3배 정확 → 진행 근거 충분.\n"
        "(OOD 증가 = 처음 보는 형상 오차 / 학습 형상군 오차. 합성 OOD와 실제 형상을 따로 판정.)")


def main():
    rows, runs = [], []
    for d in sorted(glob.glob(os.path.join(RUNS, "*_N*_s*"))):
        rp = os.path.join(d, "result.json")
        if not os.path.exists(rp):
            continue
        r = json.load(open(rp))
        t = pd.read_csv(os.path.join(d, "test.csv"))
        t["model"], t["N"], t["seed"] = r["model"], r["N"], r["seed"]
        rows.append(t)
        runs.append(dict(model=r["model"], N=r["N"], seed=r["seed"], params=r["params"], t_train_s=r["t_train_s"],
                         t_infer_s=float(t.t_infer_per_shape_s.mean())))
    D = pd.concat(rows)
    per = D.groupby(["model", "N", "seed", "split"]).rel.mean().unstack("split")
    agg = per.groupby(["model", "N"]).agg(["mean", "std"])
    L = ["# 관문 실험: 좁게 배우고 넓게 시험", "", CRIT, ""]
    L.append("## 평균 상대 L2 오차 (시드 2개 평균 ± 표준편차; 형상 × 초기조건 × 시간 평균)\n")
    L.append("| 모델 | N | 학습 형상군 (test_in) | 합성 OOD | 실제 형상 | OOD/학습 | 실제/학습 |")
    L.append("|---|---|---|---|---|---|---|")
    ratio = {}
    for (model, N), row in agg.iterrows():
        i, o, r = row[("test_in", "mean")], row[("test_ood", "mean")], row[("test_real", "mean")]
        ratio[(model, N)] = (i, o, r)
        L.append(f"| {model} | {N} | {i:.4f} ± {row[('test_in', 'std')]:.4f} | {o:.4f} ± {row[('test_ood', 'std')]:.4f} | "
                 f"{r:.4f} ± {row[('test_real', 'std')]:.4f} | {o / i:.2f}× | {r / i:.2f}× |")
    L.append("\n## 판정\n")
    L.append("| N | 대상 | 우리 증가 (≤ 2) | PDON 증가 (≥ 5) | PDON 오차 / 우리 오차 (≥ 3) | 결과 |")
    L.append("|---|---|---|---|---|---|")
    verdicts = []
    for N in sorted({k[1] for k in ratio}):
        if ("ours", N) not in ratio or ("pdon", N) not in ratio:
            continue
        oi, oo, orr = ratio[("ours", N)]
        pi, po, pr = ratio[("pdon", N)]
        for lab, ours_x, pd_x, ours_e, pd_e in (("합성 OOD", oo / oi, po / pi, oo, po), ("실제 형상", orr / oi, pr / pi, orr, pr)):
            c = [ours_x <= 2, pd_x >= 5, pd_e / ours_e >= 3]
            v = "충분" if all(c) else f"부분 ({sum(c)}/3)"
            verdicts.append(all(c))
            L.append(f"| {N} | {lab} | {ours_x:.2f}× {'✓' if c[0] else '✗'} | {pd_x:.2f}× {'✓' if c[1] else '✗'} | "
                     f"{pd_e / ours_e:.1f}× {'✓' if c[2] else '✗'} | {v} |")
    L.append("")
    # reference rows (not part of the pre-registered comparison)
    L.append("## 참고: 학습 없는 Galerkin과 시험 시 다듬은 기저 (사전 설계 비교에는 포함하지 않음)\n")
    L.append("| 모델 | N | 학습 형상군 | 합성 OOD | 실제 형상 | OOD/학습 | 실제/학습 |")
    L.append("|---|---|---|---|---|---|---|")
    gp = os.path.join(ROOT, "gate", "galerkin_G.csv")
    if os.path.exists(gp):
        g = pd.read_csv(gp).pivot_table(index="basis", columns="split", values="rel")
        for b, lab in (("raw", "G (학습 없음), 원시 NEO"), ("refined", "G (학습 없음), 다듬은 NEO")):
            i, o, r = g.loc[b, "test_in"], g.loc[b, "test_ood"], g.loc[b, "test_real"]
            L.append(f"| {lab} | 0 | {i:.4f} | {o:.4f} | {r:.4f} | {o / i:.2f}× | {r / i:.2f}× |")
    ref = []
    for d in sorted(glob.glob(os.path.join(RUNS, "ours_N*_s*"))):
        fp = os.path.join(d, "test_refined.csv")
        if os.path.exists(fp):
            t = pd.read_csv(fp); r_ = json.load(open(os.path.join(d, "result.json")))
            t["N"], t["seed"] = r_["N"], r_["seed"]; ref.append(t)
    if ref:
        R2 = pd.concat(ref).groupby(["N", "seed", "split"]).rel.mean().unstack("split").groupby("N").mean()
        for N, row in R2.iterrows():
            L.append(f"| 우리 학습 모델, 시험 때만 다듬은 NEO (재학습 없음) | {N} | {row.test_in:.4f} | {row.test_ood:.4f} | "
                     f"{row.test_real:.4f} | {row.test_ood / row.test_in:.2f}× | {row.test_real / row.test_in:.2f}× |")
    L.append("")
    L.append("## 형상군별 (N = 200, 시드 평균)\n")
    D["family"] = D["shape"].str.replace(r"_\d+$", "", regex=True)
    fam = D[D.N == 200].groupby(["split", "family", "model"]).rel.mean().unstack("model")
    L.append(fam.to_markdown(floatfmt=".4f"))
    L.append("\n## 비용\n")
    R = pd.DataFrame(runs).groupby(["model", "N"]).agg(params=("params", "first"), t_train_s=("t_train_s", "mean"),
                                                        t_infer_s=("t_infer_s", "mean"))
    L.append(R.to_markdown(floatfmt=".3g"))
    txt = "\n".join(L) + "\n"
    open(os.path.join(ROOT, "gate", "REPORT_gate.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
