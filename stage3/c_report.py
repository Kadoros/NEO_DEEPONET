"""C final tables + shape gallery -> real_shapes/C_tables.md, real_shapes/gallery.png

    python stage3/c_report.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(ROOT, "real_shapes")
B4 = os.path.join(os.path.dirname(ROOT), "stage2", "coef", "b4")
BASES = ["neo256", "neoinv2", "arpack256"]
fmt = lambda v: f"{v:.3g}"


def b4_results():
    out = {}
    for b in BASES:
        p = os.path.join(B4, f"test_real_{b}_net-base_asm32", "results.csv")          # fp32 assembly (== fp64, checked)
        if not os.path.exists(p):
            p = os.path.join(B4, f"test_real_{b}_net-base", "results.csv")
        if os.path.exists(p):
            E = pd.read_csv(p)
            E["shape"] = E.name.str.rsplit("_", n=1).str[0]
            E["vals"] = [[v["value"] for v in json.loads(pl) if v["value"] == v["value"]] if isinstance(pl, str) else []
                         for pl in E.per_lump]
            out[b] = E
    return out


def main():
    L = []
    a = pd.read_csv(os.path.join(RS, "a_basis.csv"))
    r = pd.read_csv(os.path.join(RS, "a_refine.csv"))
    q = a[(a.kind == "quasi") & (a.N == 8192)].set_index("shape")
    ind = q[["korn", "thick"]]
    # (c) lumps
    R = b4_results()
    L.append("## (c) 숨은 덩어리 찾기 (B-4, 네트워크 재학습 없음)\n")
    L.append("| 기저 | 사례 | 덩어리 | 1단계 찾음 | 2단계 찾음 | 오탐 | 분할 | 상대 L2 (1단계 → 2단계) | 피팅 κ 평균 / 중앙값 / 범위 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for b, E in R.items():
        v = sum(E.vals.tolist(), [])
        L.append(f"| {b} | {len(E)} | {int(E.n_true.sum())} | {int(E.s1_n_matched.sum())} | {int(E.s2_n_matched.sum())} | "
                 f"{int(E.s2_n_false.sum())} | {int(E.n_split.sum())} | {E.s1_rel_l2.mean():.3f} → {E.s2_rel_l2.mean():.3f} | "
                 f"{np.mean(v):.2f} / {np.median(v):.2f} / {min(v):.2f}–{max(v):.2f} |")
    if R:
        per = pd.concat({b: E.groupby("shape").agg(found=("s2_n_matched", "sum"), true=("n_true", "sum"),
                                                     false=("s2_n_false", "sum"), rel=("s2_rel_l2", "mean"))
                         for b, E in R.items()}, axis=1)
        per.columns = [f"{b}|{c}" for b, c in per.columns]                 # flat columns (joinable)
        s1 = list(R.values())[0].groupby("shape").agg(s1found=("s1_n_matched", "sum"), s1rel=("s1_rel_l2", "mean"))
        per = per.join(s1).join(ind)
        L.append("\n형상별 (3가지 배치 합계; 1단계는 기저와 무관):\n")
        L.append("| 형상 | C_K | 두께 | 덩어리 | 1단계 찾음 | NEO 원시: 찾음/오탐/L2 | NEO+다듬기: 찾음/오탐/L2 | ARPACK: 찾음/오탐/L2 | 판정 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for sh, row in per.sort_values("korn").iterrows():
            cell = lambda b: f"{int(row[f'{b}|found'])}/{int(row[f'{b}|false'])}/{row[f'{b}|rel']:.3f}" if f"{b}|found" in row else "-"
            verdict = ""
            if all(f"{b}|rel" in row for b in BASES):
                raw, ref, ex = row["neo256|rel"], row["neoinv2|rel"], row["arpack256|rel"]
                raw_bad = row["neo256|found"] < row["arpack256|found"] or raw > 1.25 * ex or row["neo256|false"] > row["arpack256|false"]
                ref_ok = abs(ref - ex) <= 0.1 * ex + 0.005 and row["neoinv2|found"] >= row["arpack256|found"]
                verdict = ("원시 실패 → 다듬기로 회복" if ref_ok else "원시 실패, 다듬기로도 부족") if raw_bad else ("원시로 충분" if ref_ok or raw <= 1.1 * ex else "확인 필요")
            L.append(f"| {sh} | {row.korn:.0f} | {row.thick:.2f} | {int(row[f'{BASES[0]}|true'])} | {int(row.s1found)} | "
                     f"{cell('neo256')} | {cell('neoinv2')} | {cell('arpack256')} | {verdict} |")
    # fp32 check
    p32 = os.path.join(B4, "test_real_neoinv2_net-base_asm32_sel", "results.csv")
    p64 = os.path.join(B4, "test_real_neoinv2_net-base_sel", "results.csv")
    if os.path.exists(p32) and os.path.exists(p64):
        F = pd.read_csv(p32).set_index("name")
        D = pd.read_csv(p64).set_index("name")
        F = F.loc[F.index.intersection(D.index)]; D = D.loc[F.index]
        L.append("\n## fp32 조립 확인 (기저 neoinv2, 6개 사례)\n")
        L.append("| 사례 | 찾음 fp64 / fp32 | 오탐 fp64 / fp32 | 상대 L2 fp64 / fp32 | 피팅 κ fp64 | 피팅 κ fp32 | 시간 fp64 / fp32 (s) |")
        L.append("|---|---|---|---|---|---|---|")
        for n in F.index:
            L.append(f"| {n} | {int(D.loc[n, 's2_n_matched'])} / {int(F.loc[n, 's2_n_matched'])} | {int(D.loc[n, 's2_n_false'])} / {int(F.loc[n, 's2_n_false'])} | "
                     f"{D.loc[n, 's2_rel_l2']:.4f} / {F.loc[n, 's2_rel_l2']:.4f} | {D.loc[n, 'values']} | {F.loc[n, 'values']} | "
                     f"{D.loc[n, 't_s']:.0f} / {F.loc[n, 't_s']:.0f} |")
    # (b) heat
    hp = os.path.join(RS, "b_heat.csv")
    if os.path.exists(hp):
        H = pd.read_csv(hp)
        L.append("\n## (b) 열확산 (정답: 세밀 FEM, 상대 L2, τ 6개 평균)\n")
        t = H.pivot_table(index=["shape"], columns=["kind", "model"], values="relL2", aggfunc="mean")
        t.columns = [f"{k} {mdl}" for k, mdl in t.columns]
        L.append(t.join(ind).sort_values("korn").to_markdown(floatfmt=".3g"))
        L.append("\n전체 평균:\n")
        L.append(H.pivot_table(index="kind", columns="model", values="relL2", aggfunc="mean").to_markdown(floatfmt=".3g"))
        L.append("\nτ별 평균 (quasi):\n")
        L.append(H[H.kind == "quasi"].pivot_table(index="model", columns="tau", values="relL2", aggfunc="mean").to_markdown(floatfmt=".3g"))
    # timing
    tp = os.path.join(RS, "timing.csv")
    if os.path.exists(tp):
        T = pd.read_csv(tp).sort_values(["N", "n_tets"])
        cols = ["shape", "N", "n_tets", "neo_gpu", "neo_rr", "scal_lu", "scal_step", "el_assemble", "el_orth_rr", "el_lu", "el_step",
                "static_corr", "red_cpu", "red_gpu", "full_amg_el", "full_amg_heat", "basis_elastic_c", "breakeven_red_cpu", "breakeven_red_gpu"]
        L.append("\n## 시간 (초, 단독 실행)\n")
        L.append(T[cols].to_markdown(index=False, floatfmt=".3g"))
        L.append("\nN별 중앙값:\n")
        L.append(T.groupby("N")[cols[3:]].median().to_markdown(floatfmt=".3g"))
    open(os.path.join(RS, "C_tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    gallery(q, r, R)


def gallery(q, r, R):
    import glob
    shapes = list(q.sort_values("korn").index)
    d = pd.read_csv(os.path.join(ROOT, "e1", "e1_v2_real.csv"))
    ratio = d[d.basis == "neo_rot_inv2_sc"].groupby("family").ratio_int.max()
    H = pd.read_csv(os.path.join(RS, "b_heat.csv")) if os.path.exists(os.path.join(RS, "b_heat.csv")) else None
    ref = r[(r.kind == "quasi") & (r.N == 8192) & (r.steps == 2)].set_index("shape")
    n = len(shapes)
    cols = 5
    fig = plt.figure(figsize=(4 * cols, 4.2 * ((n + cols - 1) // cols)))
    for i, sh in enumerate(shapes):
        z = np.load(os.path.join(RS, "cache", f"{sh}_quasi_8192.npz"))
        Y = z["Y"]
        ax = fig.add_subplot((n + cols - 1) // cols, cols, i + 1, projection="3d")
        sel = np.random.default_rng(0).choice(len(Y), min(3000, len(Y)), replace=False)
        ax.scatter(Y[sel, 0], Y[sel, 1], Y[sel, 2], s=1.2, c=Y[sel, 2], cmap="Blues", alpha=0.8)
        ax.set_axis_off(); ax.set_box_aspect(np.ptp(Y, 0))
        lines = [f"{sh}  C_K {q.loc[sh, 'korn']:.0f}, thick {q.loc[sh, 'thick']:.2f}",
                 f"NEO span96 {q.loc[sh, 'span96']:.1e} → refined {ref.loc[sh, 'span96']:.1e}",
                 f"elastic (c) worst ratio {ratio.get(sh, np.nan):.3f}"]
        if H is not None:
            h = H[(H.shape == sh) & (H.kind == "quasi")].groupby("model").relL2.mean()
            lines.append(f"heat G raw {h.get('G-neo256', np.nan):.3f} / refined {h.get('G-neoinv2', np.nan):.3f}")
        if "neoinv2" in R:
            E = R["neoinv2"]; e = E[E["shape"] == sh]
            if len(e):
                lines.append(f"lumps found {int(e.s2_n_matched.sum())}/{int(e.n_true.sum())}, false {int(e.s2_n_false.sum())}")
        ax.set_title("\n".join(lines), fontsize=7.5, color="#0b0b0b", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RS, "gallery.png"), dpi=100, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
