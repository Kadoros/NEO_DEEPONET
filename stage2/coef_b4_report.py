"""Final B-4 table + verdict "does a 256-dim basis reduce the contrast error?" -> appended to REPORT_main_and_coef.md

Reads coef/b4/<split>_<basis>_net-base/{summary.json,results.csv} for basis in arpack128 / arpack256 / neo256 and
split in test / test_ood.  Contrast error measures (true lump value 5):
  value error   |k_fit - 5| / 5 over matched lumps (fitted parameter)
  in-lump error |mean kappa_hat inside true lumps - 5| / 5 per shape (field, mass-weighted)
Verdict: 256 "reduces the contrast error" for a basis if, on the SAME shapes as arpack128, both mean errors drop
on both splits; paired per-shape win rates are listed too.

    python stage2/coef_b4_report.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
B4 = os.path.join(ROOT, "coef", "b4")
REPORT = os.path.join(ROOT, "REPORT_main_and_coef.md")
BASES = ["arpack128", "arpack256", "neo256"]
SPLITS = {"test": "같은 형상군 (20)", "test_ood": "처음 보는 형상군 torus·cup (17)"}


def load(split, basis):
    d = os.path.join(B4, f"{split}_{basis}_net-base")
    if not os.path.exists(os.path.join(d, "summary.json")):
        return None, None
    E = pd.read_csv(os.path.join(d, "results.csv"))
    E["inlump_err"] = (E.s2_in_lump - 5).abs() / 5
    E["value_err"] = [np.mean([abs(p["value"] - 5) / 5 for p in json.loads(pl) if p["value"] == p["value"]] or [np.nan])
                      if isinstance(pl, str) else np.nan for pl in E.per_lump]
    E["vals"] = [[p["value"] for p in json.loads(pl) if p["value"] == p["value"]] if isinstance(pl, str) else []
                 for pl in E.per_lump]
    return json.load(open(os.path.join(d, "summary.json"))), E.set_index("name")


def main():
    L = ["", "## B-4 최종: 2단계 방식의 기저 비교 (ARPACK 128 / ARPACK 256 / NEO 256)", "",
         "1단계는 B-2 네트워크(`base`, ARPACK 128로 학습)이고, 기저를 바꾼 것은 2단계 파라미터 피팅뿐입니다. 정답 κ = 5.", "",
         "| 기저 | 테스트 | 찾은 덩어리 | 오탐 | 분할 채택 | 상대 L2 | 덩어리 내부 평균 κ̂ | 피팅 κ 평균 / 중앙값 | 피팅 κ 범위 | 값 오차 \\|κ−5\\|/5 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    data = {}
    for split, lab in SPLITS.items():
        for b in BASES:
            s, E = load(split, b)
            data[(split, b)] = E
            if s is None:
                L.append(f"| {b} | {lab} | (결과 없음) | | | | | | | |")
                continue
            allv = sum(E.vals.tolist(), [])
            L.append(f"| {b} | {lab} | {s['s2_matched']}/{s['lumps']} (1단계 {s['s1_matched']}) | {s['s2_false']} | {s['splits']} | "
                     f"{s['s2_rel_l2']:.3f} | {s['s2_in_lump']:.2f} | {s['matched_value_mean']:.2f} / {s['matched_value_median']:.2f} | "
                     f"{min(allv):.2f}–{max(allv):.2f} | {np.nanmean(E.value_err):.3f} |")
    L += ["", "### 같은 형상끼리 쌍 비교 (ARPACK 128 대비)", "",
          "| 기저 | 테스트 | 형상 수 | 내부 평균 오차 128→기저 | 값 오차 128→기저 | 상대 L2 128→기저 | 상대 L2 개선 형상 비율 |",
          "|---|---|---|---|---|---|---|"]
    verdict = {}
    for b in ("arpack256", "neo256"):
        ok = True
        for split, lab in SPLITS.items():
            A, Bb = data[(split, "arpack128")], data[(split, b)]
            if A is None or Bb is None:
                L.append(f"| {b} | {lab} | (결과 없음) | | | | |"); ok = False; continue
            idx = A.index.intersection(Bb.index)
            a, bb = A.loc[idx], Bb.loc[idx]
            ia, ib = a.inlump_err.mean(), bb.inlump_err.mean()
            va, vb = np.nanmean(a.value_err), np.nanmean(bb.value_err)
            win = float((bb.s2_rel_l2 < a.s2_rel_l2).mean())
            L.append(f"| {b} | {lab} | {len(idx)} | {ia:.3f} → {ib:.3f} | {va:.3f} → {vb:.3f} | "
                     f"{a.s2_rel_l2.mean():.3f} → {bb.s2_rel_l2.mean():.3f} | {win:.0%} |")
            ok &= (ib < ia) and (vb < va)
        verdict[b] = ok
    L += ["", "### 판정: 256차원 기저가 세기 오차를 줄이는가", ""]
    for b, ok in verdict.items():
        L.append(f"- **{b}: {'예' if ok else '아니오 / 부분적'}**. 두 테스트 모두에서 같은 형상 기준으로 내부 평균 오차와 값 오차가 "
                 f"{'모두 줄었습니다' if ok else '모두 줄지는 않았습니다 (위 표 참고)'}.")
    L += ["- 판정 기준: 같은 형상에 대해 ARPACK 128보다 '덩어리 내부 평균 오차'와 '피팅 값 오차'의 평균이 **두 테스트 모두**에서 작으면 '예'입니다.",
          "- 과대추정 확인: 피팅 κ 범위의 상한이 5를 크게 넘으면, 오차가 줄더라도 과대추정이 섞인 것입니다 (분할된 덩어리에서 자주 나타남).",
          f"- 자동 생성: `coef_b4_report.py`, {pd.Timestamp.now():%Y-%m-%d %H:%M}.", ""]
    open(REPORT, "a").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
