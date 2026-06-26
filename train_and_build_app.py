"""
train_and_build_app.py
======================
One command to (re)train the models and rebuild the arcade predictor.

Steps:
  1. Train LightGBM on the full international history (Tier-A features).
  2. Fit the Dixon-Coles model on a recent window (current team strength).
  3. Report Dixon-Coles RPS and the LightGBM+DC blend RPS (so you see the gain).
  4. Precompute, for all 48 World Cup nations, the blended W/D/L + the DC scoreline/xG.
  5. Inject that data into retro_template.html  ->  retro_predictor.html (ready to deploy).

Run in Colab after uploading: build_training_set.py, model_compare.py, dixon_coles.py,
retro_template.html  (and results.csv is downloaded automatically if missing).

    python train_and_build_app.py
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pandas as pd

from build_training_set import (compute_features, to_Xy, export_team_states,
                                 featurize_fixture, FEATURE_COLS)
from model_compare import train_and_save, load_predictor, rps, LGBModel, time_series_folds
from dixon_coles import DixonColes

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
DC_WINDOW_START = "2018-01-01"      # recency window for Dixon-Coles strength
DC_XI = 0.0018                       # time-decay; ~0.5 weight at ~1 year

WC_TEAMS = ["Argentina","Spain","France","England","Brazil","Portugal","Netherlands","Germany",
"Colombia","Morocco","Norway","Japan","Mexico","Switzerland","Ecuador","United States","Croatia",
"Belgium","Italy","Denmark","Uruguay","Australia","Austria","Canada","Paraguay","Iran","South Korea",
"Algeria","Nigeria","Senegal","Egypt","Ivory Coast","Ukraine","Serbia","Chile","Sweden","Panama",
"Poland","Peru","Wales","Costa Rica","Cameroon","Saudi Arabia","New Zealand","Ghana","South Africa",
"Tunisia","Qatar","Turkey"]


def ensure_results(path="results.csv"):
    if not os.path.exists(path):
        print("downloading results.csv ...")
        urllib.request.urlretrieve(RESULTS_URL, path)
    return path


def evaluate_blend(M: pd.DataFrame, n_folds=5):
    """Forward-chaining RPS: LightGBM, Dixon-Coles, and the 50/50 blend on identical folds."""
    pL_all, pD_all, y_all = [], [], []
    for tr, va in time_series_folds(M["date"], n_folds):
        mL = LGBModel().fit(M.loc[M.index[tr], FEATURE_COLS].values, M["y"].to_numpy()[tr])
        pL = mL.model_.predict(M.loc[M.index[va], FEATURE_COLS].values)
        mD = DixonColes(xi=DC_XI).fit(M.iloc[tr], ref_date=M.iloc[tr]["date"].max())
        pD = mD.predict_proba_fixtures(M.iloc[va][["home_team", "away_team", "neutral"]])
        pL_all.append(pL); pD_all.append(pD); y_all.append(M["y"].to_numpy()[va])
    pL, pD, y = np.vstack(pL_all), np.vstack(pD_all), np.concatenate(y_all)
    blend = 0.5 * pL + 0.5 * pD
    return {"LightGBM": rps(pL, y), "Dixon-Coles": rps(pD, y), "blend(0.5)": rps(blend, y)}


def build_lookup(lgb_predict, dc: DixonColes, states: dict, teams: list, ref_date="2026-06-15"):
    today = pd.Timestamp(ref_date)
    idx = {t: i for i, t in enumerate(teams)}

    def one(a, b, neutral):
        row = featurize_fixture(a, b, today, neutral=neutral, states=states)
        pL = np.array(lgb_predict(pd.DataFrame([row])[FEATURE_COLS])[0])
        d = dc.predict_one(a, b, neutral=neutral)
        pD = np.array([d["p_home"], d["p_draw"], d["p_away"]])
        bl = 0.5 * pL + 0.5 * pD; bl = bl / bl.sum()
        return [round(float(bl[0]), 3), round(float(bl[1]), 3), round(float(bl[2]), 3),
                d["score"][0], d["score"][1], round(d["xg_home"], 2), round(d["xg_away"], 2)]

    lk = {}
    for a in teams:
        for b in teams:
            if a == b:
                continue
            lk[f"{idx[a]}.{idx[b]}.N"] = one(a, b, True)
            lk[f"{idx[a]}.{idx[b]}.H"] = one(a, b, False)
    return lk


def main():
    ensure_results()
    feats = compute_features("results.csv", start_date="2004-01-01")
    X, y, d = to_Xy(feats)

    # 1) LightGBM
    print("training LightGBM ...")
    train_and_save(X, y, d, out_prefix="wc_model", which="lightgbm")
    lgb = load_predictor(prefix="wc_model", which="lightgbm")

    # 2) Dixon-Coles
    print(f"fitting Dixon-Coles on {DC_WINDOW_START}+ ...")
    raw = pd.read_csv("results.csv", parse_dates=["date"])
    raw["neutral"] = raw["neutral"].astype(str).str.upper().eq("TRUE")
    dc = DixonColes(xi=DC_XI).fit(raw[raw["date"] >= DC_WINDOW_START])
    print(f"  home_adv={dc.params_['home_adv']:.3f}  rho={dc.params_['rho']:.3f}")

    # 3) evaluate (build aligned frame with scores for DC)
    M = feats.merge(raw[["date", "home_team", "away_team", "home_score", "away_score"]],
                    on=["date", "home_team", "away_team"], how="left") \
             .dropna(subset=["home_score", "away_score"]).sort_values("date").reset_index(drop=True)
    print("evaluating (forward-chaining RPS) ...")
    for k, v in evaluate_blend(M).items():
        print(f"  {k:14s} {v:.4f}")

    # 4) build lookup for the 48 nations
    states = export_team_states("results.csv")
    teams = sorted([t for t in WC_TEAMS if t in states])
    print(f"building lookup for {len(teams)} nations ...")
    lk = build_lookup(lgb, dc, states, teams)
    data = {"teams": teams, "elo": [round(states[t]["elo"]) for t in teams], "lk": lk}

    # 5) inject into template
    with open("retro_template.html") as f:
        tpl = f.read()
    html = tpl.replace("__EMBED_DATA__", json.dumps(data, separators=(",", ":")))
    with open("retro_predictor.html", "w") as f:
        f.write(html)
    print(f"wrote retro_predictor.html ({round(len(html)/1024)} KB) — ready to deploy.")


if __name__ == "__main__":
    main()
