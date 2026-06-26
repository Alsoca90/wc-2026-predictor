"""
dixon_coles.py
==============
The football-specific forecasting model: instead of predicting home/draw/away directly,
it models each team's expected GOALS as Poisson rates, then derives the scoreline grid
and collapses it to 1X2. This is the Dixon & Coles (1997) model.

Why it's the right tool here
----------------------------
  * Outputs a full scoreline distribution -> real predicted scores for the animation,
    not a sampled guess.
  * The low-score correction (tau) fixes independent Poisson's well-known mistake on
    0-0 / 1-0 / 0-1 / 1-1, which is exactly where tight tournament games live.
  * Time decay (xi) weights recent form, so 2010 results don't count like last month's.

Parameters estimated by MLE
---------------------------
  attack[team]   - scoring strength (higher = scores more)
  defense[team]  - conceding strength (higher = concedes more)
  home_adv       - global home-field goal boost (applied only at non-neutral venues)
  rho            - Dixon-Coles low-score dependence correction

Lambda (home) = exp(attack_home - defense_away + home_adv)
Mu    (away)  = exp(attack_away - defense_home)

Validation: a `predict_proba`-compatible wrapper lets it drop straight into
model_compare.evaluate_cv so you can RPS it against LightGBM on identical folds.

Dependencies: numpy, pandas, scipy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10   # scoreline grid 0..MAX_GOALS per side; >10 goals is negligible mass


def _tau(home_goals, away_goals, lam, mu, rho):
    """Dixon-Coles low-score correction. Adjusts only the four 0/1 x 0/1 cells."""
    hg, ag = home_goals, away_goals
    out = np.ones_like(lam, dtype=float)
    out = np.where((hg == 0) & (ag == 0), 1 - lam * mu * rho, out)
    out = np.where((hg == 0) & (ag == 1), 1 + lam * rho, out)
    out = np.where((hg == 1) & (ag == 0), 1 + mu * rho, out)
    out = np.where((hg == 1) & (ag == 1), 1 - rho, out)
    return out


@dataclass
class DixonColes:
    xi: float = 0.0018          # time-decay rate per day (~0.5 weight at ~1 year)
    max_iter: int = 100
    teams_: list = field(default=None)
    params_: dict = field(default=None)
    ref_date_: pd.Timestamp = None

    # ---- fit -------------------------------------------------------------
    def fit(self, df: pd.DataFrame, ref_date=None):
        """df columns: date, home_team, away_team, home_score, away_score, neutral(bool)."""
        d = df.dropna(subset=["home_score", "away_score"]).copy()
        d["home_score"] = d["home_score"].astype(int)
        d["away_score"] = d["away_score"].astype(int)
        self.ref_date_ = pd.Timestamp(ref_date) if ref_date is not None else d["date"].max()
        # time-decay weight per match
        age_days = (self.ref_date_ - pd.to_datetime(d["date"])).dt.days.clip(lower=0).to_numpy()
        w = np.exp(-self.xi * age_days)

        teams = sorted(set(d["home_team"]) | set(d["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = d["home_team"].map(idx).to_numpy()
        ai = d["away_team"].map(idx).to_numpy()
        hg = d["home_score"].to_numpy()
        ag = d["away_score"].to_numpy()
        neutral = d["neutral"].astype(bool).to_numpy() if "neutral" in d else np.zeros(len(d), bool)

        # param vector: [attack(n), defense(n), home_adv, rho]; attack mean fixed at 0 for identifiability
        def unpack(p):
            atk = p[:n]; dfn = p[n:2 * n]; ha = p[2 * n]; rho = p[2 * n + 1]
            atk = atk - atk.mean()                       # identifiability constraint
            return atk, dfn, ha, rho

        def negloglik(p):
            atk, dfn, ha, rho = unpack(p)
            home_boost = np.where(neutral, 0.0, ha)
            lam = np.exp(atk[hi] - dfn[ai] + home_boost)
            mu = np.exp(atk[ai] - dfn[hi])
            ll = (poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
                  + np.log(np.clip(_tau(hg, ag, lam, mu, rho), 1e-10, None)))
            return -np.sum(w * ll)

        p0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
        res = minimize(negloglik, p0, method="L-BFGS-B",
                       options={"maxiter": self.max_iter})
        atk, dfn, ha, rho = unpack(res.x)
        self.teams_ = teams
        self.params_ = {"attack": dict(zip(teams, atk)), "defense": dict(zip(teams, dfn)),
                        "home_adv": float(ha), "rho": float(rho)}
        return self

    # ---- core: scoreline grid for one fixture ---------------------------
    def score_matrix(self, home, away, neutral=True) -> np.ndarray:
        """(MAX_GOALS+1)x(MAX_GOALS+1) joint P(home_goals, away_goals)."""
        pr = self.params_
        # unseen team -> league-average (attack/defense 0)
        ah = pr["attack"].get(home, 0.0); dh = pr["defense"].get(home, 0.0)
        aa = pr["attack"].get(away, 0.0); da = pr["defense"].get(away, 0.0)
        ha = 0.0 if neutral else pr["home_adv"]
        lam = np.exp(ah - da + ha)
        mu = np.exp(aa - dh)

        g = np.arange(MAX_GOALS + 1)
        ph = poisson.pmf(g, lam)[:, None]
        pa = poisson.pmf(g, mu)[None, :]
        grid = ph * pa
        # apply low-score correction to the 2x2 corner
        H, A = np.meshgrid(g, g, indexing="ij")
        grid = grid * _tau(H, A, lam, mu, pr["rho"])
        return grid / grid.sum()

    def predict_one(self, home, away, neutral=True) -> dict:
        """1X2 probs + most-likely scoreline + expected goals for one fixture."""
        m = self.score_matrix(home, away, neutral)
        p_home = np.tril(m, -1).sum()           # home_goals > away_goals
        p_draw = np.trace(m)
        p_away = np.triu(m, 1).sum()
        i, j = np.unravel_index(m.argmax(), m.shape)
        g = np.arange(MAX_GOALS + 1)
        return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away,
                "score": (int(i), int(j)),
                "xg_home": float((m.sum(1) * g).sum()), "xg_away": float((m.sum(0) * g).sum())}

    # ---- sklearn-style proba for evaluate_cv ----------------------------
    def predict_proba_fixtures(self, fixtures: pd.DataFrame) -> np.ndarray:
        """fixtures: home_team, away_team, neutral -> (n,3) [home, draw, away]."""
        out = np.zeros((len(fixtures), 3))
        for k, r in enumerate(fixtures.itertuples(index=False)):
            d = self.predict_one(r.home_team, r.away_team, bool(getattr(r, "neutral", True)))
            out[k] = [d["p_home"], d["p_draw"], d["p_away"]]
        return out


# ---------------------------------------------------------------------------
# Adapter so DixonColes plugs into model_compare.evaluate_cv on identical folds
# ---------------------------------------------------------------------------

class DixonColesCV:
    """Wraps DixonColes with a fit/predict_proba interface keyed off raw match rows.

    evaluate_cv calls fit(X, y) then predict_proba(X) on numpy arrays, but DC needs the
    actual teams/dates/scores -- so we carry a parallel `matches` frame and index into it.
    Use evaluate_cv_dc() below instead of the generic evaluate_cv for this model.
    """
    def __init__(self, xi=0.0018):
        self.xi = xi
        self.model = None

    def fit(self, matches_train: pd.DataFrame):
        self.model = DixonColes(xi=self.xi).fit(matches_train, ref_date=matches_train["date"].max())
        return self

    def predict_proba(self, matches_eval: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba_fixtures(
            matches_eval[["home_team", "away_team", "neutral"]])


def evaluate_cv_dc(matches: pd.DataFrame, n_folds=5):
    """Forward-chaining RPS for Dixon-Coles on the raw match frame.

    matches: date, home_team, away_team, home_score, away_score, neutral, y(0/1/2)
    """
    from model_compare import rps, time_series_folds
    order_dates = matches["date"]
    rps_scores = []
    for tr, va in time_series_folds(order_dates, n_folds):
        m = DixonColesCV().fit(matches.iloc[tr])
        p = m.predict_proba(matches.iloc[va])
        rps_scores.append(rps(p, matches["y"].to_numpy()[va]))
    return float(np.mean(rps_scores)), rps_scores


if __name__ == "__main__":
    # Quick self-test on the real results file.
    df = pd.read_csv("results.csv", parse_dates=["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    recent = df[df["date"] >= "2018-01-01"].copy()

    dc = DixonColes(xi=0.0018).fit(recent)
    print("home_adv:", round(dc.params_["home_adv"], 3), " rho:", round(dc.params_["rho"], 3))
    top = sorted(dc.params_["attack"].items(), key=lambda t: -t[1])[:6]
    print("strongest attacks:", [(t, round(v, 2)) for t, v in top])
    for a, b in [("Spain", "Brazil"), ("Argentina", "United States")]:
        r = dc.predict_one(a, b, neutral=True)
        print(f"{a} v {b}: H {r['p_home']:.2f}/D {r['p_draw']:.2f}/A {r['p_away']:.2f} "
              f"| likely {r['score'][0]}-{r['score'][1]} | xg {r['xg_home']:.2f}-{r['xg_away']:.2f}")
