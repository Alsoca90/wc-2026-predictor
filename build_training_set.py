"""
build_training_set.py
=====================
Turn the raw international-results history into a train-ready table for model_compare:
X (A-B difference features), y (0=home win, 1=draw, 2=away win), and match dates.

Design choice: TWO feature tiers.
  Tier A (this file): cheap, computable for the ENTIRE 150-year history -> Elo,
    Elo momentum, rolling form, rest, neutrality, match importance. This trains the
    base model on tens of thousands of matches.
  Tier B (wc_squad_features + understat + chemistry + coach): expensive, realistically
    only assembled for the UPCOMING tournament's fixtures. You CANNOT backfill squad
    club-form / chemistry for 49k historical matches (you'd need every squad's club
    stats at every past date). So Tier B is an inference-time augmentation for 2026,
    merged onto the Tier-A row via build_match_row(); validate it only on recent
    tournaments where you actually assembled it.

Elo is computed here from scratch (World Football Elo conventions) so the model has no
dependency on eloratings.net / Kaggle mirrors that can move or go stale.

Input: results.csv from github.com/martj42/international_results
Output: features DataFrame + (X, y, dates) ready for model_compare.evaluate_cv().
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

# Match-importance K weights (World Football Elo style). Higher K = result moves Elo more.
def tournament_k(tournament: str) -> float:
    t = (tournament or "").lower()
    if "world cup" in t and "qualif" not in t:
        return 60.0
    if any(x in t for x in ["euro", "copa am", "african cup", "asian cup", "confederations"]) \
            and "qualif" not in t:
        return 50.0
    if "qualif" in t or "nations league" in t:
        return 40.0
    if "friendly" in t:
        return 20.0
    return 30.0


HOME_ADV_ELO = 65.0   # home-field advantage in Elo points (0 applied when neutral)
FORM_WINDOW = 10      # matches in the rolling-form window
START_ELO = 1500.0


def _goal_diff_multiplier(margin: int) -> float:
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8.0


def compute_features(results_csv: str = "results.csv",
                     start_date: str = "2004-01-01") -> pd.DataFrame:
    """Walk matches chronologically, snapshot PRE-match features, then update Elo.

    start_date filters the RETURNED rows (training window) but Elo is warmed up on the
    full history before that, so early-window ratings are already meaningful.
    """
    df = pd.read_csv(results_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")

    elo: dict[str, float] = defaultdict(lambda: START_ELO)
    last_date: dict[str, pd.Timestamp] = {}
    elo_1y_ago: dict[str, deque] = defaultdict(lambda: deque())   # (date, elo) snapshots
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_WINDOW))  # (points, gd)

    rows = []
    start = pd.Timestamp(start_date)

    for r in df.itertuples(index=False):
        h, a = r.home_team, r.away_team
        rh, ra = elo[h], elo[a]
        hfa = 0.0 if r.neutral else HOME_ADV_ELO

        # --- snapshot PRE-match features (no leakage) ---
        # Elo momentum: own Elo now minus own Elo ~365d ago
        def momentum(team):
            dq = elo_1y_ago[team]
            cutoff = r.date - pd.Timedelta(days=365)
            past = START_ELO
            for (d, e) in dq:
                if d <= cutoff:
                    past = e
                else:
                    break
            return elo[team] - past

        mom_h, mom_a = momentum(h), momentum(a)

        def form_stats(team):
            dq = form[team]
            if not dq:
                return 0.0, 0.0
            pts = np.mean([p for p, _ in dq])
            gd = np.mean([g for _, g in dq])
            return pts, gd

        fp_h, fg_h = form_stats(h)
        fp_a, fg_a = form_stats(a)

        rest_h = (r.date - last_date[h]).days if h in last_date else 180
        rest_a = (r.date - last_date[a]).days if a in last_date else 180
        rest_h, rest_a = min(rest_h, 365), min(rest_a, 365)

        if r.date >= start and pd.notna(r.home_score) and pd.notna(r.away_score):
            rows.append({
                "date": r.date, "home_team": h, "away_team": a,
                "tournament": r.tournament, "neutral": int(r.neutral),
                # difference features (home - away); home advantage encoded separately
                "diff_elo": (rh + hfa) - ra,
                "diff_elo_momentum": mom_h - mom_a,
                "diff_form_points": fp_h - fp_a,
                "diff_form_gd": fg_h - fg_a,
                "diff_rest_days": rest_h - rest_a,
                "home_advantage": hfa,                    # 0 on neutral sites
                # label: 0 home win, 1 draw, 2 away win
                "y": 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2),
            })

        # --- update Elo AFTER snapshotting ---
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            margin = abs(int(r.home_score) - int(r.away_score))
            s_home = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
            e_home = 1.0 / (1.0 + 10 ** (((ra) - (rh + hfa)) / 400.0))
            k = tournament_k(r.tournament) * _goal_diff_multiplier(margin)
            delta = k * (s_home - e_home)
            elo[h] = rh + delta
            elo[a] = ra - delta

            for team in (h, a):
                elo_1y_ago[team].append((r.date, elo[team]))
                # trim snapshots older than ~2y to bound memory
                while elo_1y_ago[team] and elo_1y_ago[team][0][0] < r.date - pd.Timedelta(days=730):
                    elo_1y_ago[team].popleft()

            hp = 3 if s_home == 1 else (1 if s_home == 0.5 else 0)
            form[h].append((hp, int(r.home_score) - int(r.away_score)))
            form[a].append((3 - hp if hp != 1 else 1, int(r.away_score) - int(r.home_score)))
            last_date[h] = last_date[a] = r.date

    return pd.DataFrame(rows)


FEATURE_COLS = ["diff_elo", "diff_elo_momentum", "diff_form_points",
                "diff_form_gd", "diff_rest_days", "home_advantage"]


def to_Xy(feats: pd.DataFrame):
    """Split the feature table into (X, y, dates) for model_compare.evaluate_cv()."""
    X = feats[FEATURE_COLS].copy()
    y = feats["y"].to_numpy()
    dates = feats["date"]
    return X, y, dates


if __name__ == "__main__":
    feats = compute_features("results.csv", start_date="2004-01-01")
    print(f"rows: {len(feats):,}  span: {feats.date.min().date()} -> {feats.date.max().date()}")
    print("class balance (0=H,1=D,2=A):", np.bincount(feats.y) / len(feats))
    print(feats[FEATURE_COLS].describe().round(2).T[["mean", "std", "min", "max"]])


# ---------------------------------------------------------------------------
# Export CURRENT team state, so upcoming 2026 fixtures can be featurized for Tier-A
# ---------------------------------------------------------------------------

def export_team_states(results_csv: str = "results.csv") -> dict:
    """Walk the full history and return each team's CURRENT state for fixture featurizing.

    Returns {team: {"elo", "elo_history":[(date,elo)...], "form_points", "form_gd",
                    "last_date"}}. elo_history is trimmed to ~400 days so featurize_fixture
    can resolve Elo ~365d before any future date (for momentum).
    """
    df = pd.read_csv(results_csv, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")

    elo = defaultdict(lambda: START_ELO)
    last_date, form = {}, defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    hist = defaultdict(lambda: deque())

    for r in df.itertuples(index=False):
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            continue
        h, a = r.home_team, r.away_team
        rh, ra = elo[h], elo[a]
        hfa = 0.0 if r.neutral else HOME_ADV_ELO
        margin = abs(int(r.home_score) - int(r.away_score))
        s_home = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        e_home = 1.0 / (1.0 + 10 ** ((ra - (rh + hfa)) / 400.0))
        delta = tournament_k(r.tournament) * _goal_diff_multiplier(margin) * (s_home - e_home)
        elo[h], elo[a] = rh + delta, ra - delta
        for t in (h, a):
            hist[t].append((r.date, elo[t]))
            while hist[t] and hist[t][0][0] < r.date - pd.Timedelta(days=400):
                hist[t].popleft()
        hp = 3 if s_home == 1 else (1 if s_home == 0.5 else 0)
        form[h].append((hp, int(r.home_score) - int(r.away_score)))
        form[a].append((3 - hp if hp != 1 else 1, int(r.away_score) - int(r.home_score)))
        last_date[h] = last_date[a] = r.date

    states = {}
    for t in elo:
        fp = float(np.mean([p for p, _ in form[t]])) if form[t] else 0.0
        fg = float(np.mean([g for _, g in form[t]])) if form[t] else 0.0
        states[t] = {"elo": elo[t], "elo_history": list(hist[t]),
                     "form_points": fp, "form_gd": fg, "last_date": last_date.get(t)}
    return states


def featurize_fixture(home: str, away: str, date, neutral: bool, states: dict) -> dict:
    """Build a Tier-A feature row (FEATURE_COLS) for an upcoming fixture from team states."""
    date = pd.Timestamp(date)
    sh, sa = states.get(home), states.get(away)
    if sh is None or sa is None:
        raise KeyError(f"missing team state for {home if sh is None else away}")
    hfa = 0.0 if neutral else HOME_ADV_ELO

    def momentum(s):
        cutoff = date - pd.Timedelta(days=365)
        past = START_ELO
        for (d, e) in s["elo_history"]:
            if d <= cutoff:
                past = e
            else:
                break
        return s["elo"] - past

    def rest(s):
        return min((date - s["last_date"]).days, 365) if s["last_date"] is not None else 180

    return {
        "diff_elo": (sh["elo"] + hfa) - sa["elo"],
        "diff_elo_momentum": momentum(sh) - momentum(sa),
        "diff_form_points": sh["form_points"] - sa["form_points"],
        "diff_form_gd": sh["form_gd"] - sa["form_gd"],
        "diff_rest_days": rest(sh) - rest(sa),
        "home_advantage": hfa,
    }
