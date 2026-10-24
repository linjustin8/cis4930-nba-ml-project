"""Prototype pre-game features plus tests that they never look at the game being predicted.

Every feature for a game is computed only from games played on EARLIER dates
(and, for standings, from the ranking.csv snapshot of an EARLIER date). This is a
feasibility prototype, not the final modeling pipeline.

Feature families
----------------
* rolling averages of each team's previous N games  (``rolling_form``)
* rest days / back-to-backs                          (``rest_days``)
* win/loss streak entering the game                  (``win_streak``)
* season record so far, from games.csv              (``season_record``)
* pre-game standings from ranking.csv (day before)   (``pregame_standings``)
* Elo ratings before tip-off                         (``elo_ratings``)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loading import SAME_GAME_STATS, to_team_games

ROLLING_STATS = ["WIN", "PTS", "PTS_OPP", "PLUS_MINUS", *[s for s in SAME_GAME_STATS if s != "PTS"]]


# --------------------------------------------------------------------------- team-level features
def rolling_form(tg: pd.DataFrame, windows=(5, 10), stats=ROLLING_STATS, within_season=True) -> pd.DataFrame:
    """Mean of each stat over the team's previous N games.

    ``shift(1)`` drops the current game before the window is applied, so game t
    only sees games t-1 ... t-N. With ``within_season`` the window restarts every
    season (first game of a season -> NaN).
    """
    keys = ["TEAM_ID", "SEASON"] if within_season else ["TEAM_ID"]
    gid = tg.groupby(keys, sort=False).ngroup()
    prev = tg.groupby(gid)[stats].shift(1)            # <- the anti-leakage step
    out = {}
    for n in windows:
        rolled = prev.groupby(gid).rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)
        rolled = rolled.sort_index()
        for s in stats:
            out[f"{s}_LAST{n}"] = rolled[s]
    return pd.DataFrame(out, index=tg.index)


def rest_days(tg: pd.DataFrame, cap: int = 7) -> pd.DataFrame:
    """Days since the team's previous game (any season), plus back-to-back and season-opener flags.

    Gaps across the offseason (or the 2020 COVID pause) are huge, so the model-ready
    ``REST_DAYS`` is capped at ``cap``; ``REST_DAYS_RAW`` keeps the original value.
    """
    prev_date = tg.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    raw = (tg["GAME_DATE"] - prev_date).dt.days
    opener = tg.groupby(["TEAM_ID", "SEASON"]).cumcount() == 0
    return pd.DataFrame({
        "REST_DAYS_RAW": raw,
        "REST_DAYS": raw.clip(upper=cap).fillna(cap),
        "BACK_TO_BACK": (raw == 1).astype(int),
        "SEASON_OPENER": opener.astype(int),
    }, index=tg.index)


def win_streak(tg: pd.DataFrame) -> pd.Series:
    """Signed streak entering the game: +3 = won last 3, -2 = lost last 2, 0 = first game of season."""
    streak = np.zeros(len(tg), dtype=int)
    team, season, win = tg["TEAM_ID"].to_numpy(), tg["SEASON"].to_numpy(), tg["WIN"].to_numpy()
    current = 0
    for i in range(len(tg)):
        if i == 0 or team[i] != team[i - 1] or season[i] != season[i - 1]:
            current = 0
        streak[i] = current                          # value BEFORE this game's result is known
        if win[i] == 1:
            current = current + 1 if current > 0 else 1
        else:
            current = current - 1 if current < 0 else -1
    return pd.Series(streak, index=tg.index, name="STREAK")


def season_record(tg: pd.DataFrame) -> pd.DataFrame:
    """Games played and win % so far this season, counted from games.csv (current game excluded)."""
    grp = tg.groupby(["TEAM_ID", "SEASON"])["WIN"]
    games_before = grp.cumcount()
    wins_before = grp.cumsum() - tg["WIN"]
    return pd.DataFrame({
        "SEASON_G": games_before,
        "SEASON_W_PCT": (wins_before / games_before.replace(0, np.nan)),
    }, index=tg.index)


def _win_pct(record: pd.Series) -> pd.Series:
    wl = record.str.split("-", expand=True).astype(float)
    return wl[0] / (wl[0] + wl[1]).replace(0, np.nan)


def pregame_standings(tg: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest regular-season standings row dated STRICTLY BEFORE the game day.

    ranking.csv snapshots already include the games played on STANDINGSDATE, so the
    same-day row would leak the result. ``allow_exact_matches=False`` enforces
    "day before or earlier". The row must belong to the same season; season openers
    therefore get 0-0 (W_PCT NaN).
    """
    r = ranking[["TEAM_ID", "SEASON", "STANDINGSDATE", "G", "W", "L", "W_PCT", "HOME_RECORD", "ROAD_RECORD"]].copy()
    r["STANDINGSDATE"] = r["STANDINGSDATE"].astype("datetime64[ns]")
    r = r.sort_values("STANDINGSDATE")
    left = tg[["TEAM_ID", "SEASON", "GAME_DATE"]].copy()
    left["GAME_DATE"] = left["GAME_DATE"].astype("datetime64[ns]")
    left["_row"] = np.arange(len(left))
    m = pd.merge_asof(left.sort_values("GAME_DATE"), r, left_on="GAME_DATE", right_on="STANDINGSDATE",
                      by=["TEAM_ID", "SEASON"], allow_exact_matches=False, direction="backward")
    m = m.sort_values("_row")
    out = pd.DataFrame({
        "ST_DATE_USED": m["STANDINGSDATE"].to_numpy(),
        "ST_G": m["G"].fillna(0).to_numpy(),
        "ST_W": m["W"].fillna(0).to_numpy(),
        "ST_L": m["L"].fillna(0).to_numpy(),
        "ST_W_PCT": m["W_PCT"].where(m["G"] > 0).to_numpy(),
        "ST_HOME_W_PCT": _win_pct(m["HOME_RECORD"].fillna("0-0")).to_numpy(),
        "ST_ROAD_W_PCT": _win_pct(m["ROAD_RECORD"].fillna("0-0")).to_numpy(),
    }, index=tg.index)
    return out


# --------------------------------------------------------------------------- game-level features
def elo_ratings(games: pd.DataFrame, k: float = 20.0, home_adv: float = 100.0,
                carryover: float = 0.75, base: float = 1500.0) -> pd.DataFrame:
    """Pre-game Elo for both teams, updated game by game in date order.

    The rating is read BEFORE the update with this game's result. Between seasons
    each rating is pulled back toward ``base`` (keep ``carryover`` of the gap).
    Neutral-site (2020 bubble) games get no home advantage.
    """
    g = games.sort_values(["GAME_DATE", "GAME_ID"])
    rating, last_season = {}, {}
    pre_h, pre_a, prob = [], [], []
    for h, a, season, win, neutral in zip(g["HOME_TEAM_ID"], g["VISITOR_TEAM_ID"], g["SEASON"],
                                          g["HOME_TEAM_WINS"], g["NEUTRAL_SITE"]):
        for t in (h, a):
            if t not in rating:
                rating[t] = base
            elif last_season[t] != season:
                rating[t] = base + carryover * (rating[t] - base)
            last_season[t] = season
        rh, ra = rating[h], rating[a]
        p = 1.0 / (1.0 + 10 ** (-(rh + (0.0 if neutral else home_adv) - ra) / 400.0))
        pre_h.append(rh); pre_a.append(ra); prob.append(p)
        rating[h] = rh + k * (win - p)                 # update AFTER recording pre-game values
        rating[a] = ra - k * (win - p)
    return pd.DataFrame({"ELO_HOME": pre_h, "ELO_AWAY": pre_a, "ELO_PROB_HOME": prob},
                        index=g["GAME_ID"].to_numpy()).reindex(games["GAME_ID"].to_numpy())


def team_features(games: pd.DataFrame, ranking: pd.DataFrame | None = None, windows=(5, 10)) -> pd.DataFrame:
    """All team-level features on the long (team, game) table."""
    tg = to_team_games(games)
    parts = [tg[["GAME_ID", "TEAM_ID", "IS_HOME", "GAME_DATE", "SEASON", "GAME_TYPE"]],
             rolling_form(tg, windows), rest_days(tg), win_streak(tg), season_record(tg)]
    if ranking is not None:
        parts.append(pregame_standings(tg, ranking))
    return pd.concat(parts, axis=1)


def build_game_features(games: pd.DataFrame, ranking: pd.DataFrame | None = None, windows=(5, 10)) -> pd.DataFrame:
    """One row per GAME_ID: HOME_* and AWAY_* team features, home-minus-away diffs and Elo."""
    tf = team_features(games, ranking, windows)
    feat_cols = [c for c in tf.columns if c not in ("GAME_ID", "TEAM_ID", "IS_HOME", "GAME_DATE", "SEASON", "GAME_TYPE")]
    home = tf[tf.IS_HOME == 1].set_index("GAME_ID")[feat_cols].add_prefix("HOME_")
    away = tf[tf.IS_HOME == 0].set_index("GAME_ID")[feat_cols].add_prefix("AWAY_")
    out = home.join(away)
    for c in feat_cols:
        if c.startswith(("WIN_", "PTS", "PLUS_MINUS", "SEASON_W_PCT", "ST_W_PCT")) or c in ("REST_DAYS", "STREAK"):
            out[f"DIFF_{c}"] = out[f"HOME_{c}"] - out[f"AWAY_{c}"]
    out = out.join(elo_ratings(games))
    out["DIFF_ELO"] = out["ELO_HOME"] - out["ELO_AWAY"]
    return out.reindex(games["GAME_ID"].to_numpy())


# --------------------------------------------------------------------------- leakage tests
_SWAP_PAIRS = [(f"{s}_home", f"{s}_away") for s in SAME_GAME_STATS]


def flip_results(games: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Copy of ``games`` where the selected games' results are reversed (home and away stat lines swapped)."""
    g = games.copy()
    for h, a in _SWAP_PAIRS:
        g.loc[mask, [h, a]] = g.loc[mask, [a, h]].to_numpy()
    g.loc[mask, "HOME_TEAM_WINS"] = 1 - g.loc[mask, "HOME_TEAM_WINS"]
    return g


def _compare(f_full: pd.DataFrame, f_test: pd.DataFrame, ids) -> None:
    a = f_full.loc[ids].select_dtypes("number")
    b = f_test.loc[ids, a.columns]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)


def assert_no_leakage(feature_fn, games: pd.DataFrame, cutoff_dates, ranking: pd.DataFrame | None = None) -> int:
    """Raise AssertionError if any feature of a game depends on its own result or on later games.

    For each cutoff date D:
      * every game AFTER D is deleted,
      * every game ON D gets its result flipped,
      * (if given) every ranking row dated ON or AFTER D is scrambled,
    then the features are recomputed. Features of all games dated <= D must be
    identical to the full-data features. Returns the number of game rows checked.
    """
    f_full = feature_fn(games, ranking)
    checked = 0
    for d in cutoff_dates:
        d = pd.Timestamp(d)
        g = games[games["GAME_DATE"] <= d]
        g = flip_results(g, g["GAME_DATE"] == d)
        r = None
        if ranking is not None:
            r = ranking.copy()
            late = r["STANDINGSDATE"] >= d
            r.loc[late, ["G", "W", "L"]] = r.loc[late, ["G", "W", "L"]] + 7
            r.loc[late, "W_PCT"] = 1 - r.loc[late, "W_PCT"]
            r.loc[late, ["HOME_RECORD", "ROAD_RECORD"]] = "41-0"
        f_test = feature_fn(g, r)
        _compare(f_full, f_test, g["GAME_ID"].to_numpy())
        checked += len(g)
    return checked


def leaky_rolling_example(games: pd.DataFrame, ranking=None) -> pd.DataFrame:
    """A deliberately WRONG feature (rolling mean without shift) used to show the test catches leakage."""
    tg = to_team_games(games)
    gid = tg.groupby(["TEAM_ID", "SEASON"]).ngroup()
    tg["WIN_LAST5_LEAKY"] = tg.groupby(gid)["WIN"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    return tg[tg.IS_HOME == 1].set_index("GAME_ID")[["WIN_LAST5_LEAKY"]].reindex(games["GAME_ID"].to_numpy())
