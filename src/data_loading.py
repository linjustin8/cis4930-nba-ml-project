"""Loading, describing, and cleaning the Kaggle "NBA games" dataset (Nathan Lauga).

All paths are resolved relative to this file, so the functions work no matter
where the notebook or script is launched from. Nothing here writes to
``src/dataset/``; every cleaning step returns a new DataFrame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "dataset"

FILES = {
    "games": "games.csv",
    "games_details": "games_details.csv",
    "ranking": "ranking.csv",
    "players": "players.csv",
    "teams": "teams.csv",
}

# First digit of GAME_ID (and of ranking.SEASON_ID) encodes the game type.
GAME_TYPES = {1: "Preseason", 2: "Regular Season", 3: "All-Star", 4: "Playoffs", 5: "Play-In"}

# 2019-20 restart in the Orlando "bubble": every game from this date on was played
# at a neutral site, so "home" is only a label.
BUBBLE_START = pd.Timestamp("2020-07-30")

# Same-game statistics in games.csv. They are only known after the final buzzer,
# so they can never be features for the game they describe.
SAME_GAME_STATS = ["PTS", "FG_PCT", "FT_PCT", "FG3_PCT", "AST", "REB"]
SAME_GAME_COLUMNS = [f"{s}_{side}" for side in ("home", "away") for s in SAME_GAME_STATS]

COLUMN_DESCRIPTIONS = {
    "games": {
        "GAME_DATE_EST": "Game date (US Eastern time).",
        "GAME_ID": "Unique game id. 1st digit = game type (1 preseason, 2 regular, 4 playoffs, 5 play-in), "
                   "next 2 = season, last 5 = game number.",
        "GAME_STATUS_TEXT": "Game status; 'Final' for every row.",
        "HOME_TEAM_ID": "Home team id (franchise id, stable across relocations).",
        "VISITOR_TEAM_ID": "Away (visiting) team id.",
        "SEASON": "Season start year (2022 = the 2022-23 season).",
        "TEAM_ID_home": "Home team id again (redundant copy of HOME_TEAM_ID).",
        "PTS_home": "Points scored by the home team in this game.",
        "FG_PCT_home": "Home field-goal percentage in this game (0-1).",
        "FT_PCT_home": "Home free-throw percentage in this game (0-1).",
        "FG3_PCT_home": "Home three-point percentage in this game (0-1).",
        "AST_home": "Home assists in this game.",
        "REB_home": "Home total rebounds in this game.",
        "TEAM_ID_away": "Away team id again (redundant copy of VISITOR_TEAM_ID).",
        "PTS_away": "Points scored by the away team in this game.",
        "FG_PCT_away": "Away field-goal percentage in this game (0-1).",
        "FT_PCT_away": "Away free-throw percentage in this game (0-1).",
        "FG3_PCT_away": "Away three-point percentage in this game (0-1).",
        "AST_away": "Away assists in this game.",
        "REB_away": "Away total rebounds in this game.",
        "HOME_TEAM_WINS": "TARGET: 1 if the home team won, else 0.",
    },
    "games_details": {
        "GAME_ID": "Game id (joins to games.csv).",
        "TEAM_ID": "Team the player played for in this game.",
        "TEAM_ABBREVIATION": "Team abbreviation at the time (e.g. SEA before OKC).",
        "TEAM_CITY": "Team city at the time.",
        "PLAYER_ID": "Unique player id.",
        "PLAYER_NAME": "Player full name.",
        "NICKNAME": "Player nickname (rarely filled).",
        "START_POSITION": "F / C / G if the player started; blank = came off the bench or did not play.",
        "COMMENT": "Why the player did not play (e.g. 'DNP - Coach's Decision', 'DND - Injury/Illness').",
        "MIN": "Minutes played, stored as text ('MM:SS' or a plain number).",
        "FGM": "Field goals made.", "FGA": "Field goals attempted.", "FG_PCT": "Field-goal percentage.",
        "FG3M": "Three-pointers made.", "FG3A": "Three-pointers attempted.", "FG3_PCT": "Three-point percentage.",
        "FTM": "Free throws made.", "FTA": "Free throws attempted.", "FT_PCT": "Free-throw percentage.",
        "OREB": "Offensive rebounds.", "DREB": "Defensive rebounds.", "REB": "Total rebounds.",
        "AST": "Assists.", "STL": "Steals.", "BLK": "Blocks.", "TO": "Turnovers.", "PF": "Personal fouls.",
        "PTS": "Points.", "PLUS_MINUS": "Point differential while the player was on the floor.",
    },
    "ranking": {
        "TEAM_ID": "Team id.",
        "LEAGUE_ID": "League id: NBA, written inconsistently as '0' or '00'.",
        "SEASON_ID": "1st digit = phase (1 preseason, 2 regular season), last 4 = season start year.",
        "STANDINGSDATE": "Date of the standings snapshot; it INCLUDES games played on that date.",
        "CONFERENCE": "East or West.",
        "TEAM": "Team city name (e.g. 'L.A. Lakers').",
        "G": "Games played so far in this phase.", "W": "Wins so far.", "L": "Losses so far.",
        "W_PCT": "Win percentage so far (W / G).",
        "HOME_RECORD": "Home record as text 'W-L'.", "ROAD_RECORD": "Road record as text 'W-L'.",
        "RETURNTOPLAY": "2019-20 only: 1 if the team was invited to the bubble restart.",
    },
    "players": {
        "PLAYER_NAME": "Player full name.",
        "TEAM_ID": "Team the player was on.",
        "PLAYER_ID": "Unique player id.",
        "SEASON": "Season start year (2009-2019 only).",
    },
    "teams": {
        "LEAGUE_ID": "League id ('00' = NBA).",
        "TEAM_ID": "Team id (joins to every other file).",
        "MIN_YEAR": "First season of the franchise.",
        "MAX_YEAR": "Last season covered by the source (2019 for all teams: this is a snapshot).",
        "ABBREVIATION": "Current abbreviation.", "NICKNAME": "Team nickname (e.g. 'Hawks').",
        "YEARFOUNDED": "Year founded.", "CITY": "Current city.", "ARENA": "Current arena.",
        "ARENACAPACITY": "Arena capacity.", "OWNER": "Owner (snapshot).",
        "GENERALMANAGER": "General manager (snapshot).", "HEADCOACH": "Head coach (snapshot).",
        "DLEAGUEAFFILIATION": "G League affiliate.",
    },
}

DATE_COLUMNS = {"games": "GAME_DATE_EST", "ranking": "STANDINGSDATE"}


# --------------------------------------------------------------------------- loading
def load_raw(name: str) -> pd.DataFrame:
    """Read one raw CSV exactly as shipped (only LEAGUE_ID kept as text)."""
    return pd.read_csv(DATA_DIR / FILES[name], dtype={"LEAGUE_ID": str}, low_memory=False)


def load_all_raw() -> dict[str, pd.DataFrame]:
    return {name: load_raw(name) for name in FILES}


def game_type(game_id: pd.Series | int) -> pd.Series | int:
    """Numeric game-type code (see GAME_TYPES) from an integer GAME_ID such as 22200477."""
    return game_id // 10_000_000


# --------------------------------------------------------------------------- describing
def file_overview(df: pd.DataFrame, name: str) -> dict:
    """Shape, memory, duplicates and date range of one file."""
    info = {
        "file": FILES[name],
        "rows": len(df),
        "columns": df.shape[1],
        "memory_MB": round(df.memory_usage(deep=True).sum() / 1e6, 1),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_range": "",
    }
    if name in DATE_COLUMNS:
        d = pd.to_datetime(df[DATE_COLUMNS[name]])
        info["date_range"] = f"{d.min():%Y-%m-%d} to {d.max():%Y-%m-%d}"
    elif "SEASON" in df.columns:
        info["date_range"] = f"seasons {df.SEASON.min()} to {df.SEASON.max()}"
    return info


def column_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """One row per column: dtype, meaning, missing count / %, number of unique values."""
    desc = COLUMN_DESCRIPTIONS.get(name, {})
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "description": [desc.get(c, "") for c in df.columns],
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })


# --------------------------------------------------------------------------- cleaning
def box_score_totals(details: pd.DataFrame) -> pd.DataFrame:
    """Team totals per (GAME_ID, TEAM_ID) recomputed from cleaned player box scores."""
    t = details.groupby(["GAME_ID", "TEAM_ID"])[["PTS", "FGM", "FGA", "FTM", "FTA", "FG3M", "FG3A", "AST", "REB"]].sum()
    return pd.DataFrame({
        "PTS": t.PTS, "FG_PCT": t.FGM / t.FGA, "FT_PCT": t.FTM / t.FTA, "FG3_PCT": t.FG3M / t.FG3A,
        "AST": t.AST, "REB": t.REB,
    }).round(3)


def _box_score_distance(rows: pd.DataFrame, box: pd.DataFrame) -> pd.Series:
    """Sum of absolute differences between each games.csv row and its box-score totals."""
    dist = pd.Series(0.0, index=rows.index)
    for side, team_col in (("home", "HOME_TEAM_ID"), ("away", "VISITOR_TEAM_ID")):
        ref = box.reindex(pd.MultiIndex.from_arrays([rows["GAME_ID"], rows[team_col]]))
        for s in SAME_GAME_STATS:
            dist += np.abs(rows[f"{s}_{side}"].to_numpy() - ref[s].to_numpy())
    return dist.fillna(0)


def clean_games(games: pd.DataFrame, drop_preseason: bool = True, details: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a cleaned copy of games.csv.

    1. Round the percentage columns to 3 decimals (removes float noise such as 0.34299999).
    2. Drop the duplicate GAME_ID rows (29 games from Dec 2020 written twice). Most
       copies are identical after rounding. Two games disagree (one rebound, one FG%):
       if ``details`` (cleaned games_details) is given, the copy closest to the player
       box scores is kept, otherwise the first copy.
    3. Drop the 99 games with no stats: all 2003 preseason games whose
       HOME_TEAM_WINS is a placeholder 0, not a real result.
    4. Optionally drop all preseason games (not part of standings, starters rest).
    5. Add GAME_DATE (datetime), GAME_TYPE (name) and NEUTRAL_SITE (2020 bubble),
       and sort chronologically.
    """
    g = games.copy()
    pct = [c for c in SAME_GAME_COLUMNS if "PCT" in c]
    g[pct] = g[pct].round(3)
    if details is not None:
        g["_dist"] = 0.0
        dup = g["GAME_ID"].duplicated(keep=False)
        g.loc[dup, "_dist"] = _box_score_distance(g[dup], box_score_totals(details))
        g = g.sort_values("_dist", kind="stable").drop(columns="_dist")
    g = g.drop_duplicates("GAME_ID", keep="first")
    g = g[g["PTS_home"].notna()]
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE_EST"])
    g["GAME_TYPE"] = game_type(g["GAME_ID"]).map(GAME_TYPES)
    if drop_preseason:
        g = g[g["GAME_TYPE"] != "Preseason"]
    g["NEUTRAL_SITE"] = (g["SEASON"] == 2019) & (g["GAME_DATE"] >= BUBBLE_START)
    g = g.drop(columns=["TEAM_ID_home", "TEAM_ID_away", "GAME_STATUS_TEXT"])
    return g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def clean_games_details(details: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (GAME_ID, PLAYER_ID) rows, strip COMMENT whitespace, add MIN as float and DID_PLAY."""
    d = details.drop_duplicates(["GAME_ID", "PLAYER_ID"], keep="first").copy()
    d["COMMENT"] = d["COMMENT"].str.strip()
    d["MINUTES"] = parse_minutes(d["MIN"])
    d["DID_PLAY"] = d["MIN"].notna()
    return d


def parse_minutes(minutes: pd.Series) -> pd.Series:
    """Convert MIN text ('38:06', '38', '38.000000:06') to float minutes."""
    s = minutes.astype("string")
    parts = s.str.split(":", n=1, expand=True)
    mins = pd.to_numeric(parts[0], errors="coerce")
    secs = pd.to_numeric(parts[1], errors="coerce").fillna(0) if parts.shape[1] > 1 else 0
    return (mins + secs / 60).astype(float)


def clean_ranking(ranking: pd.DataFrame, regular_season_only: bool = True) -> pd.DataFrame:
    """Return ranking.csv with parsed dates, decoded SEASON_ID and no duplicate (team, date) rows.

    The duplicate (team, date) rows (Dec 2020 / Jan 2021) carry identical values, so
    keeping the first is lossless. With ``regular_season_only`` the preseason rows
    (SEASON_ID 1xxxx) are removed, since their W/L are preseason results.
    """
    r = ranking.copy()
    r["STANDINGSDATE"] = pd.to_datetime(r["STANDINGSDATE"])
    r["PHASE"] = (r["SEASON_ID"] // 10000).map(GAME_TYPES)
    r["SEASON"] = r["SEASON_ID"] % 10000
    if regular_season_only:
        r = r[r["PHASE"] == "Regular Season"]
    r = r.drop_duplicates(["TEAM_ID", "SEASON_ID", "STANDINGSDATE"], keep="first")
    return r.sort_values(["STANDINGSDATE", "TEAM_ID"]).reset_index(drop=True)


def to_team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Reshape cleaned games (one row per game) to one row per team per game.

    Columns: GAME_ID, GAME_DATE, SEASON, GAME_TYPE, TEAM_ID, OPP_ID, IS_HOME,
    WIN, PTS, PTS_OPP, PLUS_MINUS and the team's FG/FT/3P %, AST, REB.
    """
    base = ["GAME_ID", "GAME_DATE", "SEASON", "GAME_TYPE"]
    frames = []
    for side, opp, team_col, opp_col, is_home in (
        ("home", "away", "HOME_TEAM_ID", "VISITOR_TEAM_ID", 1),
        ("away", "home", "VISITOR_TEAM_ID", "HOME_TEAM_ID", 0),
    ):
        f = games[base].copy()
        f["TEAM_ID"] = games[team_col]
        f["OPP_ID"] = games[opp_col]
        f["IS_HOME"] = is_home
        f["WIN"] = games["HOME_TEAM_WINS"] if is_home else 1 - games["HOME_TEAM_WINS"]
        for s in SAME_GAME_STATS:
            f[s] = games[f"{s}_{side}"]
        f["PTS_OPP"] = games[f"PTS_{opp}"]
        f["PLUS_MINUS"] = f["PTS"] - f["PTS_OPP"]
        frames.append(f)
    tg = pd.concat(frames, ignore_index=True)
    return tg.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


# --------------------------------------------------------------------------- splitting
DEFAULT_SPLIT = {
    "train": list(range(2003, 2017)),   # 2003-04 ... 2016-17
    "validation": [2017, 2018],          # 2017-18, 2018-19
    "test": [2019, 2020, 2021, 2022],    # 2019-20 ... 2022-23 (partial)
}


def season_split(games: pd.DataFrame, split: dict[str, list[int]] = DEFAULT_SPLIT) -> pd.Series:
    """Label each game 'train' / 'validation' / 'test' by its SEASON (time-based, no shuffling)."""
    lookup = {season: part for part, seasons in split.items() for season in seasons}
    return games["SEASON"].map(lookup)


def majority_baseline(y_train: pd.Series, y_eval: pd.Series) -> tuple[int, float]:
    """Predict the training majority class for every game; return (class, accuracy on y_eval)."""
    majority = int(y_train.mode().iloc[0])
    return majority, float(np.mean(y_eval == majority))
