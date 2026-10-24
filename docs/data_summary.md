# Data Summary: NBA Home-Win Prediction (CIS 4930)

**Task:** binary classification. Predict `HOME_TEAM_WINS` (1 = home team won) using only information available before tip-off.
**Source:** Kaggle "NBA games data" (Nathan Lauga), 5 CSV files, seasons 2003-04 to 2022-23.
**Code:** `notebooks/01_data_investigation.ipynb`, `src/data_loading.py`, `src/features_prototype.py`. All numbers below come from that notebook.

## The files

| File | Rows × cols | Grain | Date range | Use |
|---|---|---|---|---|
| `games.csv` | 26,651 × 21 | 1 row per game | 2003-10-05 → 2022-12-22 | **Modeling table + label**; team box score of each game |
| `games_details.csv` | 668,628 × 29 | 1 row per player per game | same games | Player box scores (optional extra features) |
| `ranking.csv` | 210,342 × 13 | 1 row per team per day | 2003-10-01 → 2022-12-22 (7,023 days) | Pre-game standings |
| `players.csv` | 7,228 × 4 | player-team-season | 2009–2019 only | Not needed |
| `teams.csv` | 30 × 14 | 1 row per franchise | static ~2019 snapshot | ID → name lookup only |

`games.csv` columns:
- **Known before the game:** date, `GAME_ID`, home and away team IDs, `SEASON`.
- **Same-game results (leakage):** 12 stat columns, `PTS`, `FG_PCT`, `FT_PCT`, `FG3_PCT`, `AST`, `REB` for each of home and away.
- **Target:** `HOME_TEAM_WINS`.

## Key numbers

- **Modeling table: 24,848 games.** That is 23,244 regular season + 1,591 playoffs + 13 play-in, after removing 29 duplicate rows, the 99 no-stat games, and all preseason games.
- **20 seasons.** 2022-23 is partial: the data ends Dec 22, 2022 and has no playoffs.
- **Label balance: 59.1% home wins** (14,685) vs 40.9% away wins (10,163). The data is moderately imbalanced.
- **Time-based split by season, never shuffled:**

| Split | Seasons | Games | Home win rate = majority-baseline accuracy |
|---|---|---|---|
| Train | 2003–2016 (14) | 18,110 | 60.0% |
| Validation | 2017–2018 (2) | 2,624 | 58.9% |
| Test | 2019–2022 (4) | 4,114 | 55.4% |

- **Shortened seasons:**
  - 2011: 66-game lockout season
  - 2019: COVID suspension, then the Orlando bubble (172 neutral-site games)
  - 2020: 72-game season
  - 2022: partial

  **2021 is a full season.** It only looked short because fewer preseason games were recorded.
- **Home-court advantage is real but shrinking.** All 30 teams win more at home. The home win rate was about 59–61% per season through 2018, fell to about 55% in 2019–2021, and is back at 61% in the partial 2022 season.

## Leakage (shown in notebook section E)

- Same-game columns separate winners almost perfectly:
  - `PTS_home > PTS_away` gives **100%** accuracy (there are no ties).
  - `FG_PCT` alone gives **79.6%**.
- `ranking.csv` snapshots **include that day's games**: snapshot G = games played *including* today in 99.93% of cases. The rule "home W went up since yesterday" matches the label **100%** of the time.
- Therefore features use only **earlier games**, plus standings from the **day before** (strictly earlier date).

## Planned pre-game features (prototyped and tested: 80 columns)

All of these are built from earlier games or earlier snapshots only:
- Rolling averages of each team's **previous 5 / 10 games**: win %, points for and against, margin, FG%, 3P%, FT%, AST, REB.
- **Rest days**, a back-to-back flag, and a season-opener flag.
- **Win/loss streak** entering the game.
- **Season record** so far.
- **Previous-day standings** from `ranking.csv`: W, L, W%, home and road W%.
- **Elo rating**: K = 20, +100 for home court, 75% of the rating carried over between seasons.
- **Home-minus-away differences** of the above.

**Leakage test:**
- Setup: for 60 cutoff dates, every later game was deleted, every same-day result flipped, and every same-day-or-later standings row scrambled.
- Result: all 775,199 recomputed feature rows were identical to the full-data features.
- Sanity check: a deliberately leaky feature (rolling mean without `shift(1)`) **is caught** by the same test.

**Signal check (descriptive, nothing trained):** across the train seasons, the home win rate rises from 34% in the bottom Elo-difference quintile to 83% in the top one.

## Issues found and how we handle them

| Issue | Actual finding | Handling |
|---|---|---|
| Duplicate `GAME_ID`s | **29** games (Dec 23, 2020 – Jan 1, 2021) appear twice. 27 differ only by float rounding; 2 really conflict (1 rebound, 0.006 FG%) | Round pct columns; keep the copy that matches the player box scores |
| 99 games with no stats | All **2003 preseason** (Oct 7–24, 2003), no box scores, and every label is a **placeholder 0** | Drop them; they would teach a fake "home loses" pattern |
| Preseason mixed into `games.csv` | **1,774** preseason games (`GAME_ID` starts with 1) | Exclude from training rows and feature history |
| `games_details` nulls | 109,690 rows (16.4%) have MIN and all stats null at once: players who did not play (DNP 91.5K / DND 13.0K / NWT 5.1K) | Keep as `DID_PLAY = False`; not errors |
| `games_details` other problems | 289 duplicate player rows (early 2020-21); 40 box scores that don't sum to the team score (mostly 2003-05); 12 negative MIN values; MIN stored as text; COMMENT with whitespace variants; PLUS_MINUS often missing before 2014 | De-duplicate, parse MIN, strip text; base team features on `games.csv` |
| `ranking.csv` codes | `SEASON_ID` = phase digit + year: **1 = preseason, 2 = regular**. There is **no playoff code**: regular-season rows continue with frozen final records through the playoffs and offseason | Use only `2xxxx` rows of the same season |
| Multiple standings rows per team per day | **29** duplicate (team, date) pairs, Dec 2020 – Jan 2021, identical values | Drop the duplicates |
| Previous-day standings coverage | **24,789 / 24,848** games (99.76%) have a previous-day row for both teams. The misses are 114 season-opener team-games (→ 0–0 record) and 4 on the bubble restart day, 2020-07-30 (→ use the 2020-07-21 snapshot) | `merge_asof`, strictly earlier date, same season |
| Missing game | One 2022 game (`22200236`, ORL–IND) is absent from `games.csv` | Negligible; documented |
| Team IDs | Consistent across all 5 files (30 IDs); relocations SEA→OKC, NJN→BKN, NOH→NOP keep their IDs | None needed |
| Era drift | Points per team rose from 93.4 to 113.2 (2003 → 2022) | Use home-minus-away differences / within-season normalization |
| Distribution shift in test | COVID seasons have a weaker home edge (55%) | Report results vs the majority baseline per season; also use log-loss/Brier |

## Questions for the professor

1. Should playoffs be included (24,848 games), or should we use the regular season only (23,244 games)?
2. Should the 172 bubble games be kept with a neutral-site flag, dropped, or reported separately?
3. Is a COVID-heavy test set (2019–2022) acceptable? The alternative is a rolling season-by-season evaluation.
4. Cold start: should 2003 be used as an Elo/rolling warm-up season, or should rolling windows carry across seasons?
5. Is there a required metric (accuracy vs log-loss/AUC)?
