# Meeting Prep: NBA Home-Win Prediction

Numbers come from `notebooks/01_data_investigation.ipynb`.

## 1. Objective
- **Problem:** predict whether the home team wins an NBA game, using only pre-game information. Useful for understanding what drives wins and how much home-court advantage matters.
- **Task:** binary **classification**. **Target:** `HOME_TEAM_WINS` (1 = home team won, 0 = away team won).

## 2. Dataset
- **Source:** Kaggle "NBA games data" (Nathan Lauga), public CSVs, seasons 2003-04 to 2022-23. Accessible and usable.
- **Main file:** `games.csv`, **one row = one game**, 26,651 rows x 21 columns -> **20 input features** (excl. target). 24,848 usable games after cleaning.
- **Features:** date, game/team/season ids, and same-game box-score stats for home and away (`PTS`, `FG_PCT`, `FT_PCT`, `FG3_PCT`, `AST`, `REB`). All numeric except the date; team ids are categorical codes. No text or image data.
- **Other files:** `games_details` (player box scores), `ranking` (daily standings), `players`, `teams` (lookup). Optional extra features.

## 3. Exploration done
- Sample rows, dtypes, summary stats; plots of class balance, home win rate by season, points per team by season.
- **Missing:** 99 games with no stats (2003 preseason, placeholder label 0) -> dropped.
- **Duplicates:** 29 duplicated `GAME_ID`s (Dec 2020 - Jan 2021) -> dropped. 1,774 preseason games excluded.
- **Target:** 59.1% home wins vs 40.9% away wins, moderately imbalanced. Majority-class baseline = ~59%.
- **Oddities:** home advantage dips during COVID seasons (~55%); scoring rises over the years (era drift).

## 4. Suitability and challenges
- **Enough data?** Yes, ~24.8K games, 20 seasons. **Target clearly defined and available:** yes.
- **Leakage (main risk):** the same-game box score predicts the label almost perfectly (`PTS_home > PTS_away` = 100%, `FG_PCT` alone = 79.6%). Those columns can't be features. Features must use earlier games only (rolling win %, rest days, streaks, Elo). `ranking.csv` snapshots include that day's games, so only the previous day's is safe.
- **Feature relevance:** the raw columns are post-game, so we'd build pre-game features like rolling win %, rest days and Elo.
- **Repeated entities / time:** the same 30 teams repeat thousands of times and games are ordered in time -> **split by season, never shuffle** (train 2003-2016, validation 2017-18, test 2019-22).
- **Distribution shift:** test set contains the COVID seasons (weaker home edge, bubble games at neutral sites).
- **Preprocessing ahead:** build pre-game features, home-minus-away differences, scaling for linear/NN models.

## 5. Initial plan
- **Models:** logistic regression (baseline) -> random forest / gradient boosting (ensemble) -> small MLP (deep learning).
- **Model rationale:** logistic regression as an interpretable baseline, tree ensembles for non-linear interactions, and an MLP because the rubric asks for deep learning.
- **Evaluation:** accuracy vs the ~59% majority baseline, plus log-loss and ROC-AUC, on the season split.
- **Team split (proposed):**
  1. Dinh Hung Nguyen: all feature engineering (rolling stats, rest days, streaks, standings, Elo)
  2. Justin Lin: baseline models (logistic regression)
  3. Anthony Le: tree / ensemble models (random forest, gradient boosting)
  4. Will Berg: deep learning (MLP)
  5. Lam Nguyen: evaluation, plots, and presentation

  Everyone documents their own part in the shared notebook. The split is a starting point and can change.
- **Questions need clarification:**
  1. Include playoffs, or regular season only?
  2. Is a COVID-heavy test set (2019-2022) acceptable?
  3. Is a specific metric required (accuracy vs log-loss/AUC)?
