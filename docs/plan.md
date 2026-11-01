# Project Plan: NBA Home-Win Prediction

**Goal:** predict whether the home team wins using only information available before tip-off. **Baseline to beat:** 59% (home teams win about 59% of games). Professor's guidance: about 70% is a good result if the data work is done properly.

**Milestones:** November status meeting (steps 1-3 done). Final presentation and notebook: Nov 30 - Dec 2.

## Ground rules
1. **No IDs as model inputs** (`GAME_ID` and all team ids), so the model can't memorize them.
2. **No same-game stats** (points, FG%, rebounds, etc. of the game being predicted). They are only known after the game.
3. **Date and season are not inputs.** They are used only to order games and to split the data.
4. **One split for everyone** (step 2), never shuffled.

## Step 1. Build one modeling table (merge + features)
- Clean `games.csv`: drop 29 duplicate games, 99 games with no stats, 1,774 preseason games -> 24,848 games.
- Reshape to one row per team per game, so we can look at a team's earlier games.
- Add form from earlier games only: rolling 5/10-game averages, rest days, win streak, season record.
- **Merge `ranking.csv`:** attach each team's standings from the latest date strictly before the game (the same-day snapshot already contains that game's result).
- Add Elo rating (the rating before the game, updated afterwards).
- Back to one row per game: `HOME_*` and `AWAY_*` columns, plus home-minus-away differences. Drop all ids.
- `games_details.csv` is optional and later (player-level features); skip for now.
- **Checks:** row count stays 24,848; no duplicate games; unmatched standings rows counted (expect about 59 games, mostly season openers); leakage test passes (deleting later games doesn't change earlier features).

## Step 2. Fix the split
| Part | Seasons | Games |
|---|---|---|
| Train | 2003-2016 | 18,110 |
| Validation | 2017-2018 | 2,624 |
| Test | 2019-2022 | 4,114 |

- Never shuffled: the model learns from the past and predicts later games.
- Tune inside the training years with forward-chaining folds (train on earlier seasons, validate on the next one).
- The test set is looked at once, at the end. It covers the COVID seasons, when the home win rate fell to about 55%, so test scores will look lower than validation.

## Step 3. Baselines (enough for the November meeting)
- Majority class (always predict home win).
- Elo only.
- Logistic regression on the features.
- Earlier trial run: about 66% validation / 63% test, vs 59% / 55% for majority.

## Step 4. Preprocessing for the models that need it
- Scale features for logistic regression, KNN, SVM and neural networks.
- Fill the few missing values (first games of a season) the same way for everyone.

## Step 5. Models (from the proposal)
Logistic regression, KNN, decision tree, SVM (linear and Gaussian), bagging / random forest, boosting (XGBoost, CatBoost), stacking or voting, neural networks including a deeper one, plus Elo as a standalone predictor and as a feature.

## Step 6. Tune and compare
- Metrics for every model: accuracy, precision, recall, F1, ROC-AUC, log loss, all on the same split.
- Analyze which features matter most.

## Step 7. Final notebook and presentation
Data exploration, preprocessing, models, results, conclusions.

## Who does what (proposed, adjust as needed)
| Person | Role |
|---|---|
| Dinh Hung Nguyen | Steps 1-2: merge, features, split |
| Justin Lin | Step 3: baselines (logistic regression) |
| Anthony Le (team leader) | Tree / ensemble models |
| Will Berg | Neural networks |
| Lam Nguyen | Evaluation, plots, presentation |

Models in steps 5-6 start once step 1 is done, so the feature table is the first handoff.

## Open questions
1. Playoffs in or out (24,848 games vs 23,244 regular season only)?
2. Keep the 172 COVID "bubble" games, with a neutral-site flag?
3. Is a COVID-heavy test set (2019-2022) acceptable, or evaluate season by season?
