# NBA Game Prediction Model

This project develops an AI-based system for predicting NBA game outcomes and point spreads using historical player and team box-score data. The main objective is to estimate both the probability that the home team wins and the expected home-team scoring margin. The system uses player-level box-score data from the 2021-22 through 2025-26 NBA seasons, aggregates those records into team-game observations, engineers rolling pre-game features, and trains multiple supervised learning models. To avoid data leakage, all rolling statistics are shifted so that each prediction uses only information available before the game being predicted.
The final system includes two major model families: a dense neural network implemented in Keras and a tabular machine learning model. The tabular model is designed to use TabPFN when available, with a HistGradientBoosting fallback when TabPFN is unavailable. The neural network and tabular model are trained for two related tasks: binary win/loss classification and spread regression. Model performance is evaluated on a chronological holdout test set containing 1,179 games. The Dense Neural Network achieved 65.7% classification accuracy, 0.721 ROC AUC, 11.36 point spread MAE, and 14.55 point RMSE. The HistGradientBoosting fallback achieved 65.6% classification accuracy, 0.705 ROC AUC, 11.66 point spread MAE, and 14.78 point RMSE.
The project also includes a matchup prediction interface and an odds-market comparison prototype. These outputs translate model predictions into interpretable game-level summaries, including home win probability, predicted spread, predicted winner, and potential differences between model estimates and sportsbook lines. Overall, the project demonstrates how modern AI and tabular machine learning can be applied to sports analytics, while also showing the limits of prediction in a high-variance domain like professional basketball.
<img width="468" height="207" alt="image" src="https://github.com/user-attachments/assets/bee3ce86-3b95-48d0-b7ff-d5f0fbb6a4f9" />


## Models

Four models are trained and compared:

1. **Dense NN (Keras)** — Win/Loss classification
2. **Dense NN (Keras)** — Points regression (for predicting spreads/totals)
3. **TabPFN** — Win/Loss classification
4. **TabPFN** — Points regression

## Feature Engineering

- Rolling shooting percentages (FG%, 3P%, FT%) over a 10-game window
- Rolling offensive and defensive stats (PTS, REB, AST, STL, BLK, TOV)
- Last 10 games form (recent win count)
- Missing player impact (combined USG% and PPG of absent key players)
- Home/away matchup framing

All rolling features are computed from games *prior to* the target game to avoid data leakage.

## Data

Four CSVs are included in this repo:

- `box_scores.csv` — per-player box scores per game
- `player_advanced.csv` — advanced player stats (USG%, PIE, etc.)
- `player_pergame.csv` — per-game player averages
- `team_standings.csv` — team standings by season

## Running the notebook

The notebook (`Full_NBA_Prediction_Model.ipynb`) was originally written for Google Colab. To run it locally:

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/nba-prediction-model.git
   cd nba-prediction-model
   ```

2. Install dependencies:
   ```bash
   pip install nba_api pandas numpy scikit-learn matplotlib seaborn tensorflow tabpfn
   ```

3. Open the notebook in Jupyter or VS Code:
   ```bash
   jupyter notebook Full_NBA_Prediction_Model.ipynb
   ```

4. If running outside Colab, skip the `drive.mount(...)` cell and update the `DATA_DIR` variable to point to this repo's root (where the CSVs live).

## Project Structure

```
nba-prediction-model/
├── Full_NBA_Prediction_Model.ipynb   # Main notebook: feature engineering, training, evaluation
├── nba_scraper.py                    # Script that scraped/built the CSV datasets
├── box_scores.csv                    # Player-level box scores
├── player_advanced.csv               # Advanced player stats
├── player_pergame.csv                # Per-game player averages
├── team_standings.csv                # Team standings
├── NBA_Prediction_Model_Report.docx  # Written project report
├── NBA_Presentation_Final.pdf        # Slide deck
├── README.md
└── .gitignore
```


