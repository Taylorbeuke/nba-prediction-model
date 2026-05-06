# NBA Game Prediction Model

A machine learning project that predicts NBA game outcomes (win/loss and final point totals) using two modeling approaches: a Dense Neural Network (Keras) and TabPFN. Built on rolling team statistics and missing-player impact features.

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

## Notes

This was originally built as a final project exploring tabular foundation models (TabPFN) versus traditional Dense NNs on a real-world prediction task.
