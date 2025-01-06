# Scrabble Player Rating Project

## Project Overview

This project aims to analyze and predict Scrabble player ratings based on game and player features. The analysis involves detailed feature engineering from multiple datasets to extract meaningful insights and build predictive models.

### Process Summary

1. **Data Loading**:
   - Data is loaded from four main CSV files:
     - `games.csv`: Contains metadata about games such as `game_id`, `first`, `time_control_name`, `game_end_reason`, `winner`, and more.
     - `turns.csv`: Contains detailed information about each turn, including `rack`, `move`, `points`, and `score`.
     - `train.csv`: Includes player scores and ratings before the game.
     - `test.csv`: Similar to `train.csv` but with missing ratings for prediction.

2. **Initial Feature Engineering**:
   - **Games Dataset**:
     - Added `negative_end_reason` (binary feature for unusual game endings).
     - Encoded `game_end_reason` and `lexicon` using `LabelEncoder`.
     - Removed redundant columns like `time_control_name`, `increment_seconds`, and `created_at`.
   - **Turns Dataset**:
     - Added features such as `rack_len`, `rack_usage`, `letter_score`, and `average_letter_score`.
     - Calculated `mean_average_letter_score` for each player in a game.
   - **Train Dataset**:
     - Engineered features like `bot_mean_rack_usage`, `user_mean_rack_usage`, and `mean_average_letter_score`.
     - Merged with games data for a comprehensive dataframe.
   - **Final Dataset**:
     - Created `f_train_df` with selected features from the train and games datasets.
     - Dropped unnecessary columns like `bot_name`, `user_name`, and `game_end_reason`.

3. **Planned Feature Engineering**:
   - Further enhancements to refine the dataset for better predictive modeling.

---

## Data Source

The data for this project can be downloaded from the [Kaggle Scrabble Player Rating Competition](https://www.kaggle.com/competitions/scrabble-player-rating/data). Follow these steps to obtain the data:

1. Register or log in to Kaggle.
2. Navigate to the competition's data page.
3. Download the following files:
   - `games.csv`: Metadata about games (e.g., who went first, time controls).
   - `turns.csv`: All turns from start to finish of each game.
   - `train.csv`: Final scores and ratings for each player in each game; ratings for each player are as of BEFORE the game was played.
   - `test.csv`: Similar to `train.csv` but with missing ratings for prediction.
   - `sample_submission.csv`: A sample submission file in the correct format.

### Data Directory

Once downloaded, place the files in the following directory structure:

```
data/
   source_data/
      games.csv
      turns.csv
      train.csv
      test.csv
      sample_submission.csv
```

Ensure the data is correctly placed for the project scripts to work seamlessly.

---

## Future Development

- Advanced feature engineering to enhance predictive performance.
- Implementation of machine learning models to predict player ratings.
- Visualization and analysis of player performance trends.

---

## Contributing

Contributions to this project are welcome. Please feel free to submit issues or feature requests.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Author

Liav Ariel