
# Scrabble Player Rating Project

## Project Overview

This project aims to analyze and predict Scrabble player ratings based on game and player features. Using data from multiple sources, the project involves detailed feature engineering, data preprocessing, and predictive modeling to understand player performance trends and predict ratings for unseen data.

---

## Process Summary

### **1. Exploratory Data Analysis (EDA)**
The project begins with an exploratory data analysis (EDA) phase to understand the structure and contents of the data. The EDA utilizes four main CSV files:

- **`games.csv`**: Contains metadata about games, such as `game_id`, `first`, `time_control_name`, `game_end_reason`, `winner`, and more.
- **`turns.csv`**: Includes detailed information about each turn, such as `rack`, `move`, `points`, and `score`.
- **`train.csv`**: Contains player scores and ratings before the game.
- **`test.csv`**: Similar to `train.csv` but with missing ratings for prediction.

For a detailed walkthrough of the EDA process, refer to the **EDA Notebook** located in the `notebooks/` directory. This notebook provides visualizations, data summaries, and initial insights used to guide feature engineering and model development.


### **2. Initial Feature Engineering**
- **Games Dataset**:
  - Added `negative_end_reason` to flag unusual game endings.
  - Encoded categorical columns like `game_end_reason` and `lexicon` using `LabelEncoder`.
  - Removed redundant columns such as `time_control_name`, `increment_seconds`, and `created_at`.

- **Turns Dataset**:
  - Engineered features such as `rack_len`, `rack_usage`, `letter_score`, and `average_letter_score`.
  - Computed `mean_average_letter_score` for each player in a game.

- **Train Dataset**:
  - Added aggregated features like `bot_mean_rack_usage`, `user_mean_rack_usage`, and `mean_average_letter_score`.
  - Merged with the games data for a comprehensive feature set.

- **Final Dataset**:
  - Created `f_train_df` with selected features from the train and games datasets.
  - Dropped unnecessary columns such as `bot_name`, `user_name`, and `game_end_reason`.

### **3. Model & Pipeline Development**
Significant progress includes:
- **Pipeline Development**:
  - Implemented a transformer-based pipeline for preprocessing and feature engineering, ensuring modularity and reusability.
- **Feature Refinement**:
  - Enhanced feature engineering with turn-level insights for better predictive performance.
- **Model Evaluation**:
  - Introduced a unified `ModelHandler` class to dynamically handle models from different libraries (e.g., scikit-learn, XGBoost).
  - Conducted hyperparameter tuning and cross-validation to optimize model performance.
- **Error Analysis**:
  - Improved False Analysis by evaluating the model on a validation set, identifying patterns in incorrect predictions, and refining features.

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

### **Data Directory**
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

## How to Run the Project

### Using PyCharm

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/liav-ariel/scrabble-rating.git
   cd scrabble-rating
   ```

2. **Set Up the Environment**:
   - Open the project in PyCharm.
   - Configure a Python interpreter with the required dependencies.
   - Install dependencies by running:
     ```bash
     pip install -r requirements.txt
     ```
---

## Future Development

- **Advanced Feature Engineering**:
  - Incorporate game-level trends and contextual features for improved predictions.
  - Address outliers and missing data with advanced techniques.

- **Visualization**:
  - Develop dashboards for player performance trends and feature importance visualization.

- **Deployment**:
  - Build an API for live predictions using Flask or FastAPI.
  - Add a web interface for data uploads and prediction viewing.

---

## Contributing

Contributions are welcome! Submit issues or feature requests, or create pull requests to help improve the project.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Author

Liav Ariel
