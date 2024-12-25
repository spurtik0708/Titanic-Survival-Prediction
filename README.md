# Titanic Survival Prediction

## Objective
The goal of this project is to build a machine learning model that predicts whether a passenger survived the Titanic disaster based on various features such as age, gender, ticket class, fare, and cabin. This project demonstrates essential skills in data preprocessing, feature engineering, and model training for classification tasks.

---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Analyzed the dataset to understand feature distributions and relationships with the target variable (`Survived`).
- Visualized survival rates across different passenger classes and genders.
- Key insights:
  - Higher survival rates observed in first-class passengers and females.
  - Missing values were significant in `Age` and `Embarked` columns.

### 2. Data Preprocessing
- **Missing Values**:
  - Imputed missing `Age` values with the median.
  - Filled missing `Embarked` values with the mode.
- **Feature Encoding**:
  - Converted categorical variables like `Sex` and `Embarked` to numerical values using Label Encoding.
- **Feature Selection**:
  - Selected features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.
  - Dropped irrelevant features such as `Name` and `Ticket`.

### 3. Model Development
- Used Random Forest Classifier for training due to its robustness and ability to handle categorical features effectively.
- Split the dataset into training (80%) and testing (20%) sets to evaluate model performance.
- Performed hyperparameter tuning for optimal results.

### 4. Evaluation
- Used the following metrics to assess model performance:
  - **Accuracy**: Overall correctness of predictions.
  - **Classification Report**: Precision, recall, F1-score for each class.
  - **Confusion Matrix**: Visualized true positives, false positives, etc.

---

## Challenges Faced
1. **Handling Missing Values**:
   - Missing `Age` data required careful imputation to avoid bias.
2. **Feature Engineering**:
   - Determining the most relevant features to improve model accuracy.
3. **Class Imbalance**:
   - Some classes (e.g., survival) were underrepresented, affecting model training.

---

## Results
- Achieved an accuracy of approximately **85%** on the test dataset.
- Insights gained from feature importance:
  - `Sex` and `Pclass` were the most significant predictors of survival.

---

## Repository Structure
```
├── data
│   ├── titanic_dataset.csv  # Dataset used for this project
├── notebooks
│   ├── titanic_survival.ipynb  # Jupyter Notebook with the full code
├── models
│   ├── titanic_model.pkl  # Saved Random Forest model
├── README.md  # Project documentation
```

---

## How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/titanic-survival-prediction.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to reproduce the results.

---

## Dependencies
- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

Install dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

## Future Improvements
- Perform advanced hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Explore additional algorithms like Gradient Boosting or XGBoost.
- Incorporate ensemble methods to further improve prediction accuracy.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgements
- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- Scikit-learn Documentation
