# Car Price Prediction

## Project Overview

This project develops a machine-learning model to predict the selling price of used cars.

The workflow includes:
- Data loading and inspection
- Exploratory data analysis (EDA)
- Data cleaning and preprocessing
- Train/test splitting
- Log transformation of the target
- Preprocessing pipelines
- Model comparison using 5-fold cross-validation
- Random Forest model selection
- Hyperparameter tuning with GridSearchCV
- Final evaluation on the untouched test set
- Feature-importance analysis
- A Dash prediction application
- Docker support

The target variable is `selling_price`.

## Dataset

The raw dataset contains 8,128 rows and 13 columns:

- `name`
- `year`
- `selling_price`
- `km_driven`
- `fuel`
- `seller_type`
- `transmission`
- `owner`
- `mileage`
- `engine`
- `max_power`
- `torque`
- `seats`

Missing values occur in `mileage`, `engine`, `max_power`, `torque`, and `seats`.

## Data Cleaning

### Remove CNG and LPG

CNG and LPG vehicles were removed because their mileage is measured in km/kg, while Diesel and Petrol vehicles use km/l. Keeping these measurements together would make the `mileage` feature inconsistent.

### Remove Test Drive Cars

Test Drive Cars were removed because their selling prices are unusually high compared with normal used cars and could distort the prediction model.

### Create `brand`

The first word of `name` was extracted as `brand`.

Examples:

```text
Maruti Swift Dzire VDI -> Maruti
Skoda Rapid 1.5 TDI Ambition -> Skoda
Honda City 2017-2020 EXi -> Honda
```

The original `name` column was then removed because it has very high categorical variety.

### Remove `torque`

The `torque` feature was removed as required by the assignment. Its values have inconsistent textual formats and would require additional parsing.

### Clean numerical features

- `mileage`: remove `kmpl` and convert to float
- `engine`: remove `CC` and convert to float
- `max_power`: remove `bhp` and convert to numeric

### Encode `owner`

The owner categories were mapped as:

```text
First Owner          -> 1
Second Owner         -> 2
Third Owner          -> 3
Fourth & Above Owner -> 4
```

## Exploratory Data Analysis

EDA was used to understand the target distribution and feature relationships.

The raw `selling_price` distribution is strongly right-skewed: most vehicles have lower prices, while a smaller number have very high prices.

Because of this large range and skewness, the target was log-transformed for model training.

## Target Log Transformation

The training target was transformed using:

```python
y_train_log = np.log(y_train)
```

Therefore, the models predict `log(selling_price)` rather than the original price.

Before calculating final metrics on the original price scale, predictions are converted back using:

```python
predicted_price = np.exp(predicted_log_price)
```

This inverse transformation is essential because the real `y_test` values are stored in the original selling-price scale.

## Train/Test Split

The target was separated from the features:

```python
X = df.drop(columns=["selling_price"])
y = df["selling_price"]
```

The data was split into training and test sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The test set was kept separate for final evaluation.

The training target was then log-transformed:

```python
y_train_log = np.log(y_train)
```

Preprocessing was placed inside the machine-learning pipeline so that preprocessing is learned from the appropriate training data during cross-validation.

## Preprocessing

A `ColumnTransformer` was used to apply different preprocessing to different feature groups.

### Continuous features

The continuous features were:

- `km_driven`
- `mileage`
- `engine`
- `max_power`

Missing values were imputed using the median and these features were standardized using `StandardScaler`.

### Other numerical features

The following numerical features were handled separately:

- `year`
- `owner`
- `seats`

Missing values were imputed using the most frequent value.

### Categorical features

The following categorical features were one-hot encoded:

- `brand`
- `fuel`
- `seller_type`
- `transmission`

`OneHotEncoder(handle_unknown="ignore")` was used so unknown categories do not cause prediction errors.

The preprocessing steps were included in the model pipeline to avoid fitting preprocessing outside the appropriate training data.

## Model Comparison

Five regression models were compared using 5-fold cross-validation:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)

The comparison metric was RMSE on the log-transformed target.

| Model | CV RMSE |
|---|---:|
| Random Forest | 0.213107 |
| Gradient Boosting | 0.226682 |
| Linear Regression | 0.258175 |
| Ridge | 0.258240 |
| SVR | 0.818603 |

Lower RMSE indicates better performance.

Random Forest achieved the lowest CV RMSE (`0.213107`) and was selected for hyperparameter tuning.

Random Forest performed well because it can model nonlinear relationships and interactions between vehicle characteristics.

## Hyperparameter Tuning

GridSearchCV was applied to the selected Random Forest model.

The searched parameters were:

```python
{
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}
```

A 5-fold cross-validation strategy with negative RMSE scoring was used.

The best parameters were:

```text
n_estimators = 200
max_depth = 20
min_samples_split = 5
min_samples_leaf = 1
```

## Final Model

The final model is a preprocessing pipeline followed by the tuned Random Forest regressor.

Random Forest parameters:

```text
n_estimators = 200
max_depth = 20
min_samples_split = 5
min_samples_leaf = 1
random_state = 42
```

## Final Test Evaluation

The final model was evaluated on the untouched test set.

Predictions were converted from log-price back to the original price scale using `np.exp()`.

| Metric | Result |
|---|---:|
| MAE | 69,043.28 |
| RMSE | 204,713.91 |
| R² | 0.94587 |

### Interpretation

The R² score of approximately 0.946 means that the model explains about 94.6% of the variation in selling prices in the test set.

The MAE is approximately 69,043 price units, representing the average absolute prediction error.

The RMSE is approximately 204,714. It is higher than MAE because RMSE gives greater weight to large errors.

The model performs well overall, although some high-priced vehicles have larger prediction errors.

## Feature Importance

Feature importance from the final Random Forest model was analysed after preprocessing.

The most important features were:

1. `year`
2. `max_power`
3. `engine`

`km_driven` and `mileage` also contribute to predictions.

This suggests that vehicle age and performance-related characteristics are important factors in used-car selling prices. Newer cars and cars with greater engine capacity or higher power tend to have higher predicted prices.

## Dash Application

A Dash application was created to allow users to enter vehicle information and obtain a predicted selling price.

The saved model is:

```text
app/code/car_price_model.joblib
```

The application accepts:

- Brand
- Year
- Kilometers driven
- Fuel
- Seller type
- Transmission
- Owner
- Mileage
- Engine
- Max power
- Seats

The model predicts log-price, and the application converts it back to the original selling-price scale using `np.exp()`.

## Docker

The application can be run in Docker.

Project app structure:

```text
app/
├── Dockerfile
├── docker-compose.yml
└── code/
    ├── app.py
    ├── requirements.txt
    └── car_price_model.joblib
```

The application uses `scikit-learn==1.5.2`, matching the scikit-learn version used when the saved model was created. Pinning this version helps avoid compatibility problems when loading the `.joblib` model.

## Project Structure

```text
project/
│
├── app/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── code/
│       ├── app.py
│       ├── requirements.txt
│       └── car_price_model.joblib
│
├── data/
├── notebooks/
│   └── HauNguyen_st127260_A1_Predicting_Car_final(5).ipynb
│
└── README.md
```

## How to Run

### Run with Python

Install the dependencies:

```bash
python3 -m pip install -r app/code/requirements.txt
```

Run the application:

```bash
python3 app/code/app.py
```

Then open:

```text
http://localhost:8050
```

### Run with Docker

From the project root:

```bash
docker compose -f app/docker-compose.yml up --build
```

Then open:

```text
http://localhost:8050
```

To stop the application:

```bash
docker compose -f app/docker-compose.yml down
```

## Conclusion

The project demonstrates a complete regression workflow for used-car price prediction.

The dataset was cleaned according to the assignment requirements, missing values were handled inside the preprocessing pipeline, categorical features were one-hot encoded, continuous numerical features were scaled, and the target was log-transformed.

Five regression models were compared using 5-fold cross-validation. Random Forest achieved the lowest CV RMSE and was selected for GridSearchCV tuning.

The tuned Random Forest achieved an R² of approximately 0.946 on the untouched test set. This indicates strong predictive performance, although unusually expensive vehicles remain more difficult to predict accurately.

The final trained model was saved and integrated into a Dash application so that users can enter vehicle information and receive a predicted selling price.
