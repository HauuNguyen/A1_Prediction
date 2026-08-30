# Car Price Prediction

## Project Overview

This project develops a machine-learning regression model to predict the selling price of used cars.

The project follows a complete machine-learning workflow:

- Data loading and inspection
- Exploratory data analysis (EDA)
- Data cleaning
- Feature engineering
- Train/test splitting
- Target log transformation
- Preprocessing pipelines
- Model comparison using 5-fold cross-validation
- Model selection
- Hyperparameter tuning with GridSearchCV
- Final evaluation on the untouched test set
- Feature-importance analysis
- Model saving with Joblib
- Dash prediction application
- Docker support
- Public deployment

The target variable is `selling_price`.

---

## Dataset

The raw Car Price dataset contains 8,128 rows and 13 columns:

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

Missing values occur in several features, including:

- `mileage`
- `engine`
- `max_power`
- `torque`
- `seats`

The dataset was cleaned according to the requirements of the assignment.

---

# Data Cleaning and Feature Engineering

## Remove CNG and LPG

CNG and LPG vehicles were removed from the dataset.

The reason is that CNG and LPG mileage is measured in `km/kg`, while Diesel and Petrol mileage is measured in `kmpl` (km/l).

Combining these different measurement systems would make the `mileage` feature inconsistent and could negatively affect the model.

---

## Owner Mapping

The `owner` feature was converted from categorical text into numerical values.

The mapping was:

```text
First Owner          -> 1
Second Owner         -> 2
Third Owner          -> 3
Fourth & Above Owner -> 4
Test Drive Car       -> 5
```

After applying the mapping, all samples belonging to `Test Drive Car` were removed.

Test Drive Cars were excluded because their selling prices were extremely high compared with normal used cars. Including them could distort the model's predictions for ordinary used vehicles.

---

## Create `brand`

The original `name` feature contains detailed vehicle names with many unique values.

Instead of using the complete name, the first word was extracted as the `brand`.

Examples:

```text
Maruti Swift Dzire VDI       -> Maruti
Skoda Rapid 1.5 TDI Ambition -> Skoda
Honda City 2017-2020 EXi     -> Honda
Hyundai i20 Sportz Diesel    -> Hyundai
```

The original `name` column was then removed.

Using `brand` instead of the complete vehicle name reduces the number of categorical levels while retaining useful information about the manufacturer.

---

## Remove `torque`

The `torque` feature was removed as required by the assignment.

The feature contains textual values in different formats, which would require additional parsing and interpretation. Since the assignment explicitly asks to drop this feature, it was excluded from the final model.

---

## Clean Numerical Features

### Mileage

The `mileage` feature originally contains values such as:

```text
23.4 kmpl
19.7 kmpl
17.0 kmpl
```

The `kmpl` unit was removed and the values were converted to numerical values.

For example:

```text
23.4 kmpl -> 23.4
```

### Engine

The `engine` feature originally contains values such as:

```text
1248 CC
1498 CC
1197 CC
```

The `CC` unit was removed and the values were converted to numerical values.

For example:

```text
1248 CC -> 1248
```

### Max Power

The `max_power` feature originally contains values with the `bhp` unit.

The unit was removed and the feature was converted to numerical values.

---

# Exploratory Data Analysis

EDA was performed before model training to understand the dataset, feature distributions, missing values, categorical variables, and relationships between features and the target.

Several plots were used to investigate the data.

## Selling Price Distribution

The original `selling_price` distribution is strongly right-skewed.

Most cars have relatively low selling prices, while a smaller number of vehicles have substantially higher prices.

This large range and skewness motivated the use of a logarithmic transformation of the target.

## Feature Relationships

The numerical features were examined using descriptive statistics, distributions, and correlation analysis.

Important relationships were observed between selling price and vehicle characteristics such as:

- `year`
- `max_power`
- `engine`
- `km_driven`
- `mileage`

In general, newer vehicles and vehicles with stronger performance-related characteristics tend to have higher selling prices.

However, correlation does not imply causation, and the relationships between features and price are not necessarily linear. This was one reason why nonlinear models such as Random Forest and Gradient Boosting were also evaluated.

---

# Train/Test Split

The target was separated from the features:

```python
X = df.drop(columns=["selling_price"])
y = df["selling_price"]
```

The dataset was split into training and test sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The test set was kept completely separate from model selection and hyperparameter tuning.

This is important because the test set should provide an unbiased estimate of how the final model performs on unseen data.

---

# Target Log Transformation

The selling prices have a large range and a strong right-skewed distribution.

Therefore, the training target was log-transformed:

```python
y_train_log = np.log(y_train)
```

The models were trained to predict:

```text
log(selling_price)
```

rather than the original selling price.

This transformation reduces the effect of extremely large target values and makes the target distribution more suitable for regression.

During inference, the predicted log-price was transformed back to the original price scale:

```python
predicted_price = np.exp(predicted_log_price)
```

The final evaluation metrics were calculated on the original selling-price scale.

---

# Preprocessing

Preprocessing was implemented using a `ColumnTransformer` and included inside the machine-learning pipeline.

This prevents preprocessing from being fitted using information from the validation or test data.

## Continuous Numerical Features

The continuous numerical features are:

- `km_driven`
- `mileage`
- `engine`
- `max_power`

Missing values were handled using median imputation.

The features were then standardized using:

```text
StandardScaler
```

## Other Numerical Features

The following features were handled separately:

- `year`
- `owner`
- `seats`

Missing values were handled using the most frequent value.

## Categorical Features

The categorical features are:

- `brand`
- `fuel`
- `seller_type`
- `transmission`

These features were one-hot encoded using:

```python
OneHotEncoder(handle_unknown="ignore")
```

`handle_unknown="ignore"` allows the model to make predictions even if a category appears during inference that was not present in the training data.

---

# Model Comparison

Five regression models were compared using 5-fold cross-validation:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)

The comparison metric was RMSE calculated on the log-transformed target.

| Model | CV RMSE |
|---|---:|
| Random Forest | 0.213107 |
| Gradient Boosting | 0.226682 |
| Linear Regression | 0.258175 |
| Ridge | 0.258240 |
| SVR | 0.818603 |

Lower RMSE indicates better performance.

Random Forest achieved the lowest cross-validation RMSE:

```text
Random Forest: 0.213107
```

Therefore, Random Forest was selected for further hyperparameter tuning.

Random Forest performed well because it can capture nonlinear relationships and interactions between vehicle characteristics without assuming that the relationship between each feature and price is linear.

---

# Hyperparameter Tuning

GridSearchCV was applied to the Random Forest model.

The following parameters were searched:

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

---

# Final Model

The final model is a complete preprocessing and regression pipeline.

The pipeline includes:

```text
Input Data
    ↓
Missing-value handling
    ↓
Scaling numerical features
    ↓
One-hot encoding categorical features
    ↓
Random Forest Regressor
```

The final Random Forest parameters are:

```text
n_estimators = 200
max_depth = 20
min_samples_split = 5
min_samples_leaf = 1
random_state = 42
```

The complete trained pipeline was saved using Joblib.

---

# Final Test Evaluation

The final model was evaluated using the untouched test set.

Predictions were converted from log-price back to the original selling-price scale using:

```python
predicted_price = np.exp(predicted_log_price)
```

The final results were:

| Metric | Result |
|---|---:|
| MAE | 69,043.28 |
| RMSE | 204,713.91 |
| R² | 0.94587 |

## Interpretation

The R² score of approximately `0.946` means that the model explains about 94.6% of the variation in selling prices in the test set.

The MAE is approximately `69,043` price units. This represents the average absolute difference between the predicted and actual selling prices.

The RMSE is approximately `204,714` price units.

RMSE is higher than MAE because RMSE gives greater weight to large prediction errors. Therefore, some cars have substantially larger prediction errors than the typical prediction.

Overall, the model performs strongly on the test set, although very expensive vehicles can still be more difficult to predict accurately.

---

# Feature Importance

Feature importance was analysed from the final Random Forest model after preprocessing.

The most important features included:

1. `year`
2. `max_power`
3. `engine`

Other relevant features included:

- `km_driven`
- `mileage`

This suggests that vehicle age and performance-related characteristics have a strong relationship with used-car selling prices in this dataset.

For example, newer vehicles generally have higher selling prices, while engine and power characteristics provide additional information about the vehicle's market value.

Feature importance indicates which variables are useful for prediction, but it should not be interpreted as proof of a causal relationship.

---

# Dash Prediction Application

A Dash web application was developed to allow users to enter vehicle information and obtain a predicted selling price.

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

The saved model is:

```text
app/code/car_price_model.joblib
```

The application loads the complete trained pipeline and passes the user input directly to the model.

The model predicts the log-transformed selling price.

The application then converts the prediction back to the original price scale:

```python
predicted_price = np.exp(predicted_log_price[0])
```

The predicted selling price is then displayed to the user.

---

# Public Application

The Dash application has been deployed as a public web service.

**Public URL:**

https://st127260-a1-prediction.onrender.com

Users can open the URL in a web browser, enter the vehicle information, and receive a predicted selling price.

The application is deployed using Docker and hosted on Render.

> Note: The application uses Render's free hosting tier, so the service may temporarily sleep after a period of inactivity. The first request after sleeping may take some time while the service starts again.

---

# Docker

The Dash application is containerized using Docker.

The application structure is:

```text
app/
├── Dockerfile
├── docker-compose.yml
└── code/
    ├── app.py
    ├── requirements.txt
    └── car_price_model.joblib
```

The Docker image uses:

```dockerfile
FROM python:3.12-slim
```

The application listens on `0.0.0.0` and uses the `PORT` environment variable when provided by the hosting platform.

The `scikit-learn` version is pinned to:

```text
scikit-learn==1.5.2
```

This matches the version used when the saved model was created and helps reduce compatibility problems when loading the `.joblib` file.

---

# Project Structure

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
├── notebooks/
│   └── HauNguyen_st127260_A1_Predicting_Car_final.ipynb
│
└── README.md
```

---

# How to Run Locally

## Run with Python

Install the required dependencies:

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

---

## Run with Docker

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

---

# Conclusion

This project demonstrates a complete machine-learning workflow for used-car price prediction.

The dataset was cleaned according to the assignment requirements, including removing CNG and LPG vehicles, handling the owner feature, removing Test Drive Cars, extracting the vehicle brand, cleaning numerical features, and dropping the torque feature.

The target variable was log-transformed because the original selling-price distribution was highly right-skewed.

The data was split into training and test sets before model fitting. Preprocessing was implemented inside machine-learning pipelines to prevent data leakage during cross-validation and model training.

Five regression algorithms were compared using 5-fold cross-validation. Random Forest achieved the lowest CV RMSE and was therefore selected for hyperparameter tuning.

The tuned Random Forest achieved an R² of approximately `0.946` on the untouched test set.

Finally, the trained pipeline was saved and integrated into a Dash application. The application was containerized using Docker and deployed publicly so users can enter vehicle information and receive a predicted selling price.
