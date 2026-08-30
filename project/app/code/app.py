import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from dash import Dash, html, dcc, Input, Output, State


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "car_price_model.joblib"

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Get categories from the trained OneHotEncoder
# --------------------------------------------------

encoder = (
    model
    .named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
)

brands = encoder.categories_[0].tolist()
fuels = encoder.categories_[1].tolist()
seller_types = encoder.categories_[2].tolist()
transmissions = encoder.categories_[3].tolist()


# --------------------------------------------------
# Create Dash app
# --------------------------------------------------

app = Dash(__name__)

app.title = "Car Price Prediction"


# --------------------------------------------------
# Layout
# --------------------------------------------------

app.layout = html.Div(
    [
        html.H1("Car Price Prediction"),

        html.P(
            "Enter the car information below to predict its selling price."
        ),

        html.Br(),

        html.Label("Brand"),
        dcc.Dropdown(
            id="brand",
            options=[
                {"label": brand, "value": brand}
                for brand in brands
            ],
            placeholder="Select brand",
        ),

        html.Br(),

        html.Label("Year"),
        dcc.Input(
            id="year",
            type="number",
            placeholder="e.g. 2018",
        ),

        html.Br(),
        html.Br(),

        html.Label("Kilometers Driven"),
        dcc.Input(
            id="km_driven",
            type="number",
            placeholder="e.g. 50000",
        ),

        html.Br(),
        html.Br(),

        html.Label("Fuel"),
        dcc.Dropdown(
            id="fuel",
            options=[
                {"label": fuel, "value": fuel}
                for fuel in fuels
            ],
            placeholder="Select fuel",
        ),

        html.Br(),

        html.Label("Seller Type"),
        dcc.Dropdown(
            id="seller_type",
            options=[
                {"label": seller_type, "value": seller_type}
                for seller_type in seller_types
            ],
            placeholder="Select seller type",
        ),

        html.Br(),

        html.Label("Transmission"),
        dcc.Dropdown(
            id="transmission",
            options=[
                {
                    "label": transmission,
                    "value": transmission,
                }
                for transmission in transmissions
            ],
            placeholder="Select transmission",
        ),

        html.Br(),

        html.Label("Owner"),
        dcc.Input(
            id="owner",
            type="number",
            placeholder="e.g. 1",
        ),

        html.Br(),
        html.Br(),

        html.Label("Mileage"),
        dcc.Input(
            id="mileage",
            type="number",
            placeholder="e.g. 20.5",
        ),

        html.Br(),
        html.Br(),

        html.Label("Engine"),
        dcc.Input(
            id="engine",
            type="number",
            placeholder="e.g. 1498",
        ),

        html.Br(),
        html.Br(),

        html.Label("Max Power"),
        dcc.Input(
            id="max_power",
            type="number",
            placeholder="e.g. 100",
        ),

        html.Br(),
        html.Br(),

        html.Label("Seats"),
        dcc.Input(
            id="seats",
            type="number",
            placeholder="e.g. 5",
        ),

        html.Br(),
        html.Br(),

        html.Button(
            "Predict Price",
            id="predict-button",
            n_clicks=0,
        ),

        html.Br(),
        html.Br(),

        html.H2(id="prediction-output"),
    ],
    style={
        "maxWidth": "700px",
        "margin": "40px auto",
        "padding": "20px",
    },
)


# --------------------------------------------------
# Prediction callback
# --------------------------------------------------

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),

    State("brand", "value"),
    State("year", "value"),
    State("km_driven", "value"),
    State("fuel", "value"),
    State("seller_type", "value"),
    State("transmission", "value"),
    State("owner", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("seats", "value"),
)
def predict_price(
    n_clicks,
    brand,
    year,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner,
    mileage,
    engine,
    max_power,
    seats,
):

    if n_clicks == 0:
        return "Enter the car information and click Predict Price."

    # Create input DataFrame with exactly the same
    # feature names used when training the model.
    input_data = pd.DataFrame(
        [
            {
                "brand": brand,
                "year": year,
                "km_driven": km_driven,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
                "mileage": mileage,
                "engine": engine,
                "max_power": max_power,
                "seats": seats,
            }
        ]
    )

    # Model predicts log(price)
    predicted_log_price = model.predict(input_data)

    # Convert log(price) back to the original selling price
    predicted_price = np.exp(predicted_log_price[0])

    return f"Predicted Selling Price: {predicted_price:,.0f}"


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )