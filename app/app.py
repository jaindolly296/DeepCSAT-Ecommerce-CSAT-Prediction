
from pathlib import Path
from datetime import datetime
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="DeepCSAT",
    page_icon="⭐",
    layout="wide"
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRECTORY = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIRECTORY / "deepcsat_final_model.keras"
PREPROCESSOR_PATH = MODEL_DIRECTORY / "deepcsat_preprocessor.joblib"
FREQUENCY_MAPS_PATH = MODEL_DIRECTORY / "deepcsat_frequency_maps.joblib"
METADATA_PATH = MODEL_DIRECTORY / "deepcsat_metadata.json"


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    frequency_maps = joblib.load(FREQUENCY_MAPS_PATH)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as metadata_file:
        metadata = json.load(metadata_file)

    return model, preprocessor, frequency_maps, metadata


def get_category_options(preprocessor, feature_names):
    categorical_pipeline = (
        preprocessor.named_transformers_["categorical"]
    )

    onehot_encoder = (
        categorical_pipeline.named_steps["onehot"]
    )

    return {
        feature: [
            str(value)
            for value in categories
        ]
        for feature, categories in zip(
            feature_names,
            onehot_encoder.categories_
        )
    }


def apply_frequency_encoding(
    data,
    frequency_maps,
    high_cardinality_features
):
    encoded_data = data.copy()

    for column, frequency_map in frequency_maps.items():
        prepared_values = (
            encoded_data[column]
            .fillna("__MISSING__")
        )

        encoded_data[f"{column}_frequency"] = (
            prepared_values
            .map(frequency_map)
            .fillna(0.0)
            .astype(float)
        )

    return encoded_data.drop(
        columns=high_cardinality_features
    )


def prepare_input(
    raw_input,
    preprocessor,
    frequency_maps,
    metadata
):
    input_data = pd.DataFrame([raw_input])

    encoded_data = apply_frequency_encoding(
        input_data,
        frequency_maps,
        metadata["high_cardinality_features"]
    )

    for column in metadata["low_cardinality_features"]:
        encoded_data[column] = (
            encoded_data[column]
            .astype(object)
            .where(
                encoded_data[column].notna(),
                np.nan
            )
        )

    processed_data = preprocessor.transform(
        encoded_data
    )

    return processed_data.astype(np.float32)


try:
    model, preprocessor, frequency_maps, metadata = (
        load_artifacts()
    )
except Exception as error:
    st.error(
        "The saved model artifacts could not be loaded. "
        "Confirm that the models folder is beside the app folder."
    )
    st.exception(error)
    st.stop()


category_options = get_category_options(
    preprocessor,
    metadata["low_cardinality_features"]
)


st.title(
    "DeepCSAT: E-Commerce Customer Satisfaction Score Prediction"
)

st.write(
    "Enter the available customer-support interaction details. "
    "The application will estimate the probability of each CSAT score."
)

st.caption(
    "Prediction point: after the support response and before the "
    "customer submits the satisfaction survey."
)


with st.form("deepcsat_prediction_form"):
    left_column, right_column = st.columns(2)

    with left_column:
        channel_name = st.selectbox(
            "Support channel",
            category_options["channel_name"]
        )

        category = st.selectbox(
            "Interaction category",
            category_options["category"]
        )

        sub_category = st.selectbox(
            "Interaction sub-category",
            category_options["sub_category"]
        )

        tenure_bucket = st.selectbox(
            "Agent tenure bucket",
            category_options["tenure_bucket"]
        )

        agent_shift = st.selectbox(
            "Agent shift",
            category_options["agent_shift"]
        )

        supervisor = st.selectbox(
            "Supervisor",
            category_options["supervisor"]
        )

        manager = st.selectbox(
            "Manager",
            category_options["manager"]
        )

    with right_column:
        product_options = (
            ["<Missing>"]
            + category_options["product_category"]
        )

        selected_product_category = st.selectbox(
            "Product category",
            product_options
        )

        customer_city = st.text_input(
            "Customer city",
            placeholder="Leave blank if unavailable"
        )

        agent_name = st.text_input(
            "Agent name",
            placeholder="Enter the agent name"
        )

        issue_date = st.date_input(
            "Issue-reported date"
        )

        issue_time = st.time_input(
            "Issue-reported time"
        )

        response_time_minutes = st.number_input(
            "Response time in minutes",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

    st.subheader("Optional order information")

    option_column_1, option_column_2 = st.columns(2)

    with option_column_1:
        order_id_available = st.checkbox(
            "Order ID is available",
            value=True
        )

        order_date_available = st.checkbox(
            "Order date is available",
            value=False
        )

        remarks_available = st.checkbox(
            "Customer remarks are available",
            value=False
        )

    with option_column_2:
        item_price_available = st.checkbox(
            "Item price is available",
            value=False
        )

        if item_price_available:
            item_price = st.number_input(
                "Item price",
                min_value=0.0,
                value=1000.0,
                step=100.0
            )
        else:
            item_price = np.nan

        if order_date_available:
            order_to_issue_hours = st.number_input(
                "Hours between order and issue",
                min_value=0.0,
                value=24.0,
                step=1.0
            )
        else:
            order_to_issue_hours = np.nan

    submit_prediction = st.form_submit_button(
        "Predict CSAT Score",
        use_container_width=True
    )


if submit_prediction:
    issue_timestamp = datetime.combine(
        issue_date,
        issue_time
    )

    product_category = (
        np.nan
        if selected_product_category == "<Missing>"
        else selected_product_category
    )

    cleaned_city = (
        customer_city.strip()
        if customer_city.strip()
        else np.nan
    )

    cleaned_agent = (
        agent_name.strip()
        if agent_name.strip()
        else np.nan
    )

    raw_model_input = {
        "log_item_price": (
            np.log1p(item_price)
            if item_price_available
            else np.nan
        ),
        "log_response_time_minutes": np.log1p(
            response_time_minutes
        ),
        "log_order_to_issue_hours": (
            np.log1p(order_to_issue_hours)
            if order_date_available
            else np.nan
        ),
        "issue_hour": issue_timestamp.hour,
        "issue_day_number": issue_timestamp.weekday(),
        "is_weekend": int(
            issue_timestamp.weekday() in [5, 6]
        ),
        "invalid_response_time": 0,
        "invalid_order_to_issue_time": 0,
        "is_order_id_missing": int(
            not order_id_available
        ),
        "is_order_date_time_missing": int(
            not order_date_available
        ),
        "is_customer_remarks_missing": int(
            not remarks_available
        ),
        "is_customer_city_missing": int(
            pd.isna(cleaned_city)
        ),
        "is_product_category_missing": int(
            pd.isna(product_category)
        ),
        "is_item_price_missing": int(
            not item_price_available
        ),
        "channel_name": channel_name,
        "category": category,
        "sub_category": sub_category,
        "product_category": product_category,
        "supervisor": supervisor,
        "manager": manager,
        "tenure_bucket": tenure_bucket,
        "agent_shift": agent_shift,
        "customer_city": cleaned_city,
        "agent_name": cleaned_agent
    }

    processed_input = prepare_input(
        raw_model_input,
        preprocessor,
        frequency_maps,
        metadata
    )

    probabilities = model.predict(
        processed_input,
        verbose=0
    )[0]

    predicted_csat = int(
        np.argmax(probabilities) + 1
    )

    low_csat_probability = float(
        probabilities[0] + probabilities[1]
    )

    result_column_1, result_column_2 = st.columns(2)

    with result_column_1:
        st.metric(
            "Predicted CSAT Score",
            f"{predicted_csat} / 5"
        )

    with result_column_2:
        st.metric(
            "Probability of Low CSAT (1 or 2)",
            f"{low_csat_probability:.1%}"
        )

    probability_table = pd.DataFrame({
        "CSAT Score": [1, 2, 3, 4, 5],
        "Model Probability": probabilities
    }).set_index("CSAT Score")

    st.subheader("Predicted Score Probabilities")
    st.bar_chart(probability_table)

    st.dataframe(
        probability_table.style.format({
            "Model Probability": "{:.2%}"
        }),
        use_container_width=True
    )

    if low_csat_probability >= 0.50:
        st.warning(
            "High predicted risk of low satisfaction. "
            "Consider proactive review or follow-up."
        )
    elif low_csat_probability >= 0.25:
        st.info(
            "Moderate predicted risk of low satisfaction. "
            "Monitor the interaction outcome."
        )
    else:
        st.success(
            "The model estimates a relatively low risk "
            "of CSAT scores 1 or 2."
        )

    st.caption(
        "Model probabilities are predictive outputs and should not be "
        "interpreted as verified confidence unless probability calibration "
        "has been evaluated."
    )

