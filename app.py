import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression



# Config
st.set_page_config(
    page_title="Battery Drain Predictor",
    page_icon="🔋",
    layout="centered"
)

DATA_PATH = "data/user_behavior_dataset.csv"

TARGET_COL = "Battery Drain (mAh/day)"
FEATURE_COLS = [
    "App Usage Time (min/day)",
    "Screen On Time (hours/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)",
    "Age",
    "User Behavior Class",
    "Operating System",
]

NUMERIC_COLS = [
    "App Usage Time (min/day)",
    "Screen On Time (hours/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)",
    "Age",
    "User Behavior Class",
]
CATEGORICAL_COLS = ["Operating System"]


# Model training (cached)
@st.cache_resource
def load_data_and_train_model():
    df = pd.read_csv(DATA_PATH)

    df = df[FEATURE_COLS + [TARGET_COL]].dropna()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ],
        remainder="drop"
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X, y)

    # Useful dataset stats for UI defaults/ranges
    stats = {
        "min": df[NUMERIC_COLS].min(numeric_only=True),
        "max": df[NUMERIC_COLS].max(numeric_only=True),
        "mean": df[NUMERIC_COLS].mean(numeric_only=True),
    }

    return model, df, stats


def battery_life_hours(capacity_mah: float, drain_mah_per_day: float) -> float:
    """
    Convert predicted drain (mAh/day) to estimated hours of battery life,
    assuming drain is uniform across the day.
    hours = capacity / (drain_per_hour) = capacity / (drain_per_day / 24)
    """
    if drain_mah_per_day <= 0:
        return np.inf
    return 24.0 * capacity_mah / drain_mah_per_day


# UI
st.title("🔋 Battery Drain Predictor")
st.caption(
    "Enter your daily usage + device info to predict **Battery Drain (mAh/day)**. "
    "Optionally convert that to **estimated battery life (hours)** using your battery capacity."
)

# Load model/data
try:
    model, df, stats = load_data_and_train_model()
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_PATH}`.\n\n"
        "Make sure your folder looks like:\n"
        "- app.py\n"
        "- data/user_behavior_dataset.csv"
    )
    st.stop()

with st.expander("Dataset preview", expanded=False):
    st.write("First 10 rows of your dataset:")
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# Inputs
st.subheader("Inputs")

col1, col2 = st.columns(2)

with col1:
    app_usage = st.number_input(
        "App Usage Time (min/day)",
        min_value=float(stats["min"]["App Usage Time (min/day)"]),
        max_value=float(stats["max"]["App Usage Time (min/day)"]),
        value=float(stats["mean"]["App Usage Time (min/day)"]),
        step=1.0
    )
    screen_on = st.number_input(
        "Screen On Time (hours/day)",
        min_value=float(stats["min"]["Screen On Time (hours/day)"]),
        max_value=float(stats["max"]["Screen On Time (hours/day)"]),
        value=float(stats["mean"]["Screen On Time (hours/day)"]),
        step=0.1
    )
    num_apps = st.number_input(
        "# Apps Installed",
        min_value=int(stats["min"]["Number of Apps Installed"]),
        max_value=int(stats["max"]["Number of Apps Installed"]),
        value=int(round(stats["mean"]["Number of Apps Installed"])),
        step=1
    )
    data_usage = st.number_input(
        "Data Usage (MB/day)",
        min_value=float(stats["min"]["Data Usage (MB/day)"]),
        max_value=float(stats["max"]["Data Usage (MB/day)"]),
        value=float(stats["mean"]["Data Usage (MB/day)"]),
        step=10.0
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=int(stats["min"]["Age"]),
        max_value=int(stats["max"]["Age"]),
        value=int(round(stats["mean"]["Age"])),
        step=1
    )
    behavior_class = st.selectbox(
        "User Behavior Class",
        options=sorted(df["User Behavior Class"].unique().tolist()),
        index=0
    )
    os_choice = st.radio(
        "Operating System",
        options=["Android", "iOS"],
        horizontal=True
    )

st.divider()

# Optional battery life section
st.subheader("Optional: Battery life estimate")

enable_life = st.checkbox("Also estimate battery life (hours) using battery capacity", value=True)
capacity = None
if enable_life:
    capacity = st.number_input(
        "Battery capacity (mAh)",
        min_value=500.0,
        max_value=20000.0,
        value=4500.0,
        step=100.0,
        help="Typical phones are ~3000–6000 mAh."
    )

st.divider()

# Predict
if st.button("Predict", type="primary"):
    input_df = pd.DataFrame([{
        "App Usage Time (min/day)": app_usage,
        "Screen On Time (hours/day)": screen_on,
        "Number of Apps Installed": num_apps,
        "Data Usage (MB/day)": data_usage,
        "Age": age,
        "User Behavior Class": behavior_class,
        "Operating System": os_choice
    }])

    pred = float(model.predict(input_df)[0])

    st.success(f"**Predicted Battery Drain:** {pred:,.0f} mAh/day")

    if enable_life and capacity is not None:
        hours = battery_life_hours(capacity, pred)
        st.info(f"**Estimated battery life:** {hours:,.1f} hours (given {capacity:,.0f} mAh)")

    with st.expander("Show model inputs used"):
        st.dataframe(input_df, use_container_width=True)

st.caption(
    "Note: This is a simple linear model trained on the provided dataset. "
    "Real-world battery life varies with many factors (battery health, brightness, radios, temperature, etc.)."
)
