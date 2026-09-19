"""
utils.py
Shared utilities for the Household Product Predictive Maintenance Dashboard.

Features:
- Blynk configuration
- Secure Blynk token using Streamlit Secrets
- Live sensor fetching
- Product configuration
- Product age calculation
- Cached ML model loading
- Condition prediction
- Remaining Useful Life (RUL) prediction
- Prediction history save/load
"""

# ============================================================
# IMPORTS
# ============================================================

import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# BLYNK CONFIGURATION
# ============================================================

BASE_URL = "https://blynk.cloud/external/api"


def get_blynk_token():
    """
    Get Blynk token securely.

    Priority:
    1. Streamlit Secrets - recommended for Streamlit Cloud
    2. Environment variable - useful for local development
    """

    # Streamlit Cloud
    try:
        token = st.secrets.get("BLYNK_TOKEN")

        if token:
            return str(token).strip()

    except Exception:
        pass

    # Local development
    token = os.getenv("BLYNK_TOKEN")

    if token:
        return str(token).strip()

    return None


BLYNK_TOKEN = get_blynk_token()


def build_blynk_url(pin):
    """
    Build Blynk GET API URL for a virtual pin.
    """

    if not BLYNK_TOKEN:
        return None

    return f"{BASE_URL}/get?token={BLYNK_TOKEN}&{pin}"


# ============================================================
# BLYNK VIRTUAL PIN CONFIGURATION
# ============================================================

VPIN_URLS = {
    "voltage": build_blynk_url("V0"),
    "product1_current": build_blynk_url("V1"),
    "product2_current": build_blynk_url("V2"),
    "product3_current": build_blynk_url("V3"),
    "temperature": build_blynk_url("V4"),
    "vibration": build_blynk_url("V5"),
}


# ============================================================
# PRODUCT CONFIGURATION
# ============================================================

PRODUCT_CONFIG = {
    "PRODUCT 1": {
        "manufacturing_date": "2025-01-15",
        "installation_date": "2025-02-01",
        "current_key": "product1_current",
        "icon": "⚙️",
    },

    "PRODUCT 2": {
        "manufacturing_date": "2025-03-10",
        "installation_date": "2025-04-05",
        "current_key": "product2_current",
        "icon": "🔩",
    },

    "PRODUCT 3": {
        "manufacturing_date": "2025-06-20",
        "installation_date": "2025-07-01",
        "current_key": "product3_current",
        "icon": "🛠️",
    },
}


# ============================================================
# FILE PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

# Create data folder if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)


# ML MODEL FILES
SCALER_FILE = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)

LABEL_ENCODER_FILE = os.path.join(
    BASE_DIR,
    "label_encoder.pkl"
)

CONDITION_MODEL_FILE = os.path.join(
    BASE_DIR,
    "condition.pkl"
)

LIFE_MODEL_FILE = os.path.join(
    BASE_DIR,
    "life.pkl"
)


# ============================================================
# HISTORY COLUMNS
# ============================================================

HISTORY_COLUMNS = [
    "timestamp",
    "product",
    "voltage",
    "current",
    "temperature",
    "vibration",
    "condition",
    "rul_months",
    "manufacturing_age_years",
    "installation_age_years",
]


# ============================================================
# HISTORY FILE
# ============================================================

def history_file(product_name: str) -> str:
    """
    Return CSV history path for a product.
    """

    safe_name = (
        str(product_name)
        .strip()
        .replace(" ", "_")
        .lower()
    )

    return os.path.join(
        DATA_DIR,
        f"history_{safe_name}.csv"
    )


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load all trained ML models.

    Returns:
        models, errors

    models:
        Dictionary containing scaler, encoder,
        condition model and life model.

    errors:
        List of errors if loading fails.
    """

    model_files = {
        "scaler": SCALER_FILE,
        "label_encoder": LABEL_ENCODER_FILE,
        "condition_model": CONDITION_MODEL_FILE,
        "life_model": LIFE_MODEL_FILE,
    }

    # --------------------------------------------------------
    # CHECK MISSING FILES
    # --------------------------------------------------------

    missing_files = [
        file_path
        for file_path in model_files.values()
        if not os.path.exists(file_path)
    ]

    if missing_files:

        missing_names = [
            os.path.basename(file_path)
            for file_path in missing_files
        ]

        return None, [
            "Missing model files: "
            + ", ".join(missing_names)
        ]

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    try:

        scaler = joblib.load(
            model_files["scaler"]
        )

        label_encoder = joblib.load(
            model_files["label_encoder"]
        )

        condition_model = joblib.load(
            model_files["condition_model"]
        )

        life_model = joblib.load(
            model_files["life_model"]
        )

        models = {
            "scaler": scaler,
            "label_encoder": label_encoder,
            "condition_model": condition_model,
            "life_model": life_model,
        }

        return models, []

    except Exception as e:

        return None, [
            f"Model loading error: {str(e)}"
        ]


# ============================================================
# BLYNK SENSOR FETCH
# ============================================================

def fetch_sensor(url: str, timeout: int = 5):
    """
    Fetch one Blynk virtual-pin value.

    Returns:
        value, error_message
    """

    # --------------------------------------------------------
    # CHECK TOKEN
    # --------------------------------------------------------

    if not BLYNK_TOKEN:
        return (
            None,
            "BLYNK_TOKEN is not configured. "
            "Add BLYNK_TOKEN in Streamlit Cloud Secrets."
        )

    # --------------------------------------------------------
    # CHECK URL
    # --------------------------------------------------------

    if not url:
        return (
            None,
            "Blynk URL is not configured."
        )

    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    try:

        response = requests.get(
            url,
            timeout=timeout
        )

        # HTTP ERROR
        if response.status_code != 200:

            return (
                None,
                f"HTTP {response.status_code}: "
                f"{response.text}"
            )

        # RESPONSE TEXT
        raw_value = response.text.strip()

        if raw_value == "":
            return (
                None,
                "Empty value returned from Blynk."
            )

        # CONVERT TO FLOAT
        try:

            value = float(raw_value)

        except ValueError:

            return (
                None,
                f"Non-numeric value received: "
                f"'{raw_value}'"
            )

        return value, None

    except requests.exceptions.Timeout:

        return (
            None,
            "Blynk request timed out."
        )

    except requests.exceptions.ConnectionError:

        return (
            None,
            "Blynk connection error. "
            "Check internet connection and device status."
        )

    except requests.exceptions.RequestException as e:

        return (
            None,
            f"Blynk request error: {str(e)}"
        )

    except Exception as e:

        return (
            None,
            f"Unexpected Blynk error: {str(e)}"
        )


# ============================================================
# FETCH COMMON SENSOR VALUES
# ============================================================

def fetch_common_values():
    """
    Fetch common sensors:

    V0 -> Voltage
    V4 -> Temperature
    V5 -> Vibration

    Returns dictionary containing values and errors.
    """

    voltage, voltage_error = fetch_sensor(
        VPIN_URLS["voltage"]
    )

    temperature, temperature_error = fetch_sensor(
        VPIN_URLS["temperature"]
    )

    vibration, vibration_error = fetch_sensor(
        VPIN_URLS["vibration"]
    )

    return {
        "voltage": voltage,
        "temperature": temperature,
        "vibration": vibration,

        "errors": {
            "voltage": voltage_error,
            "temperature": temperature_error,
            "vibration": vibration_error,
        },
    }


# ============================================================
# FETCH PRODUCT CURRENT
# ============================================================

def fetch_product_current(product_name: str):
    """
    Fetch current for selected product.

    PRODUCT 1 -> V1
    PRODUCT 2 -> V2
    PRODUCT 3 -> V3
    """

    if product_name not in PRODUCT_CONFIG:

        return (
            None,
            f"Unknown product: {product_name}"
        )

    current_key = PRODUCT_CONFIG[
        product_name
    ]["current_key"]

    url = VPIN_URLS.get(current_key)

    return fetch_sensor(url)


# ============================================================
# PRODUCT AGE CALCULATION
# ============================================================

def calculate_product_age(
    manufacturing_date: str,
    installation_date: str
):
    """
    Calculate product age from manufacturing
    and installation dates.
    """

    try:

        today = datetime.today()

        manufacture_dt = datetime.strptime(
            manufacturing_date,
            "%Y-%m-%d"
        )

        installation_dt = datetime.strptime(
            installation_date,
            "%Y-%m-%d"
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if installation_dt < manufacture_dt:

            raise ValueError(
                "Installation date cannot be "
                "before manufacturing date."
            )

        # ----------------------------------------------------
        # AGE
        # ----------------------------------------------------

        manufacturing_age_days = (
            today - manufacture_dt
        ).days

        installation_age_days = (
            today - installation_dt
        ).days

        return {
            "manufacturing_age_days":
                max(0, manufacturing_age_days),

            "installation_age_days":
                max(0, installation_age_days),

            "manufacturing_age_years":
                max(0, manufacturing_age_days) / 365.25,

            "installation_age_years":
                max(0, installation_age_days) / 365.25,
        }

    except Exception:

        return {
            "manufacturing_age_days": 0,
            "installation_age_days": 0,
            "manufacturing_age_years": 0,
            "installation_age_years": 0,
        }


# ============================================================
# PREDICTION
# ============================================================

def predict_product(
    models,
    product_name,
    voltage,
    load_current,
    temperature,
    vibration
):
    """
    Run:

    Sensor values
        ↓
    Scaling
        ↓
    Condition Model
        ↓
    Condition Label
        ↓
    Life/RUL Model
        ↓
    Prediction Result

    Returns:
        result, error
    """

    # --------------------------------------------------------
    # VALIDATE MODEL
    # --------------------------------------------------------

    if models is None:

        return (
            None,
            "ML models are not loaded."
        )

    # --------------------------------------------------------
    # VALIDATE PRODUCT
    # --------------------------------------------------------

    if product_name not in PRODUCT_CONFIG:

        return (
            None,
            f"Invalid product: {product_name}"
        )

    # --------------------------------------------------------
    # VALIDATE INPUTS
    # --------------------------------------------------------

    try:

        voltage = float(voltage)
        load_current = float(load_current)
        temperature = float(temperature)
        vibration = float(vibration)

    except (TypeError, ValueError):

        return (
            None,
            "Invalid sensor input. "
            "All values must be numeric."
        )

    # --------------------------------------------------------
    # PRODUCT INFORMATION
    # --------------------------------------------------------

    product_info = PRODUCT_CONFIG[
        product_name
    ]

    age_info = calculate_product_age(
        product_info["manufacturing_date"],
        product_info["installation_date"]
    )

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    input_data = np.array(
        [[
            voltage,
            load_current,
            temperature,
            vibration
        ]],
        dtype=float
    )

    # --------------------------------------------------------
    # SCALE INPUT
    # --------------------------------------------------------

    try:

        scaled_data = models[
            "scaler"
        ].transform(input_data)

    except Exception as e:

        return (
            None,
            f"Scaling error: {str(e)}"
        )

    # --------------------------------------------------------
    # CONDITION PREDICTION
    # --------------------------------------------------------

    try:

        condition_encoded = models[
            "condition_model"
        ].predict(scaled_data)

        condition = models[
            "label_encoder"
        ].inverse_transform(
            condition_encoded
        )[0]

        condition = str(condition)

    except Exception as e:

        return (
            None,
            f"Condition prediction error: {str(e)}"
        )

    # --------------------------------------------------------
    # RUL / LIFE PREDICTION
    # --------------------------------------------------------

    try:

        rul_prediction = models[
            "life_model"
        ].predict(scaled_data)

        rul = float(
            rul_prediction[0]
        )

        # RUL cannot be negative
        rul = max(0.0, rul)

    except Exception as e:

        return (
            None,
            f"RUL prediction error: {str(e)}"
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {

        "product":
            product_name,

        "manufacturing_date":
            product_info[
                "manufacturing_date"
            ],

        "installation_date":
            product_info[
                "installation_date"
            ],

        "manufacturing_age_years":
            age_info[
                "manufacturing_age_years"
            ],

        "installation_age_years":
            age_info[
                "installation_age_years"
            ],

        "voltage":
            voltage,

        "current":
            load_current,

        "temperature":
            temperature,

        "vibration":
            vibration,

        "condition":
            condition,

        "rul":
            rul,

        "timestamp":
            datetime.now(),
    }

    return result, None


# ============================================================
# CONDITION COLOR
# ============================================================

def condition_color(condition: str) -> str:
    """
    Return display color based on condition.
    """

    condition_text = str(
        condition
    ).lower()

    # GOOD
    if any(
        word in condition_text
        for word in [
            "good",
            "normal",
            "healthy",
            "excellent",
            "ok"
        ]
    ):

        return "#16a34a"

    # WARNING
    if any(
        word in condition_text
        for word in [
            "warn",
            "moderate",
            "fair",
            "degrad"
        ]
    ):

        return "#f59e0b"

    # CRITICAL
    if any(
        word in condition_text
        for word in [
            "bad",
            "critical",
            "fault",
            "fail",
            "poor"
        ]
    ):

        return "#dc2626"

    # DEFAULT
    return "#2563eb"


# ============================================================
# HISTORY - SAVE
# ============================================================

def append_history(result: dict):
    """
    Append prediction result to product CSV history.
    """

    # --------------------------------------------------------
    # VALIDATE RESULT
    # --------------------------------------------------------

    if not result:
        return False

    product_name = result.get(
        "product"
    )

    if not product_name:
        return False

    # --------------------------------------------------------
    # FILE PATH
    # --------------------------------------------------------

    path = history_file(
        product_name
    )

    # --------------------------------------------------------
    # ROW
    # --------------------------------------------------------

    row = {

        "timestamp":
            result[
                "timestamp"
            ].strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "product":
            result[
                "product"
            ],

        "voltage":
            round(
                float(result["voltage"]),
                3
            ),

        "current":
            round(
                float(result["current"]),
                3
            ),

        "temperature":
            round(
                float(result["temperature"]),
                3
            ),

        "vibration":
            round(
                float(result["vibration"]),
                4
            ),

        "condition":
            result[
                "condition"
            ],

        "rul_months":
            round(
                float(result["rul"]),
                2
            ),

        "manufacturing_age_years":
            round(
                float(
                    result[
                        "manufacturing_age_years"
                    ]
                ),
                2
            ),

        "installation_age_years":
            round(
                float(
                    result[
                        "installation_age_years"
                    ]
                ),
                2
            ),
    }

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df_row = pd.DataFrame(
        [row],
        columns=HISTORY_COLUMNS
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    try:

        if os.path.exists(path):

            df_row.to_csv(
                path,
                mode="a",
                header=False,
                index=False
            )

        else:

            df_row.to_csv(
                path,
                mode="w",
                header=True,
                index=False
            )

        return True

    except Exception:

        return False


# ============================================================
# HISTORY - LOAD
# ============================================================

def load_history(
    product_name: str
) -> pd.DataFrame:
    """
    Load prediction history for selected product.
    """

    path = history_file(
        product_name
    )

    # --------------------------------------------------------
    # FILE DOES NOT EXIST
    # --------------------------------------------------------

    if not os.path.exists(path):

        return pd.DataFrame(
            columns=HISTORY_COLUMNS
        )

    # --------------------------------------------------------
    # READ FILE
    # --------------------------------------------------------

    try:

        df = pd.read_csv(path)

        # Make sure all expected columns exist
        for column in HISTORY_COLUMNS:

            if column not in df.columns:

                df[column] = None

        # Keep expected column order
        df = df[
            HISTORY_COLUMNS
        ]

        return df

    except Exception:

        return pd.DataFrame(
            columns=HISTORY_COLUMNS
        )


# ============================================================
# HISTORY - CLEAR
# ============================================================

def clear_history(
    product_name: str
):
    """
    Delete prediction history CSV
    for selected product.
    """

    path = history_file(
        product_name
    )

    try:

        if os.path.exists(path):

            os.remove(path)

        return True

    except Exception:

        return False


# ============================================================
# CHECK BLYNK CONNECTION
# ============================================================

def blynk_configured():
    """
    Check whether Blynk token is configured.
    """

    return bool(BLYNK_TOKEN)


# ============================================================
# GET AVAILABLE PRODUCTS
# ============================================================

def get_products():
    """
    Return available product names.
    """

    return list(
        PRODUCT_CONFIG.keys()
    )
