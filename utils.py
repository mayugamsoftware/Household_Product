"""
utils.py
Household Product Predictive Maintenance Dashboard

Features:
- Secure Blynk token using Streamlit Secrets
- Live Blynk sensor fetching
- Product configuration
- Product age calculation
- Cached ML model loading
- Condition prediction
- RUL prediction
- Prediction history save/load
"""

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
    Get Blynk token.

    Priority:
    1. Streamlit Secrets
    2. Environment variable

    Returns:
        str | None
    """

    # --------------------------------------------------------
    # STREAMLIT CLOUD SECRETS
    # --------------------------------------------------------

    try:
        if "BLYNK_TOKEN" in st.secrets:

            token = st.secrets["BLYNK_TOKEN"]

            if token is not None:
                token = str(token).strip()

                if token:
                    return token

    except Exception:
        pass

    # --------------------------------------------------------
    # LOCAL ENVIRONMENT VARIABLE
    # --------------------------------------------------------

    try:
        token = os.getenv("BLYNK_TOKEN")

        if token:
            token = str(token).strip()

            if token:
                return token

    except Exception:
        pass

    return None


def get_blynk_status():
    """
    Return Blynk configuration status.
    """

    token = get_blynk_token()

    if token:
        return True, "BLYNK_TOKEN configured successfully."

    return (
        False,
        "BLYNK_TOKEN is not configured. "
        "Add BLYNK_TOKEN in Streamlit Cloud Secrets."
    )


# ============================================================
# BLYNK URL
# ============================================================

def build_blynk_url(pin):
    """
    Build Blynk API URL dynamically.

    Example:
        V0
        V1
        V2
    """

    token = get_blynk_token()

    if not token:
        return None

    return (
        f"{BASE_URL}/get"
        f"?token={token}"
        f"&{pin}"
    )


# ============================================================
# BLYNK VIRTUAL PIN CONFIGURATION
# ============================================================

VPINS = {
    "voltage": "V0",
    "product1_current": "V1",
    "product2_current": "V2",
    "product3_current": "V3",
    "temperature": "V4",
    "vibration": "V5",
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

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ============================================================
# ML MODEL FILES
# ============================================================

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

    model_files = {

        "scaler":
            SCALER_FILE,

        "label_encoder":
            LABEL_ENCODER_FILE,

        "condition_model":
            CONDITION_MODEL_FILE,

        "life_model":
            LIFE_MODEL_FILE,
    }

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    missing_files = [

        file_path

        for file_path in model_files.values()

        if not os.path.exists(file_path)

    ]

    if missing_files:

        missing_names = [

            os.path.basename(
                file_path
            )

            for file_path in missing_files

        ]

        return None, [
            "Missing model files: "
            + ", ".join(missing_names)
        ]

    # --------------------------------------------------------
    # LOAD
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

            "scaler":
                scaler,

            "label_encoder":
                label_encoder,

            "condition_model":
                condition_model,

            "life_model":
                life_model,

        }

        return models, []

    except Exception as e:

        return None, [
            f"Model loading error: {str(e)}"
        ]


# ============================================================
# FETCH ONE BLYNK SENSOR
# ============================================================

def fetch_sensor(
    pin_or_url: str,
    timeout: int = 10
):
    """
    Fetch one Blynk virtual pin.

    Accepts:
        V0
        V1
        V2

    or a complete Blynk URL.

    Returns:
        value, error
    """

    # --------------------------------------------------------
    # TOKEN
    # --------------------------------------------------------

    token = get_blynk_token()

    if not token:

        return (
            None,
            "BLYNK_TOKEN is not configured. "
            "Add BLYNK_TOKEN in Streamlit Cloud Secrets."
        )

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    if not pin_or_url:

        return (
            None,
            "Blynk pin is not configured."
        )

    if str(pin_or_url).startswith("http"):

        url = pin_or_url

    else:

        url = build_blynk_url(
            str(pin_or_url)
        )

    if not url:

        return (
            None,
            "Unable to build Blynk API URL."
        )

    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    try:

        response = requests.get(
            url,
            timeout=timeout
        )

        # ----------------------------------------------------
        # HTTP STATUS
        # ----------------------------------------------------

        if response.status_code != 200:

            return (
                None,
                f"Blynk HTTP {response.status_code}: "
                f"{response.text}"
            )

        # ----------------------------------------------------
        # VALUE
        # ----------------------------------------------------

        raw_value = response.text.strip()

        if not raw_value:

            return (
                None,
                "Empty value returned from Blynk."
            )

        # ----------------------------------------------------
        # NUMERIC
        # ----------------------------------------------------

        try:

            value = float(
                raw_value
            )

        except ValueError:

            return (
                None,
                f"Non-numeric Blynk value: "
                f"'{raw_value}'"
            )

        return value, None

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    except requests.exceptions.Timeout:

        return (
            None,
            "Blynk request timed out."
        )

    # --------------------------------------------------------
    # CONNECTION
    # --------------------------------------------------------

    except requests.exceptions.ConnectionError:

        return (
            None,
            "Blynk connection error. "
            "Check Blynk Cloud and device connection."
        )

    # --------------------------------------------------------
    # REQUEST ERROR
    # --------------------------------------------------------

    except requests.exceptions.RequestException as e:

        return (
            None,
            f"Blynk request error: {str(e)}"
        )

    # --------------------------------------------------------
    # UNKNOWN ERROR
    # --------------------------------------------------------

    except Exception as e:

        return (
            None,
            f"Unexpected Blynk error: {str(e)}"
        )


# ============================================================
# FETCH COMMON VALUES
# ============================================================

def fetch_common_values():

    voltage, voltage_error = fetch_sensor(
        VPINS["voltage"]
    )

    temperature, temperature_error = fetch_sensor(
        VPINS["temperature"]
    )

    vibration, vibration_error = fetch_sensor(
        VPINS["vibration"]
    )

    return {

        "voltage":
            voltage,

        "temperature":
            temperature,

        "vibration":
            vibration,

        "errors": {

            "voltage":
                voltage_error,

            "temperature":
                temperature_error,

            "vibration":
                vibration_error,

        },

    }


# ============================================================
# FETCH PRODUCT CURRENT
# ============================================================

def fetch_product_current(
    product_name: str
):

    if product_name not in PRODUCT_CONFIG:

        return (
            None,
            f"Unknown product: {product_name}"
        )

    current_key = PRODUCT_CONFIG[
        product_name
    ]["current_key"]

    pin = VPINS.get(
        current_key
    )

    if not pin:

        return (
            None,
            f"No Blynk pin configured for "
            f"{product_name}"
        )

    return fetch_sensor(
        pin
    )


# ============================================================
# FETCH ALL REQUIRED VALUES
# ============================================================

def fetch_all_sensor_values(
    product_name: str
):

    common = fetch_common_values()

    current, current_error = (
        fetch_product_current(
            product_name
        )
    )

    errors = dict(
        common["errors"]
    )

    errors["current"] = current_error

    values = {

        "voltage":
            common["voltage"],

        "current":
            current,

        "temperature":
            common["temperature"],

        "vibration":
            common["vibration"],

    }

    return values, errors


# ============================================================
# PRODUCT AGE
# ============================================================

def calculate_product_age(
    manufacturing_date: str,
    installation_date: str
):

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

        if installation_dt < manufacture_dt:

            raise ValueError(
                "Installation date cannot be "
                "before manufacturing date."
            )

        manufacturing_age_days = (
            today - manufacture_dt
        ).days

        installation_age_days = (
            today - installation_dt
        ).days

        return {

            "manufacturing_age_days":
                max(
                    0,
                    manufacturing_age_days
                ),

            "installation_age_days":
                max(
                    0,
                    installation_age_days
                ),

            "manufacturing_age_years":
                max(
                    0,
                    manufacturing_age_days
                ) / 365.25,

            "installation_age_years":
                max(
                    0,
                    installation_age_days
                ) / 365.25,

        }

    except Exception:

        return {

            "manufacturing_age_days":
                0,

            "installation_age_days":
                0,

            "manufacturing_age_years":
                0,

            "installation_age_years":
                0,

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

    if models is None:

        return (
            None,
            "ML models are not loaded."
        )

    if product_name not in PRODUCT_CONFIG:

        return (
            None,
            f"Invalid product: {product_name}"
        )

    try:

        voltage = float(
            voltage
        )

        load_current = float(
            load_current
        )

        temperature = float(
            temperature
        )

        vibration = float(
            vibration
        )

    except (
        TypeError,
        ValueError
    ):

        return (
            None,
            "Invalid sensor input. "
            "All values must be numeric."
        )

    # --------------------------------------------------------
    # PRODUCT AGE
    # --------------------------------------------------------

    product_info = PRODUCT_CONFIG[
        product_name
    ]

    age_info = calculate_product_age(

        product_info[
            "manufacturing_date"
        ],

        product_info[
            "installation_date"
        ]

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
    # SCALE
    # --------------------------------------------------------

    try:

        scaled_data = models[
            "scaler"
        ].transform(
            input_data
        )

    except Exception as e:

        return (
            None,
            f"Scaling error: {str(e)}"
        )

    # --------------------------------------------------------
    # CONDITION
    # --------------------------------------------------------

    try:

        condition_encoded = (
            models[
                "condition_model"
            ].predict(
                scaled_data
            )
        )

        condition = (
            models[
                "label_encoder"
            ].inverse_transform(
                condition_encoded
            )[0]
        )

        condition = str(
            condition
        )

    except Exception as e:

        return (
            None,
            f"Condition prediction error: {str(e)}"
        )

    # --------------------------------------------------------
    # RUL
    # --------------------------------------------------------

    try:

        rul_prediction = (
            models[
                "life_model"
            ].predict(
                scaled_data
            )
        )

        rul = float(
            rul_prediction[0]
        )

        rul = max(
            0.0,
            rul
        )

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

        "rul_months":
            rul,

        "timestamp":
            datetime.now(),

    }

    return result, None


# ============================================================
# CONDITION COLOR
# ============================================================

def condition_color(
    condition: str
):

    condition_text = str(
        condition
    ).lower()

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

    return "#2563eb"


# ============================================================
# HISTORY SAVE
# ============================================================

def append_history(
    result: dict
):

    if not result:

        return False

    product_name = result.get(
        "product"
    )

    if not product_name:

        return False

    path = history_file(
        product_name
    )

    timestamp = result.get(
        "timestamp"
    )

    if isinstance(
        timestamp,
        datetime
    ):

        timestamp_text = (
            timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    else:

        timestamp_text = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    row = {

        "timestamp":
            timestamp_text,

        "product":
            result.get(
                "product"
            ),

        "voltage":
            round(
                float(
                    result.get(
                        "voltage",
                        0
                    )
                ),
                3
            ),

        "current":
            round(
                float(
                    result.get(
                        "current",
                        0
                    )
                ),
                3
            ),

        "temperature":
            round(
                float(
                    result.get(
                        "temperature",
                        0
                    )
                ),
                3
            ),

        "vibration":
            round(
                float(
                    result.get(
                        "vibration",
                        0
                    )
                ),
                4
            ),

        "condition":
            result.get(
                "condition",
                ""
            ),

        "rul_months":
            round(
                float(
                    result.get(
                        "rul",
                        0
                    )
                ),
                2
            ),

        "manufacturing_age_years":
            round(
                float(
                    result.get(
                        "manufacturing_age_years",
                        0
                    )
                ),
                2
            ),

        "installation_age_years":
            round(
                float(
                    result.get(
                        "installation_age_years",
                        0
                    )
                ),
                2
            ),

    }

    df_row = pd.DataFrame(
        [row],
        columns=HISTORY_COLUMNS
    )

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
# HISTORY LOAD
# ============================================================

def load_history(
    product_name: str
) -> pd.DataFrame:

    path = history_file(
        product_name
    )

    if not os.path.exists(path):

        return pd.DataFrame(
            columns=HISTORY_COLUMNS
        )

    try:

        df = pd.read_csv(
            path
        )

        for column in HISTORY_COLUMNS:

            if column not in df.columns:

                df[column] = None

        return df[
            HISTORY_COLUMNS
        ]

    except Exception:

        return pd.DataFrame(
            columns=HISTORY_COLUMNS
        )


# ============================================================
# HISTORY CLEAR
# ============================================================

def clear_history(
    product_name: str
):

    path = history_file(
        product_name
    )

    try:

        if os.path.exists(path):

            os.remove(
                path
            )

        return True

    except Exception:

        return False


# ============================================================
# BLYNK CONFIGURATION CHECK
# ============================================================

def blynk_configured():

    return bool(
        get_blynk_token()
    )


# ============================================================
# GET PRODUCTS
# ============================================================

def get_products():

    return list(
        PRODUCT_CONFIG.keys()
    )
