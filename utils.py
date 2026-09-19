"""
Shared utilities for the Predictive Maintenance Dashboard.

Contains:
- Blynk / product configuration
- Cached ML model loading
- Live sensor fetching from Blynk
- Product-age calculation
- Prediction pipeline
- Per-product CSV history read/write
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

BLYNK_TOKEN = "urrF6RolLNdGuvfL2NV7XLmtGsBjJcDB"
BASE_URL = "https://blynk.cloud/external/api"

VPIN_URLS = {
    "voltage": f"{BASE_URL}/get?token={BLYNK_TOKEN}&V0",
    "product1_current": f"{BASE_URL}/get?token={BLYNK_TOKEN}&V1",
    "product2_current": f"{BASE_URL}/get?token={BLYNK_TOKEN}&V2",
    "product3_current": f"{BASE_URL}/get?token={BLYNK_TOKEN}&V3",
    "temperature": f"{BASE_URL}/get?token={BLYNK_TOKEN}&V4",
    "vibration": f"{BASE_URL}/get?token={BLYNK_TOKEN}&V5",
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
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")
LABEL_ENCODER_FILE = os.path.join(BASE_DIR, "label_encoder.pkl")
CONDITION_MODEL_FILE = os.path.join(BASE_DIR, "condition.pkl")
LIFE_MODEL_FILE = os.path.join(BASE_DIR, "life.pkl")

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


def history_file(product_name: str) -> str:
    safe_name = product_name.replace(" ", "_").lower()
    return os.path.join(DATA_DIR, f"history_{safe_name}.csv")


# ============================================================
# MODEL LOADING (cached across reruns)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():
    missing = [
        f
        for f in [
            SCALER_FILE,
            LABEL_ENCODER_FILE,
            CONDITION_MODEL_FILE,
            LIFE_MODEL_FILE,
        ]
        if not os.path.exists(f)
    ]
    if missing:
        return None, missing

    try:
        scaler = joblib.load(SCALER_FILE)
        label_encoder = joblib.load(LABEL_ENCODER_FILE)
        condition_model = joblib.load(CONDITION_MODEL_FILE)
        life_model = joblib.load(LIFE_MODEL_FILE)
        return {
            "scaler": scaler,
            "label_encoder": label_encoder,
            "condition_model": condition_model,
            "life_model": life_model,
        }, []
    except Exception as e:  # noqa: BLE001
        return None, [str(e)]


# ============================================================
# BLYNK SENSOR FETCH
# ============================================================

def fetch_sensor(url: str, timeout: int = 5):
    """Fetch a single Blynk virtual-pin value. Returns (value, error_message)."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text}"

        raw_value = response.text.strip()
        if raw_value == "":
            return None, "Empty value returned from Blynk"

        return float(raw_value), None

    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except requests.exceptions.ConnectionError:
        return None, "Connection error (check internet / device online status)"
    except ValueError:
        return None, f"Non-numeric value received: '{response.text.strip()}'"
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def fetch_common_values():
    """Fetch voltage, temperature, vibration. Returns dict with values + errors."""
    voltage, v_err = fetch_sensor(VPIN_URLS["voltage"])
    temperature, t_err = fetch_sensor(VPIN_URLS["temperature"])
    vibration, vib_err = fetch_sensor(VPIN_URLS["vibration"])

    return {
        "voltage": voltage,
        "temperature": temperature,
        "vibration": vibration,
        "errors": {
            "voltage": v_err,
            "temperature": t_err,
            "vibration": vib_err,
        },
    }


def fetch_product_current(product_name: str):
    key = PRODUCT_CONFIG[product_name]["current_key"]
    return fetch_sensor(VPIN_URLS[key])


# ============================================================
# PRODUCT AGE CALCULATION
# ============================================================

def calculate_product_age(manufacturing_date: str, installation_date: str):
    try:
        today = datetime.today()
        manufacture_dt = datetime.strptime(manufacturing_date, "%Y-%m-%d")
        installation_dt = datetime.strptime(installation_date, "%Y-%m-%d")

        if installation_dt < manufacture_dt:
            raise ValueError("Installation date cannot be before manufacturing date.")

        manufacturing_age_days = (today - manufacture_dt).days
        installation_age_days = (today - installation_dt).days

        return {
            "manufacturing_age_days": manufacturing_age_days,
            "installation_age_days": installation_age_days,
            "manufacturing_age_years": manufacturing_age_days / 365.25,
            "installation_age_years": installation_age_days / 365.25,
        }
    except Exception:  # noqa: BLE001
        return {
            "manufacturing_age_days": 0,
            "installation_age_days": 0,
            "manufacturing_age_years": 0,
            "installation_age_years": 0,
        }


# ============================================================
# PREDICTION
# ============================================================

def predict_product(models, product_name, voltage, load_current, temperature, vibration):
    """Run the condition + RUL prediction pipeline. Returns (result_dict, error)."""

    product_info = PRODUCT_CONFIG[product_name]
    age_info = calculate_product_age(
        product_info["manufacturing_date"], product_info["installation_date"]
    )

    input_data = np.array([[voltage, load_current, temperature, vibration]])

    try:
        scaled_data = models["scaler"].transform(input_data)
    except Exception as e:  # noqa: BLE001
        return None, f"Scaling error: {e}"

    try:
        condition_encoded = models["condition_model"].predict(scaled_data)
        condition = models["label_encoder"].inverse_transform(condition_encoded)[0]
    except Exception as e:  # noqa: BLE001
        return None, f"Condition prediction error: {e}"

    try:
        rul_prediction = models["life_model"].predict(scaled_data)
        rul = max(0.0, float(rul_prediction[0]))
    except Exception as e:  # noqa: BLE001
        return None, f"RUL prediction error: {e}"

    result = {
        "product": product_name,
        "manufacturing_date": product_info["manufacturing_date"],
        "installation_date": product_info["installation_date"],
        "manufacturing_age_years": age_info["manufacturing_age_years"],
        "installation_age_years": age_info["installation_age_years"],
        "voltage": voltage,
        "current": load_current,
        "temperature": temperature,
        "vibration": vibration,
        "condition": condition,
        "rul": rul,
        "timestamp": datetime.now(),
    }
    return result, None


# ============================================================
# CONDITION -> COLOR MAPPING
# ============================================================

def condition_color(condition: str) -> str:
    c = str(condition).lower()
    if any(word in c for word in ["good", "normal", "healthy", "excellent", "ok"]):
        return "#16a34a"  # green
    if any(word in c for word in ["warn", "moderate", "fair", "degrad"]):
        return "#f59e0b"  # amber
    if any(word in c for word in ["bad", "critical", "fault", "fail", "poor"]):
        return "#dc2626"  # red
    return "#2563eb"  # blue fallback


# ============================================================
# HISTORY: SAVE / LOAD
# ============================================================

def append_history(result: dict):
    path = history_file(result["product"])
    row = {
        "timestamp": result["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "product": result["product"],
        "voltage": round(result["voltage"], 3),
        "current": round(result["current"], 3),
        "temperature": round(result["temperature"], 3),
        "vibration": round(result["vibration"], 4),
        "condition": result["condition"],
        "rul_months": round(result["rul"], 2),
        "manufacturing_age_years": round(result["manufacturing_age_years"], 2),
        "installation_age_years": round(result["installation_age_years"], 2),
    }

    df_row = pd.DataFrame([row], columns=HISTORY_COLUMNS)

    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def load_history(product_name: str) -> pd.DataFrame:
    path = history_file(product_name)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def clear_history(product_name: str):
    path = history_file(product_name)
    if os.path.exists(path):
        os.remove(path)
