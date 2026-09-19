# Predictive Maintenance Dashboard

A multi-page Streamlit dashboard that pulls live sensor data from Blynk (voltage,
temperature, vibration, and per-product load current), runs it through your trained
ML models, and shows condition + Remaining Useful Life (RUL) predictions — with
CSV history logging per product.

## 1. Folder structure

```
dashboard/
├── Home.py                     ← main entry point / overview page
├── product_view.py             ← shared rendering logic for product pages
├── utils.py                    ← Blynk fetch, model loading, prediction, history I/O
├── style.py                    ← custom CSS / theming
├── requirements.txt
├── pages/
│   ├── 1_⚙️_Product_1.py
│   ├── 2_🔩_Product_2.py
│   └── 3_🛠️_Product_3.py
├── data/                       ← auto-created; per-product history CSVs land here
├── scaler.pkl                  ← ⚠️ YOU must add this (your trained scaler)
├── label_encoder.pkl           ← ⚠️ YOU must add this
├── condition.pkl               ← ⚠️ YOU must add this
└── life.pkl                    ← ⚠️ YOU must add this
```

## 2. Add your model files

Copy your four trained model files (`scaler.pkl`, `label_encoder.pkl`,
`condition.pkl`, `life.pkl`) into the same folder as `Home.py`. The app looks for
them right next to it.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run it

```bash
streamlit run Home.py
```

Streamlit automatically turns everything in `pages/` into sidebar navigation, so
you'll see **Home**, **Product 1**, **Product 2**, and **Product 3** in the sidebar.

## 5. What each page does

- **Home** — one-click "Fetch Live Data & Run All Predictions" for all three
  products at once, plus a summary card per product (condition badge, current,
  RUL, install/manufacture age, history count).
- **Product 1 / 2 / 3** — fetch that product's live values individually, view the
  model inputs, see condition + RUL as gauge charts, browse the full prediction
  history table, download it as CSV, view simple trend charts, or clear the log.

## 6. History storage

Every time you click **Fetch Live Data & Predict** on a product page (or run the
all-products fetch on Home), a row is appended to
`data/history_product_1.csv` (etc.) with: timestamp, voltage, current,
temperature, vibration, condition, RUL (months), and both age figures. Each
product page has a **Download History CSV** button and a **Clear History**
button.

## 7. Configuration

Blynk token, virtual pin mapping, and each product's manufacturing/installation
dates live at the top of `utils.py` in `BLYNK_TOKEN`, `VPIN_URLS`, and
`PRODUCT_CONFIG` — edit these to match your setup.

## Notes

- The condition badge color is inferred from keywords in your model's condition
  label (e.g. "good/normal" → green, "warning/moderate" → amber,
  "bad/critical/fault" → red). Adjust `condition_color()` in `utils.py` if your
  labels differ.
- If your Blynk device is offline or a datastream is missing, the page will show
  a clear error for exactly which value failed to fetch, instead of crashing.
