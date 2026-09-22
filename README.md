# 🏠 House Price Prediction

A full-stack machine learning web application that predicts house prices based on area, bedrooms, amenities, and other features.

- **Backend**: Flask REST API (`backend/app.py`, port 5000)
- **Frontend**: Streamlit (`frontend/ui.py`, port 8501)
- **ML Model**: Linear Regression / Ridge / Lasso (scikit-learn) — best selected by RMSE
- **Dataset**: [Housing Prices Dataset — mannatpruthi/house-prediction](https://www.kaggle.com/code/mannatpruthi/house-prediction) — 545 rows × 13 columns

---

## Technologies Used

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Machine Learning | scikit-learn, pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Backend API | Flask, flask-cors |
| Frontend | Streamlit |
| Model Serialisation | joblib |
| Report Generation | python-docx |

---

## Project Structure

```
house_price_project/
├── backend/
│   └── app.py                              # Flask REST API (port 5000)
├── data/
│   └── train.csv                           # Housing Prices dataset (545 rows)
├── frontend/
│   └── ui.py                               # Streamlit UI (port 8501)
├── model/
│   ├── house_price_model.pkl               # Trained sklearn Pipeline
│   ├── feature_columns.json                # Ordered feature list (12 features)
│   └── metrics.json                        # RMSE, MAE, R2 scores
├── report_images/                          # Auto-generated chart PNGs
├── train_model.py                          # ML training pipeline
├── model.py                                # Simple model trainer (house_price.csv)
├── app.py                                  # Simple Flask app (port 5001)
├── generate_report.py                      # Word report generator
├── generate_html_report.py                 # HTML report generator
├── generate_submission_report.py           # Submission report generator
├── house_price.csv                         # Simple dataset (area, rooms, price)
├── house_price_model.pkl                   # Simple trained model
├── house-price-prediction-project-report.html
├── House_Price_Prediction_Report.docx
├── SrijanChatterjee_ProjectReport.docx
├── SrijanChatterjee_HousePricePrediction.ipynb
├── requirements.txt
└── README.md
```

---

## Dataset

- **Dataset:** [Housing Prices Dataset — mannatpruthi/house-prediction](https://www.kaggle.com/code/mannatpruthi/house-prediction) (`data/train.csv`)
- **Size:** 545 rows × 13 columns
- **Target Variable:** `price`
- **Features Used:** 12 features

### 12 Features:

| Feature | Type | Description |
|---------|------|-------------|
| `area` | Numeric | House area in square feet |
| `bedrooms` | Numeric | Number of bedrooms |
| `bathrooms` | Numeric | Number of bathrooms |
| `stories` | Numeric | Number of floors/stories |
| `mainroad` | Binary (yes/no) | Connected to main road |
| `guestroom` | Binary (yes/no) | Has a guest room |
| `basement` | Binary (yes/no) | Has a basement |
| `hotwaterheating` | Binary (yes/no) | Has hot water heating system |
| `airconditioning` | Binary (yes/no) | Has air conditioning |
| `parking` | Numeric | Number of parking spaces |
| `prefarea` | Binary (yes/no) | Located in preferred area |
| `furnishingstatus` | Ordinal | furnished / semi-furnished / unfurnished |

### Preprocessing Applied:
- Binary yes/no columns → encoded as **1 / 0**
- `furnishingstatus` → encoded as **furnished=2, semi-furnished=1, unfurnished=0**

---

## Model Performance

| Model | RMSE | MAE | R2 Score |
|-------|------|-----|----------|
| **LinearRegression** | **653,784** | **532,353** | **0.9136** |
| Ridge (alpha=10) | 661,877 | 542,915 | 0.9115 |
| Lasso (alpha=50) | 653,827 | 532,389 | 0.9136 |

> Best model: **LinearRegression** — selected by lowest RMSE on 20% validation split.

---

## Setup & Installation

```bash
# 1. Clone or download the project
cd house_price_project

# 2. (Recommended) Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Run Instructions

### Step 1 — Train the ML Model
```bash
python train_model.py
```
- Reads `data/train.csv`
- Encodes binary + categorical columns
- Trains 3 models, picks best by RMSE
- Saves `model/house_price_model.pkl`, `model/metrics.json`
- Saves 5 chart PNGs to `report_images/`

---

### Step 2 — Start Flask Backend API
```bash
python backend/app.py
```
- API starts at **http://localhost:5000**
- Keep this terminal **running**

---

### Step 3 — Launch Streamlit Frontend
```bash
python -m streamlit run frontend/ui.py
```
- Web app opens at **http://localhost:8501**
- 3 Tabs: **Predict** | **Dataset Explorer** | **Model Info**

---

### Step 4 (Optional) — Generate Word Report
```bash
python generate_report.py
```
- Creates `SrijanChatterjee_ProjectReport.docx`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `http://localhost:5000/` | Health check |
| GET | `http://localhost:5000/model-info` | Model name, features, metrics |
| POST | `http://localhost:5000/predict` | Returns predicted house price |

### Example API Request
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"area\": 7420, \"bedrooms\": 4, \"bathrooms\": 2, \"stories\": 3,
       \"mainroad\": 1, \"guestroom\": 0, \"basement\": 0, \"hotwaterheating\": 0,
       \"airconditioning\": 1, \"parking\": 2, \"prefarea\": 1, \"furnishingstatus\": 2}"
```

> **Note:** Send binary features as `0`/`1` and `furnishingstatus` as `0` (unfurnished), `1` (semi-furnished), or `2` (furnished).

### Response
```json
{
  "predicted_price": 9850000.00,
  "predicted_price_formatted": "9,850,000"
}
```

---

## Streamlit App Features

| Tab | Contents |
|-----|----------|
| **Predict** | Sliders + dropdowns for 12 features → calls Flask API → shows predicted price + bar chart |
| **Dataset Explorer** | Raw data table, price distribution, scatter plots, correlation heatmap |
| **Model Info** | RMSE/MAE/R2 metrics, model comparison chart, all training images |

---

## Key Information

- **ML Pipeline:** `StandardScaler` + best estimator inside `sklearn.pipeline.Pipeline`
- **Validation Strategy:** 80/20 train-validation split with `random_state=42`
- **Preprocessing:** Binary yes/no → 1/0; furnishingstatus → 2/1/0 ordinal
- **Missing Values:** Imputed with column median values
- **Model saved with:** `joblib.dump()` for fast loading
- **CORS enabled** on Flask API so Streamlit can communicate cross-port
- **`app.py` (root):** Standalone simple Flask app on **port 5001** using `house_price.csv` (area + rooms only) — separate from the main backend

---

## License

MIT License — free to use, modify, and distribute.
