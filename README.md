# Heart Disease Prediction API

A FastAPI web service that predicts the presence of heart disease using a Random Forest classifier trained on the UCI Heart Disease dataset.

## Project Structure

```
heart-disease-prediction/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application with endpoints
│   └── schemas.py       # Pydantic models for request/response
├── model/
│   ├── train.py         # Model training script
│   └── heart_model.joblib  # Trained model (auto-generated)
├── data/
│   └── heart.csv        # Heart Disease dataset
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # Python dependencies
├── .dockerignore
├── .gitignore
└── README.md
```

## Features

- **13 clinical input features**: age, sex, chest pain type, blood pressure, cholesterol, etc.
- **Binary prediction**: presence vs. absence of heart disease
- **Confidence score**: probability of the prediction
- **Swagger UI**: interactive API documentation at `/docs`

## API Endpoints

| Method | Endpoint   | Description                          |
|--------|-----------|--------------------------------------|
| GET    | `/health` | Health check — confirms model loaded |
| GET    | `/info`   | Model metadata and feature list      |
| POST   | `/predict`| Predict heart disease from features  |

## Quick Start (Local)

### 1. Train the Model

```bash
pip install -r requirements.txt
python -m model.train
```

### 2. Run with Docker Compose

```bash
docker-compose build
docker-compose up
```

The API will be available at **http://localhost:8000**.

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/info

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

## Deployment on Render

1. Push this repository to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com/) → **New** → **Web Service**.
3. Connect your GitHub repo.
4. Set:
   - **Environment**: Docker
   - **Build Context**: `.` (root)
5. Deploy and wait for the build to complete.
6. Test your live URL endpoints.

## Model Details

- **Algorithm**: Random Forest Classifier (100 estimators, max depth 10)
- **Pipeline**: StandardScaler → RandomForestClassifier
- **Dataset**: UCI Heart Disease Dataset (1,025 samples, 13 features)
- **Test Accuracy**: ~72%

## Input Features

| Feature   | Description                                    | Range   |
|-----------|------------------------------------------------|---------|
| age       | Age in years                                   | 1–120   |
| sex       | Sex (1 = male, 0 = female)                     | 0–1     |
| cp        | Chest pain type                                | 0–3     |
| trestbps  | Resting blood pressure (mm Hg)                 | 50–300  |
| chol      | Serum cholesterol (mg/dl)                      | 100–600 |
| fbs       | Fasting blood sugar > 120 mg/dl                | 0–1     |
| restecg   | Resting ECG results                            | 0–2     |
| thalach   | Maximum heart rate achieved                    | 50–250  |
| exang     | Exercise-induced angina                        | 0–1     |
| oldpeak   | ST depression induced by exercise              | 0–10    |
| slope     | Slope of peak exercise ST segment              | 0–2     |
| ca        | Major vessels colored by fluoroscopy           | 0–4     |
| thal      | Thalassemia type                               | 0–3     |
