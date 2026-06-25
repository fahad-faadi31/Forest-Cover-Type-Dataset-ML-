<<<<<<< HEAD
# ForestML — Forest Cover Type Classifier
Open-End Lab | Spring 2026 | CLO 03, 04, 05

## Project Structure

```
forestml/
├── backend/
│   ├── train_model.py      ← Step 1: Run this to train & save models
│   ├── main.py             ← Step 2: FastAPI server
│   └── requirements.txt
├── frontend/
│   ├── index.html          ← Open in browser after starting API
│   ├── style.css
│   └── script.js
├── models/                 ← Auto-created by train_model.py
│   ├── scaler.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── kmeans.pkl
│   └── metrics.json
└── data/
    └── covtype_csv.zip     ← Put your dataset here
```

---

## How to Run

### Step 0 — Place the dataset
Copy your `covtype_csv.zip` into the `data/` folder:
```
forestml/data/covtype_csv.zip
```

### Step 1 — Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Train and save the models
```bash
cd backend
python train_model.py
```
This will:
- Load the dataset from `../data/covtype_csv.zip`
- Train KNN, Naïve Bayes, Random Forest, and K-Means
- Save each model as a `.pkl` file in `../models/`
- Save evaluation metrics to `../models/metrics.json`

Expected output:
```
[1/5] Loading dataset...
[2/5] Fitting scaler...
[3/5] Training KNN...        Accuracy=68.97%  F1=68.34%
[3/5] Training Naïve Bayes... Accuracy=12.53%  F1=11.76%
[3/5] Training Random Forest... Accuracy=71.53%  F1=70.95%
[3/5] Training K-Means...
[4/5] Saving model files...
[5/5] Done!
```

### Step 3 — Start the API
```bash
cd backend
uvicorn main:app --reload --port 8000
```
API will be live at: http://localhost:8000
Interactive docs at: http://localhost:8000/docs

### Step 4 — Open the Frontend
Simply open `frontend/index.html` in your browser.
The status badge in the navbar will turn green when connected.

---

## API Endpoints

| Method | URL        | Description                        |
|--------|------------|------------------------------------|
| GET    | /health    | Check server + loaded models       |
| GET    | /metrics   | Get all model evaluation metrics   |
| POST   | /predict   | Run a prediction (JSON body)       |
| GET    | /docs      | Swagger UI (auto-generated)        |

### Example Prediction Request
```json
POST http://localhost:8000/predict
{
  "elevation": 2596,
  "aspect": 51,
  "slope": 3,
  "h_dist_hydrology": 258,
  "v_dist_hydrology": 0,
  "h_dist_roadways": 510,
  "hillshade_9am": 221,
  "hillshade_noon": 232,
  "hillshade_3pm": 148,
  "h_dist_fire_points": 6279,
  "wilderness_area": 1,
  "soil_type": 29,
  "algorithm": "random_forest"
}
```

---

## Dataset Info
- **Name**: Forest Cover Type
- **Source**: UCI Machine Learning Repository
- **Rows**: 581,012  |  **Features**: 55  |  **Classes**: 7
- **Target**: Cover_Type (1=Spruce/Fir, 2=Lodgepole Pine, ..., 7=Krummholz)
=======
# Forest-Cover-Type-Dataset-ML-
In this project on a data set named as Forest Cover Type. i will apply different algorithms to check which algorithm will fit perfectly.
<br>
i will apply different algorithms like knn,naive base,Random Forest e.t.c to check which will fit perfectly<br>
i will make a frontend using html,css,js then connect to fast APi <br>
that's it for now 
>>>>>>> 6f41f19a7f094436daf40d75f3d8cc0f3a336b5f
