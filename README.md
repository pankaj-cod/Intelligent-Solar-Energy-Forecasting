# ☀️ Intelligent Solar Energy Forecasting & AI Grid Optimization

A two-phase ML + GenAI system that **predicts solar PV power output** and **generates AI-powered grid optimization strategies** using RAG and LLM agents.

---

## What This Project Does

### Phase 1 — Solar Forecasting (ML)
Trains a **RandomForestRegressor** to predict **AC Power output** (kW) for two solar plants using weather sensor data, time features, and autoregressive lag features.

### Phase 2 — AI Grid Optimization (GenAI)
Takes Phase 1 predictions and runs them through a **5-step agentic pipeline**:

```
24-hour predictions → Forecast Summary → Risk Analysis → RAG Retrieval → LLM Recommendation
```

The pipeline uses **FAISS + sentence-transformers** for retrieval and **Groq (Llama 3.3 70B)** for structured strategy generation.

---

## Architecture

```
Intelligent-Solar-Energy-Forecasting/
│
├── app.py              ← Streamlit UI (tabbed: Phase 1 + Phase 2)
├── analysis.py         ← summarize_forecast() + analyze_risk()
├── rag.py              ← FAISS vector index + retrieve_guidelines()
├── llm.py              ← Groq LLM + generate_recommendation()
├── pipeline.py         ← run_ai_optimization() — unified agent pipeline
│
├── model_plant1.pkl    ← Trained RF model for Plant 1
├── model_plant2.pkl    ← Trained RF model for Plant 2
├── dataset/            ← Generation + weather CSVs
├── notebooks/          ← Training notebook (GenAI_Project)
├── requirements.txt    ← Python dependencies
├── PHASE2_DOCUMENTATION.md  ← Detailed Phase 2 technical docs
└── README.md           ← This file
```

---

## Pipeline Flow

### Phase 1: ML Prediction

| Step | Detail |
|---|---|
| **Data** | 4 CSVs — generation + weather sensor data for Plant 1 & Plant 2 |
| **Features** | Weather (irradiation, ambient temp, module temp), time (hour, day, month), autoregressive (lag-1, lag-2, lag-24, rolling-3 mean) |
| **Model** | `RandomForestRegressor(n_estimators=200, max_depth=10)` |
| **Results** | Plant 1: R² = 0.9948 · Plant 2: R² = 0.9925 |

### Phase 2: AI Optimization Pipeline

| Step | Function | What It Does |
|---|---|---|
| 1 | `summarize_forecast()` | Computes avg/max/min generation + variability (CV-based) |
| 2 | `analyze_risk()` | Rule-based risk classification + sudden drop detection (>30%) |
| 3 | `retrieve_guidelines()` | FAISS cosine search over 14 grid guidelines using `all-MiniLM-L6-v2` |
| 4 | `generate_recommendation()` | Groq Llama 3.3 70B generates structured JSON strategy |
| 5 | `run_ai_optimization()` | Chains steps 1–4 into a single callable pipeline |

---

## Tech Stack

| Component | Technology |
|---|---|
| **ML Model** | scikit-learn (RandomForestRegressor) |
| **UI** | Streamlit (dark theme, tabbed layout) |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **Vector DB** | FAISS (in-memory, cosine similarity) |
| **LLM** | Groq API — Llama 3.3 70B Versatile |
| **Data** | pandas, numpy |

---

## RAG System

The RAG (Retrieval-Augmented Generation) module grounds the LLM output in **specific operational guidelines** rather than generic advice.

- **Knowledge base:** 14 curated grid management guidelines (battery storage, load balancing, curtailment, maintenance, monitoring)
- **Embeddings:** 384-dim vectors via `all-MiniLM-L6-v2` (~23M params)
- **Index:** FAISS `IndexFlatIP` — inner product on L2-normalized vectors = cosine similarity
- **Retrieval:** Top-3 most relevant guidelines per query
- **Why RAG:** Prevents hallucination by constraining the LLM to reference defined rules

---

## LLM Integration

| Setting | Value |
|---|---|
| Provider | Groq |
| Model | `llama-3.3-70b-versatile` |
| Temperature | 0 (deterministic) |
| Response format | `json_object` (guaranteed valid JSON) |
| Max tokens | 512 |

**Output schema:**
```json
{
  "risk_interpretation": "...",
  "strategy": "...",
  "actions": "...",
  "justification": "..."
}
```

---

## UI Design

The app uses a **tabbed layout** with distinct visual identities:

| Tab | Theme | Content |
|---|---|---|
| **Phase 1 — Forecast** | Gold/amber accent | Plant selection, input sliders, predict button, AC power result |
| **Phase 2 — AI Grid Optimization** | Cyan/teal accent | Pipeline visualization, 24-hour summary, risk banner, RAG guidelines, LLM strategy |

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app.py
```

### Usage
1. **Phase 1 tab:** Select a plant, adjust inputs, click **⚡ Predict AC Power**
2. **Phase 2 tab:** Click **🚀 Run AI Optimization Pipeline** to see full analysis
3. Expand raw JSON sections for pipeline transparency

---

## Data

### Generation Data
- `Plant_1_Generation_Data.csv` / `Plant_2_Generation_Data.csv`
- Columns: `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `DC_POWER`, `AC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`

### Weather Sensor Data
- `Plant_1_Weather_Sensor_Data.csv` / `Plant_2_Weather_Sensor_Data.csv`
- Columns: `DATE_TIME`, `PLANT_ID`, `SOURCE_KEY`, `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`

---

## Feature Engineering (Phase 1)

| Feature | Source |
|---|---|
| `hour`, `day`, `month` | Extracted from `DATE_TIME` |
| `ac_power_prev_1` | `AC_POWER.shift(1)` |
| `ac_power_prev_2` | `AC_POWER.shift(2)` |
| `ac_power_prev_24` | `AC_POWER.shift(24)` |
| `ac_power_roll_3` | `AC_POWER.rolling(3).mean()` |
| `AMBIENT_TEMPERATURE` | Weather sensor |
| `MODULE_TEMPERATURE` | Weather sensor |
| `IRRADIATION` | Weather sensor |

**Dropped to prevent leakage:** `DC_POWER`, `DAILY_YIELD`, `TOTAL_YIELD`, `PLANT_ID`, `SOURCE_KEY`

---

## Key Design Decisions

1. **Rule-based risk over ML risk:** Interpretable, debuggable, no training needed
2. **FAISS over cloud vector DBs:** In-memory, zero latency, no external services
3. **Groq over OpenAI:** Free tier, faster inference, open-source model
4. **Temperature 0:** Reproducible outputs for same inputs
5. **No framework (LangChain etc.):** Pipeline is 5 function calls — frameworks add complexity without value here

---

## Dependencies

```
streamlit
pandas
scikit-learn
joblib
numpy
faiss-cpu
sentence-transformers
groq
```

---

## License

Apache License 2.0
