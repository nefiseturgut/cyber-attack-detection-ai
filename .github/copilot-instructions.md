# 🤖 Copilot & AI Agent Instructions for siber_saldırı_project

## Project Overview
- Multi-dataset, multi-model cyber attack detection system (KDD Cup 1999, CICIDS2018, UNSW-NB15)
- 15 trained models: LSTM, CNN, LightGBM, XGBoost, and Ensemble for each dataset
- Modular scripts for preprocessing, training, evaluation, and deployment

## Key Directories & Files
- `datasets/` — Raw data (KDD, CICIDS2018, UNSW_NB15)
- `processed_data*/` — Preprocessed data for each dataset
- `lstm_data*/` — LSTM sequence data
- `models/` — Saved models, reports, and visualizations
- `deployment/` — API server (`api_server.py`), dashboard (`dashboard.py`), Docker, deployment guides
- `compare_*.py`, `ensemble_model*.py`, `*_model*.py` — Model training and comparison scripts

## Developer Workflows
- **Setup:** `pip install -r requirements.txt` (main) or `requirements_deployment.txt` (deployment)
- **Preprocessing:** `python run_all_preprocessing.py` or dataset-specific scripts
- **LSTM Data:** `python prepare_lstm_data_<dataset>.py`
- **Model Training:** Run `*_model*.py` scripts for each dataset/model
- **Evaluation:** Use `compare_*.py` scripts for model/dataset comparison
- **Deployment:**
  - REST API: `python deployment/api_server.py`
  - Dashboard: `streamlit run deployment/dashboard.py`
  - Docker: See `deployment/README.md`
- **Testing:** `python deployment/test_api.py` (API tests)

## Project Conventions
- File naming: `<model>_model_<dataset>.py` (e.g., `cnn_model_unsw.py`)
- Each dataset and model has a dedicated script; ensemble scripts combine predictions
- All model artifacts and reports are saved in `models/` with clear naming
- Visualizations auto-generated and saved in `models/`
- Deployment expects models in `../models/` relative to `deployment/`
- API endpoints and dashboard model names match script/model file names

## Integration & Patterns
- Models loaded dynamically by name in API/dashboard
- Ensemble models use weighted voting (see `ensemble_model*.py`)
- Data flows: raw → preprocessing → LSTM prep (if needed) → model training → evaluation → deployment
- Docker and requirements files are kept separate for main and deployment environments

## Troubleshooting & Tips
- If models are not found, check paths in `deployment/api_server.py`
- For GPU, TensorFlow auto-detects (see deployment README)
- Use `deployment/DEPLOYMENT_GUIDE.md` for advanced deployment, cloud, and CI/CD
- All scripts are designed to be run independently; outputs are saved to disk

## Examples
- Train CNN for UNSW: `python cnn_model_unsw.py`
- Compare all models: `python compare_all_datasets.py`
- Start API: `python deployment/api_server.py`
- Run API tests: `python deployment/test_api.py`

---
For more, see `README.md` and `deployment/README.md`. Update this file if project structure or conventions change.
