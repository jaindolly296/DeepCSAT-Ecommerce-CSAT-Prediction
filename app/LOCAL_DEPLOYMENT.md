# DeepCSAT Local Deployment

## Required project structure

DeepCSAT - E-Commerce Customer Satisfaction Score Prediction/
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── LOCAL_DEPLOYMENT.md
└── models/
    ├── deepcsat_final_model.keras
    ├── deepcsat_preprocessor.joblib
    ├── deepcsat_frequency_maps.joblib
    └── deepcsat_metadata.json

## Windows setup

Open PowerShell inside the main project folder.

Create a virtual environment:

    python -m venv .venv

Activate the environment:

    .venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r app\requirements.txt

Run the application:

    streamlit run app\app.py

Streamlit will display a local URL, normally:

    http://localhost:8501

Open that URL in a browser to use the application.

