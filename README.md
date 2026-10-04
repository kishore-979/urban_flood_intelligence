# Intelligent Urban Flood Risk Prediction and A*-Based Safe Route Planning System

This repository is a semester-scale academic prototype for urban flood-risk prediction and A*-based safe-route planning. It uses verified public weather, terrain, and road-network sources where possible, and a clearly labeled synthetic demonstration risk layer where real citywide road-flood labels are not publicly available.

## Phase 0 status

- Feasibility verdict: YELLOW — feasible with limitations
- Selected region for prototype: Houston / Harris County, Texas
- Verified public sources:
  - NOAA NCEI daily summaries for rainfall and weather conditions
  - USGS 3DEP DEM for elevation and terrain features
  - Geofabrik Texas OSM extract for road-network topology
  - FEMA / local flood-control sources for flood-zone context

## Data and scope honesty

This project does not claim to be a national real-time flood forecasting system. The prototype is intentionally limited to a selected urban region and a transparent design that demonstrates:

- ML-based flood risk prediction
- FOL and rule-based reasoning
- forward chaining
- backward chaining
- an intelligent agent
- A* route planning
- flood-aware cost logic
- Streamlit demonstration

## Windows setup

```powershell
cd path\to\project
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_project.py
streamlit run app\streamlit_app.py
```

## Project structure

```text
urban_flood_intelligence/
├── app/
├── data/
├── models/
├── notebooks/
├── outputs/
├── src/
├── README.md
├── requirements.txt
├── run_project.py
└── phase0_feasibility_report.md
```

## Run the demonstration

```bash
python run_project.py
```

This script trains a simple Random Forest classifier on a synthetic demonstration dataset, performs reasoning steps, and compares the shortest path with the flood-aware route.

## Important limitation

Because a complete and verified road-level flood-label dataset for the selected city was not available in this environment, the project keeps the ML data layer honest and explicit:

- weather + terrain + road-network information are based on actual public sources,
- the flood label layer used for the demonstration is synthetic and is labeled as such,
- the route-planning and reasoning components remain runnable and explainable for an academic prototype.

## Project phases

1. Feasibility and dataset verification
2. Environment setup
3. Data acquisition
4. Cleaning and EDA
5. Feature engineering
6. ML model and evaluation
7. FOL and knowledge representation
8. Forward and backward chaining
9. Agent and route planning
10. Streamlit interface
11. Documentation and viva preparation
