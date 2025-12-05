# Smart Prosthetic Gait Load Optimizer

Educational Streamlit app for exploring prosthetic design trade-offs. The app simulates conceptual joint loading (knee & hip), estimated metabolic cost, and a heuristic injury risk score as prosthetic and patient parameters vary.

This tool is intended for teaching and exploration — it is not a clinical predictor.

## Features
- Adjustable patient parameters: body mass, walking speed, cadence
- Adjustable prosthetic parameters: stiffness, damping, foot type
- Interactive Plotly charts showing joint moments and metabolic trends
- Injury risk heatmap across mass and speed
- Downloadable CSV / JSON results for further analysis

## Run locally

1. Create a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run Streamlit:

```bash
streamlit run .app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501).

## Notes for learners
- The models inside are simplified mathematical trends, designed to illustrate how stiffness, damping, and gait speed/cadence influence joint moments and energy cost.
- Use the *Explore injury risk* heatmap to see how mass and speed interact with your chosen prosthetic settings.

## Dependencies
See `requirements.txt` for exact package versions.

## License
This repository contains educational code. Please adapt and extend for coursework or demonstrations.
# Project Title

[One-sentence summary of your project]

## Biomedical Context

[Who/what this app or game is for]

## Quick Start Instructions

### Opening the Repository in GitHub Codespaces

[Instructions on how to open this repo in GitHub Codespaces]

### Running the Application

[Exact command(s) to run the app/game, e.g., `pip install streamlit` then `streamlit run app.py` or `DISPLAY=:0 love .`]

## Usage Guide

[Step-by-step explanation with screenshots or text]

- **Step 1:** [Description]
- **Step 2:** [Description]
- **Step 3:** [Description]

## Data Description (optional)

### Data Source

[Where your data came from]



## Project Structure

[Description of the project structure and organization]

