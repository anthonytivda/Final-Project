## Project Title

Smart Prosthetic Gait Load Optimizer — An interactive educational tool for simulating and analyzing how prosthetic design parameters affect joint loading, metabolic cost, and injury risk during walking.

## Biomedical Context

This application is designed for students, engineers, and clinicians interested in prosthetics and gait biomechanics. It helps users understand how different prosthetic design choices (stiffness, damping, foot type, material) affect lower-limb joint loading, energy expenditure, and injury risk. The tool enables exploration of trade-offs in prosthetic design and provides insight into how patient characteristics (body mass, walking speed, cadence) interact with device properties.

## Quick Start Instructions

### Opening the Repository in GitHub Codespaces

1. Navigate to the repository on GitHub: https://github.com/anthonytivda/Final-Project
2. Click the green **Code** button
3. Select the **Codespaces** tab
4. Click **Create codespace on main**
5. Wait for the codespace to initialize (this may take a few minutes)
6. Once ready, open a terminal in the codespace and proceed to "Running the Application" below

### Running the Application

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run .app.py
```

3. Open the URL provided in the terminal output (typically `http://localhost:8501`)

## Usage Guide

### Step 1: Configure Patient Parameters

In the left sidebar, adjust patient parameters:
- **Body Mass (kg)**: Set the patient's body weight (e.g., 70 kg)
- **Walking Speed (m/s)**: Choose preferred walking speed (e.g., 1.2 m/s for comfortable pace)
- **Cadence (steps/min)**: Set the walking cadence (e.g., 110 steps/min as default)

These parameters affect baseline joint loading and metabolic cost estimates.

### Step 2: Configure Prosthetic Design Parameters

Adjust prosthetic properties:
- **Stiffness (0-100)**: Controls how rigid the prosthetic is; stiffer prosthetics increase peak joint loads
- **Damping (0-50)**: Energy dissipation; higher damping smooths force peaks but increases energy cost
- **Foot Type**: Select from Default, Energy-Return, Stiff Blade, or Flexible Roll-over
- **Advanced Parameters** (foot length, heel height, toe stiffness, alignment, prosthesis weight, material type): Fine-tune additional design aspects

### Step 3: Explore Results and Optimize

- Read the **Contributing Risk Factors & Recommendations** panel and the circular risk gauge to understand how your current setup ranks (Low / Moderate / High) and what tweaks the tool suggests
- Inspect the **Primary Outputs** cards for peak knee/hip forces, peak moments, body-weight multiples, energy cost, and the calculated injury-risk percentage
- Use the **Explore Prosthetic Design Relationships** chart to sweep any single design parameter (stiffness, damping, alignment, etc.) and see how it affects a chosen outcome such as knee force or energy cost
- Export the generated data via the provided CSV/JSON download buttons to keep a record of simulations or perform offline comparisons

## Data Description

### Data Source
This application generates synthetic simulation data based on parametric biomechanical models. No real patient data is used. The models are conceptual and designed for educational exploration of how prosthetic parameters influence joint loading and metabolic trends. All outputs are normalized and should not be used for clinical decision-making.

## Project Structure

```
Final-Project/
├── .app.py              # Main Streamlit application file containing all simulation logic, UI, and data visualization
├── README.md            # Project documentation (context, setup, usage, reproducibility)
├── requirements.txt     # Python dependencies for the Streamlit simulator
├── .streamlit/          # Streamlit config (theme, layout preferences)
└── .devcontainer/       # Codespaces/dev-container setup (Dockerfile, scripts, supervisor config)
```

**Key components in `.app.py`:**
- **Gait simulation models**: Generate synthetic joint load profiles based on prosthetic and patient parameters
- **Metabolic cost estimation**: Calculates energy expenditure using parametric relationships
- **Injury risk computation**: Produces a heuristic risk score incorporating joint loading, cadence, and speed
- **Interactive dashboards**: Streamlit layouts for risk recommendations, peak-load metric cards, and the Explore Prosthetic Design Relationships sweep plot
- **Data export functionality**: Download simulation and summary results (moments CSV, summary JSON, metabolic trend CSV) for reproducibility

