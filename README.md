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

- Inspect the **Peak Load Summary** cards that report maximum knee/hip moments, derived joint forces, and the computed risk factor (Low / Moderate / High)
- Review the **Joint Relationship Graphs** to see how individual prosthetic parameters (stiffness, damping, foot design, materials) influence peak moments and forces
- Use the optional **Design Interaction View** to visualize relationships between prosthetic components (e.g., stiffness vs damping) and how those pairings shift joint loading trends
- Download the synthesized results (CSV or JSON) for custom comparison or further plotting outside the app

## Data Description (optional)

### Data Source

This application generates synthetic simulation data based on parametric biomechanical models. No real patient data is used. The models are conceptual and designed for educational exploration of how prosthetic parameters influence joint loading and metabolic trends. All outputs are normalized and should not be used for clinical decision-making.

## Project Structure

```
Final-Project/
├── .app.py              # Main Streamlit application file containing all simulation logic, UI, and data visualization
├── README.md            # This documentation file
├── requirements.txt     # Python dependencies (streamlit, plotly, pandas, numpy, etc.)
```

**Key components in `.app.py`:**
- **Gait simulation models**: Generate synthetic joint moment curves based on prosthetic and patient parameters
- **Metabolic cost estimation**: Calculates energy expenditure using parametric relationships
- **Injury risk computation**: Produces a heuristic risk score incorporating joint loading, cadence, and speed
- **Interactive visualizations**: Plotly charts for moments, metabolic trends, and heatmaps
- **Data export functionality**: Download simulation results in CSV or JSON format

