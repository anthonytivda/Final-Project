"""
Smart Prosthetic Gait Load Optimizer
Streamlit app entrypoint. Modular, well-commented, educational simulation
for exploring prosthetic stiffness/damping effects on joint loads,
metabolic cost, and injury risk trends.

Run: streamlit run .app.py

Note: This is a conceptual educational tool - not a clinical predictor.
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from streamlit_lottie import st_lottie
except Exception:
    # graceful fallback if lottie isn't available
    def st_lottie(o, height=200, key=None):
        return None

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

# ------------------------- Styling -------------------------

_PRIMARY = "#0f4c75"
_ACCENT = "#3282b8"
_BG = "#0b132b"
_CARD = "#1b263b"

st.set_page_config(page_title="Smart Prosthetic Gait Load Optimizer", layout="wide", initial_sidebar_state="expanded")

def local_css():
    css = f"""
    <style>
    .css-18e3th9 {{padding-top: 1rem;}}
    .stApp {{background: linear-gradient(180deg, #071426 0%, #071a2a 100%); color: #e6eef8;}}
    .stSidebar {{background-color: rgba(10,15,30,0.85);}}
    .block-container {{padding:1.5rem 2rem 2rem 2rem}}
    .big-number {{font-size:26px; font-weight:700; color:{_ACCENT}}}
    .small-muted {{color:#9fb3d6; font-size:13px}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ------------------------- Simulation Models -------------------------

def simulate_joint_moments(mass, speed, cadence, stiffness, damping, foot_type, num_points=200):
    phase = np.linspace(0, 1, num_points)  # normalized gait phase 0–1
    base_amp = 0.6 * mass * (0.6 + speed)
    fmod = {"Default":1.0, "Energy-Return":0.9, "Stiff Blade":1.1, "Flexible Roll-over":0.95}[foot_type]
    k_factor = 1 + (stiffness - 50) / 300.0
    d_factor = 1.0 / (1.0 + damping / 50.0)

    def bump(x, center, width, height):
        return height * np.exp(-0.5 * ((x - center) / width)**2)

    knee = (
        bump(phase, 0.15, 0.08, +1.0) +
        bump(phase, 0.45, 0.10, -0.7) +
        bump(phase, 0.75, 0.12, +0.4)
    )
    hip = (
        bump(phase, 0.10, 0.10, -1.0) +
        bump(phase, 0.50, 0.20, +0.8) +
        bump(phase, 0.85, 0.08, +0.4)
    )
    knee = knee * base_amp * fmod * k_factor * d_factor
    hip  = hip  * base_amp * fmod * k_factor * d_factor
    time = np.linspace(0, 1000 / cadence if cadence > 0 else 1.0, num_points)

    return pd.DataFrame({
        "time_ms": time,
        "phase_rad": phase * 2 * np.pi,
        "knee_Nm": knee,
        "hip_Nm": hip
    })


def estimate_metabolic_cost(mass, speed, cadence, stiffness, damping, foot_type):
    base_cost = 3.5 + 0.1 * mass + 2.0 * speed ** 2
    cadence_effect = 1.0 + 0.0008 * (cadence - 110) ** 2 / 1000.0
    stiffness_penalty = 1.0 + max(0, (stiffness - 80)) / 500.0
    damping_penalty = 1.0 + max(0, (40 - damping)) / 300.0
    foot_mod = {"Default": 1.0, "Energy-Return": 0.94, "Stiff Blade": 1.06, "Flexible Roll-over": 0.98}
    total = base_cost * cadence_effect * stiffness_penalty * damping_penalty * foot_mod.get(foot_type, 1.0)
    kcal_per_min = total * 0.01433 * mass / 60.0
    return max(0.0, kcal_per_min)


def compute_injury_risk(df_moments, mass, speed, cadence):
    peak_knee = float(df_moments['knee_Nm'].abs().max())
    peak_hip  = float(df_moments['hip_Nm'].abs().max())
    knee_ratio = peak_knee / (2.0 * mass)
    hip_ratio  = peak_hip / (2.5 * mass)
    load_index = 0.7 * knee_ratio + 0.3 * hip_ratio
    load_norm = float(np.clip((load_index - 0.3) / 1.2, 0.0, 2.0))
    cadence_norm = float(np.clip(abs(cadence - 110) / 25.0, 0.0, 2.0))
    speed_norm = float(np.clip(max(0.0, speed - 1.2) / 0.6, 0.0, 2.0))
    overall_index = 0.7 * load_norm + 0.2 * cadence_norm + 0.1 * speed_norm
    risk = float(np.clip((overall_index / 2.0) * 100.0, 0.0, 100.0))
    return risk, peak_knee, peak_hip

# ------------------------- Plotting -------------------------

def plot_summary_bar(risk, peak_knee, peak_hip):
    df = pd.DataFrame({
        'Metric': ['Injury Risk (0-100)', 'Peak Knee Moment (N·m)', 'Peak Hip Moment (N·m)'],
        'Value': [risk, peak_knee, peak_hip]
    })
    fig = px.bar(df, x='Metric', y='Value', text='Value', color='Metric', template='plotly_dark')
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside', showlegend=False)
    fig.update_layout(yaxis_title='Value', xaxis_title='', title='Summary Metrics',
                      yaxis=dict(range=[0, max(200, peak_knee*1.2, peak_hip*1.2, risk*1.2)]))
    return fig


def plot_metabolic_bar(metabolic):
    df = pd.DataFrame({'Metric': ['Estimated Metabolic Cost (kcal/min)'], 'Value': [metabolic]})
    fig = px.bar(df, x='Metric', y='Value', text='Value', color='Metric', template='plotly_dark')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', showlegend=False)
    fig.update_layout(yaxis_title='kcal/min', xaxis_title='', title='Metabolic Cost',
                      yaxis=dict(range=[0, max(5, metabolic*1.2)]))
    return fig


def plot_metabolic_vs_speed(mass, speed, cadence, stiffness, damping, foot_type):
    speed_range = np.linspace(max(0.4, speed-0.6), min(2.2, speed+0.6), 15)
    costs = [estimate_metabolic_cost(mass, s, cadence, stiffness, damping, foot_type) for s in speed_range]
    df = pd.DataFrame({"Speed_m_s": speed_range, "kcal_per_min": costs})
    fig = px.line(df, x='Speed_m_s', y='kcal_per_min', markers=True,
                  title='Estimated Metabolic Cost vs Speed', template='plotly_dark')
    fig.update_layout(xaxis_title='Speed (m/s)', yaxis_title='Estimated kcal/min')
    return fig, df


def plot_injury_risk_heatmap(mass_range, speed_range, cadence, stiffness, damping, foot_type):
    rows = []
    for m in mass_range:
        for s in speed_range:
            df = simulate_joint_moments(m, s, cadence, stiffness, damping, foot_type)
            risk, _, _ = compute_injury_risk(df, m, s, cadence)
            rows.append({"mass": m, "speed": s, "risk": risk})
    df2 = pd.DataFrame(rows)
    pivot = df2.pivot(index='mass', columns='speed', values='risk')
    fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, color_continuous_scale='ylorrd',
                    origin='lower', aspect='auto', template='plotly_dark')
    fig.update_layout(title='Injury Risk Heatmap (0-100)', xaxis_title='Speed (m/s)', yaxis_title='Mass (kg)')
    return fig, df2

# ------------------------- UI -------------------------

def build_sidebar():
    st.sidebar.header('Patient Parameters')
    mass = st.sidebar.slider('Body mass (kg)', 30.0, 150.0, 75.0, 1.0)
    speed = st.sidebar.slider('Walking speed (m/s)', 0.4, 2.5, 1.2, 0.05)
    cadence = st.sidebar.slider('Cadence (steps/min)', 60, 160, 110, 1)

    st.sidebar.header('Prosthetic Parameters')
    stiffness = st.sidebar.slider('Stiffness (N·m/rad)', 10, 300, 60, 1)
    damping = st.sidebar.slider('Damping (N·m·s/rad)', 0, 100, 20, 1)
    foot_type = st.sidebar.selectbox('Foot Type', ['Default', 'Energy-Return', 'Stiff Blade', 'Flexible Roll-over'])

    st.sidebar.markdown('---')
    theme = st.sidebar.selectbox('Theme', ['Dark (recommended)', 'Light'])
    if st.sidebar.button('Reset to defaults'):
        st.experimental_rerun()

    return dict(mass=mass, speed=speed, cadence=cadence,
                stiffness=stiffness, damping=damping, foot_type=foot_type, theme=theme)


def main():
    local_css()

    with st.container():
        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Medical_symbol.svg", width=64)
        with col2:
            st.title('Smart Prosthetic Gait Load Optimizer')
            st.caption('Educational simulation tool — visualize prosthetic design trade-offs')

    params = build_sidebar()

    df_moments = simulate_joint_moments(params['mass'], params['speed'], params['cadence'],
                                        params['stiffness'], params['damping'], params['foot_type'])
    metabolic = estimate_metabolic_cost(params['mass'], params['speed'], params['cadence'],
                                        params['stiffness'], params['damping'], params['foot_type'])
    risk, peak_knee, peak_hip = compute_injury_risk(df_moments, params['mass'], params['speed'], params['cadence'])

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.plotly_chart(plot_summary_bar(risk, peak_knee, peak_hip), use_container_width=True)
    with col2:
        st.plotly_chart(plot_metabolic_bar(metabolic), use_container_width=True)
    with col3:
        st.markdown('**Summary Metrics**')
        st.markdown(f"- **Estimated metabolic cost:** {metabolic:.3f} kcal/min")
        st.markdown(f"- **Injury risk (0-100):** {risk:.1f}")
        st.markdown(f"- **Peak knee moment:** {peak_knee:.1f} N·m")
        st.markdown(f"- **Peak hip moment:** {peak_hip:.1f} N·m")
        if risk < 25:
            st.success('Low estimated risk ✅')
        elif risk < 60:
            st.warning('Moderate estimated risk ⚠️')
        else:
            st.error('High estimated risk 🔥')

        gb = GridOptionsBuilder.from_dataframe(df_moments[['time_ms', 'knee_Nm', 'hip_Nm']].round(2))
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
        gb.configure_selection('single')
        gridOptions = gb.build()
        AgGrid(df_moments[['time_ms', 'knee_Nm', 'hip_Nm']].round(2), gridOptions=gridOptions, enable_enterprise_modules=False)

    # Injury Risk Heatmap + Metabolic vs Speed
    with st.expander('Explore injury risk and metabolic trends'):
        mass_range = np.linspace(max(30, params['mass'] - 20), min(150, params['mass'] + 20), 9)
        speed_range = np.linspace(max(0.4, params['speed'] - 0.6), min(2.2, params['speed'] + 0.6), 9)
        heat_fig, df_heat = plot_injury_risk_heatmap(mass_range, speed_range, params['cadence'],
                                                     params['stiffness'], params['damping'], params['foot_type'])
        st.plotly_chart(heat_fig, use_container_width=True)

        metabolic_fig, df_metabolic = plot_metabolic_vs_speed(params['mass'], params['speed'], params['cadence'],
                                                              params['stiffness'], params['damping'], params['foot_type'])
        st.plotly_chart(metabolic_fig, use_container_width=True)

    # Download outputs
    results = {
        'params': params,
        'summary': {'metabolic_kcal_min': metabolic, 'injury_risk': risk, 'peak_knee_Nm': peak_knee, 'peak_hip_Nm': peak_hip},
    }
    csv_buffer = df_moments.to_csv(index=False)
    json_buffer = json.dumps(results)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button('Download moments CSV', csv_buffer, file_name='moments.csv', mime='text/csv')
    with c2:
        st.download_button('Download summary JSON', json_buffer, file_name='summary.json', mime='application/json')
    with c3:
        st.download_button('Download metabolic trend CSV', df_metabolic.to_csv(index=False),
                           file_name='metabolic_trend.csv', mime='text/csv')


if __name__ == '__main__':
    main()
