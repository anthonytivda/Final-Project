# User's active file:

"""
Smart Prosthetic Gait Load Optimizer
Streamlit app entrypoint. Modular, well-commented, educational simulation
for exploring prosthetic stiffness/damping effects on joint loads,
metabolic cost, and injury risk trends.

Run: streamlit run .app.py

Note: This is a conceptual educational tool - not a clinical predictor.
"""

import json
from functools import lru_cache
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
	from streamlit_lottie import st_lottie
except Exception:
	# graceful fallback if lottie isn't available
	def st_lottie(o, height=200, key=None):
		return None

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from streamlit_option_menu import option_menu

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

def generate_gait_cycle(num_points=200):
	"""Return a normalized gait phase from 0 to 2*pi."""
	return np.linspace(0, 2 * np.pi, num_points)


def simulate_joint_moments(mass, speed, cadence, stiffness, damping, foot_type, num_points=200):
	"""
	Generate synthetic time-series of knee and hip moments across a gait cycle.
	These are simplified, smooth curves illustrating trends.
	Units: N·m (arbitrary scale normalized to body mass and speed)
	"""
	phase = generate_gait_cycle(num_points)

	# base moment amplitude scales with mass * speed
	base_amp = 0.8 * mass * (0.5 + speed)

	# foot_type modifies moment patterns (conceptual)
	foot_mod = {"Default": 1.0, "Energy-Return": 0.9, "Stiff Blade": 1.1, "Flexible Roll-over": 0.95}
	fmod = foot_mod.get(foot_type, 1.0)

	# stiffness increases peak moment (if too stiff, bad shock absorption)
	k_factor = 1 + (stiffness - 50) / 300.0

	# damping smooths peaks
	d_factor = 1.0 / (1.0 + damping / 50.0)

	# create knee and hip moment patterns
	knee = base_amp * fmod * k_factor * d_factor * (0.6 * np.sin(phase) + 0.4 * np.sin(2 * phase) * 0.6)
	hip = base_amp * fmod * (k_factor * 0.7 * np.cos(phase * 0.9) + 0.3 * np.sin(phase * 1.2)) * d_factor

	# add small prosthetic-specific modulation
	knee += 0.08 * stiffness * np.sin(3 * phase) / 10.0
	hip += 0.05 * damping * np.cos(2 * phase) / 10.0

	time = np.linspace(0, 1000 / cadence if cadence > 0 else 1.0, num_points)  # ms per stride approx

	return pd.DataFrame({"time_ms": time, "phase_rad": phase, "knee_Nm": knee, "hip_Nm": hip})


def estimate_metabolic_cost(mass, speed, cadence, stiffness, damping, foot_type):
	"""
	A simple parametric metabolic cost estimate (W/kg or J/s per kg).
	Not validated clinically — only conceptual trends.
	"""
	# baseline cost increases with speed squared and body mass
	base_cost = 3.5 + 0.1 * mass + 2.0 * speed ** 2

	# cadence influences economy (very low or very high cadence hurts efficiency)
	cadence_effect = 1.0 + 0.0008 * (cadence - 110) ** 2 / 1000.0

	# prosthetic penalties: extremely stiff or extremely low damping increases cost
	stiffness_penalty = 1.0 + max(0, (stiffness - 80)) / 500.0
	damping_penalty = 1.0 + max(0, (40 - damping)) / 300.0

	# foot type modifier (conceptual)
	foot_mod = {"Default": 1.0, "Energy-Return": 0.94, "Stiff Blade": 1.06, "Flexible Roll-over": 0.98}

	total = base_cost * cadence_effect * stiffness_penalty * damping_penalty * foot_mod.get(foot_type, 1.0)

	# return per-minute (kcal/min) and normalized
	kcal_per_min = total * 0.01433 * mass / 60.0  # made-up scaling for visualization
	return max(0.0, kcal_per_min)


def compute_injury_risk(df_moments, mass, speed, cadence):
	"""
	Compute a simple injury risk score from peak joint moments relative to heuristic thresholds,
	plus cadence extremes and speed. Output 0-100 risk percent.
	"""
	peak_knee = df_moments['knee_Nm'].abs().max()
	peak_hip = df_moments['hip_Nm'].abs().max()

	# thresholds scale with mass
	knee_thresh = 12.0 * mass
	hip_thresh = 18.0 * mass

	knee_ratio = peak_knee / knee_thresh
	hip_ratio = peak_hip / hip_thresh

	cadence_penalty = 0.02 * abs(cadence - 110) / 10.0
	speed_penalty = 0.05 * max(0, speed - 1.6)

	raw = (0.6 * knee_ratio + 0.4 * hip_ratio) + cadence_penalty + speed_penalty
	risk = np.clip(100 * (raw / 1.3), 0, 100)
	return float(risk), peak_knee, peak_hip


# ------------------------- Plotting Helpers -------------------------

def plot_joint_moments(df):
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['phase_rad'], y=df['knee_Nm'], mode='lines', name='Knee (N·m)', line=dict(color='#ff7f50')))
	fig.add_trace(go.Scatter(x=df['phase_rad'], y=df['hip_Nm'], mode='lines', name='Hip (N·m)', line=dict(color='#6a5acd')))
	fig.update_layout(title='Joint Moments Across Gait Phase', xaxis_title='Phase (rad)', yaxis_title='Moment (N·m)', template='plotly_dark', hovermode='x')
	return fig


def plot_metabolic_over_params(mass, speed_range, cadence, stiffness, damping, foot_type):
	# compute metabolic cost across speeds (for an interactive trend)
	speeds = np.array(speed_range)
	costs = [estimate_metabolic_cost(mass, s, cadence, stiffness, damping, foot_type) for s in speeds]
	df = pd.DataFrame({"speed_m_s": speeds, "kcal_per_min": costs})
	fig = px.line(df, x='speed_m_s', y='kcal_per_min', markers=True, title='Estimated Metabolic Cost vs Speed', template='plotly_dark')
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
	fig = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, color_continuous_scale='ylorrd', origin='lower', aspect='auto')
	fig.update_layout(title='Injury Risk Heatmap (0-100)', xaxis_title='Speed (m/s)', yaxis_title='Mass (kg)', template='plotly_dark')
	return fig, df2


# ------------------------- UI -------------------------

def build_sidebar():
	st.sidebar.header('Patient Parameters')
	mass = st.sidebar.slider('Body mass (kg)', min_value=30.0, max_value=150.0, value=75.0, step=1.0)
	speed = st.sidebar.slider('Walking speed (m/s)', min_value=0.4, max_value=2.5, value=1.2, step=0.05)
	cadence = st.sidebar.slider('Cadence (steps/min)', min_value=60, max_value=160, value=110, step=1)

	st.sidebar.header('Prosthetic Parameters')
	stiffness = st.sidebar.slider('Stiffness (N·m/rad)', min_value=10, max_value=300, value=60, step=1)
	damping = st.sidebar.slider('Damping (N·m·s/rad)', min_value=0, max_value=100, value=20, step=1)
	foot_type = st.sidebar.selectbox('Foot Type', options=['Default', 'Energy-Return', 'Stiff Blade', 'Flexible Roll-over'])

	st.sidebar.markdown('---')
	theme = st.sidebar.selectbox('Theme', options=['Dark (recommended)', 'Light'])
	if st.sidebar.button('Reset to defaults'):
		st.experimental_rerun()

	return dict(mass=mass, speed=speed, cadence=cadence, stiffness=stiffness, damping=damping, foot_type=foot_type, theme=theme)


def main():
	local_css()

	# top nav menu (small)
	with st.container():
		col1, col2 = st.columns([0.15, 0.85])
		with col1:
			st.image("https://raw.githubusercontent.com/anthonytivda/Final-Project/main/logo.png" if False else "https://upload.wikimedia.org/wikipedia/commons/8/89/Medical_symbol.svg", width=64)
		with col2:
			st.title('Smart Prosthetic Gait Load Optimizer')
			st.caption('Educational simulation tool — visualize prosthetic design trade-offs')

	params = build_sidebar()

	# run simulation
	df_moments = simulate_joint_moments(params['mass'], params['speed'], params['cadence'], params['stiffness'], params['damping'], params['foot_type'])
	metabolic = estimate_metabolic_cost(params['mass'], params['speed'], params['cadence'], params['stiffness'], params['damping'], params['foot_type'])
	risk, peak_knee, peak_hip = compute_injury_risk(df_moments, params['mass'], params['speed'], params['cadence'])

	# layout results
	left, right = st.columns([2, 1])

	with left:
		st.plotly_chart(plot_joint_moments(df_moments), use_container_width=True)
		spd_range = np.linspace(max(0.4, params['speed'] - 0.8), min(2.5, params['speed'] + 0.8), 20)
		met_fig, met_df = plot_metabolic_over_params(params['mass'], spd_range, params['cadence'], params['stiffness'], params['damping'], params['foot_type'])
		st.plotly_chart(met_fig, use_container_width=True)

	with right:
		st.markdown('**Summary Metrics**')
		st.markdown(f"- **Estimated metabolic cost:**  {metabolic:.3f} kcal/min")
		st.markdown(f"- **Injury risk (0-100):**  {risk:.1f}")
		st.markdown(f"- **Peak knee moment:**  {peak_knee:.1f} N·m")
		st.markdown(f"- **Peak hip moment:**  {peak_hip:.1f} N·m")
		# small animated indicator (emoji-based)
		if risk < 25:
			st.success('Low estimated risk ✅')
		elif risk < 60:
			st.warning('Moderate estimated risk ⚠️')
		else:
			st.error('High estimated risk 🔥')

		# show small interactive table
		gb = GridOptionsBuilder.from_dataframe(df_moments[['time_ms', 'knee_Nm', 'hip_Nm']].round(2))
		gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
		gb.configure_selection('single')
		gridOptions = gb.build()
		AgGrid(df_moments[['time_ms', 'knee_Nm', 'hip_Nm']].round(2), gridOptions=gridOptions, enable_enterprise_modules=False)

	# injury risk heatmap section
	with st.expander('Explore injury risk across mass and speed'): 
		mass_range = np.linspace(max(30, params['mass'] - 20), min(150, params['mass'] + 20), 9)
		speed_range = np.linspace(max(0.4, params['speed'] - 0.6), min(2.2, params['speed'] + 0.6), 9)
		heat_fig, df_heat = plot_injury_risk_heatmap(mass_range, speed_range, params['cadence'], params['stiffness'], params['damping'], params['foot_type'])
		st.plotly_chart(heat_fig, use_container_width=True)

	# educational panel
	with st.expander('How parameters influence gait mechanics (educational)'):
		st.markdown(
			"""
			- **Stiffness**: Higher stiffness can increase peak joint moments and reduce shock absorption, often increasing risk and discomfort.
			- **Damping**: Helps dissipate energy; too little damping can cause abrupt loading, too much can reduce efficiency.
			- **Foot Type**: Energy-return feet can lower metabolic cost but may change loading patterns; stiffer blades concentrate forces.
			- **Speed & Cadence**: Faster speeds and non-optimal cadence typically raise metabolic cost and may increase peak joint moments.
			"""
		)

	# download outputs
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
		st.download_button('Download metabolic trend CSV', met_df.to_csv(index=False), file_name='metabolic_trend.csv', mime='text/csv')


if __name__ == '__main__':
	main()

