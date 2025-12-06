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
import os


def _get_active_theme_from_config():
	"""Read `.streamlit/config.toml` and return 'dark' or 'light'."""
	try:
		cfg_path = os.path.join('.streamlit', 'config.toml')
		if not os.path.exists(cfg_path):
			return 'dark'
		with open(cfg_path, 'r', encoding='utf-8') as fh:
			txt = fh.read().lower()
		if 'base' in txt and 'light' in txt:
			return 'light'
		return 'dark'
	except Exception:
		return 'dark'

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


# ------------------------- Simulation Models -------------------------

def generate_gait_cycle(num_points=200):
	"""Return a normalized gait phase from 0 to 2*pi."""
	return np.linspace(0, 2 * np.pi, num_points)


def simulate_joint_moments(mass, speed, cadence, stiffness, damping, foot_type, foot_length=26, heel_height=2.0, toe_stiffness=30, alignment=0, prosthesis_weight=1.2, material_type='Carbon Fiber', num_points=200):
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

	# New prosthetic parameters influence factors
	foot_length_factor = 1.0 + (foot_length - 26) / 100.0  # longer foot increases moment slightly
	heel_height_factor = 1.0 + (heel_height - 2.0) / 10.0  # higher heel increases moment
	toe_stiffness_factor = 1.0 + (toe_stiffness - 30) / 200.0  # stiffer toe increases moment
	alignment_factor = 1.0 + alignment / 50.0  # misalignment increases moment
	prosthesis_weight_factor = 1.0 + (prosthesis_weight - 1.2) / 10.0  # heavier prosthesis increases moment
	material_mod = {
		'Carbon Fiber': 1.0,
		'Titanium': 1.05,
		'Aluminum': 1.03,
		'Composite': 0.98,
		'Plastic': 0.95
	}
	material_factor = material_mod.get(material_type, 1.0)

	# create knee and hip moment patterns
	total_factor = fmod * k_factor * d_factor * foot_length_factor * heel_height_factor * toe_stiffness_factor * alignment_factor * prosthesis_weight_factor * material_factor
	knee = base_amp * total_factor * (0.6 * np.sin(phase) + 0.4 * np.sin(2 * phase) * 0.6)
	hip = base_amp * total_factor * (0.7 * np.cos(phase * 0.9) + 0.3 * np.sin(phase * 1.2))

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
	More sensitive injury risk model with stronger scaling.
	- Lower joint load thresholds so realistic peaks can exceed them
	- Normalize loads, cadence, and speed into 0–2 severity scores
	- Joint loads dominate the overall risk

	Returns:
		risk (0–100), peak_knee, peak_hip
	"""
	peak_knee = float(df_moments['knee_Nm'].abs().max())
	peak_hip = float(df_moments['hip_Nm'].abs().max())

	# --- 1. Joint load thresholds (sub-linearly scaled with mass) ---
	# If thresholds scaled perfectly linearly with mass, the mass term
	# would cancel with moment scaling and risk would be insensitive to mass.
	# Use an exponent < 1 so heavier mass increases risk moderately.
	mass_scale_exponent = 0.85
	knee_thresh = 2.0 * (mass ** mass_scale_exponent)
	hip_thresh = 2.5 * (mass ** mass_scale_exponent)

	knee_ratio = peak_knee / knee_thresh
	hip_ratio = peak_hip / hip_thresh

	# Weighted combined load index
	load_index = 0.7 * knee_ratio + 0.3 * hip_ratio

	# Convert load_index → 0–2 severity:
	#   < 0.3 → basically "safe"
	#   ~1.0 → high but not extreme
	#   ≥ 1.5 → very high
	load_norm = (load_index - 0.3) / 1.2
	load_norm = float(np.clip(load_norm, 0.0, 2.0))

	# --- 2. Cadence severity (0–2) ---
	# 110 steps/min is nominal. 25 spm away ≈ severity ~1.
	cadence_dev = abs(cadence - 110)
	cadence_norm = (cadence_dev / 25.0)  # 25 spm off → 1.0
	cadence_norm = float(np.clip(cadence_norm, 0.0, 2.0))

	# --- 3. Speed severity (0–2) ---
	# 1.2 m/s (~comfortable walk) is nominal.
	# Every +0.6 m/s adds ~1 severity (so 1.8 m/s → 1.0, 2.4 m/s → ~2.0).
	speed_excess = max(0.0, speed - 1.2)
	speed_norm = speed_excess / 0.6
	speed_norm = float(np.clip(speed_norm, 0.0, 2.0))

	# --- 4. Combine into a 0–2 overall index ---
	# Joint loads dominate; cadence next; speed least.
	overall_index = (
		0.7 * load_norm +
		0.2 * cadence_norm +
		0.1 * speed_norm
	)

	# Max overall_index (when all three max out at 2) = 2.0
	# Map 0–2 → 0–100
	risk = float(np.clip((overall_index / 2.0) * 100.0, 0.0, 100.0))

	return risk, peak_knee, peak_hip


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


# ------------------------- Additional Helpers (Requested Outputs) -------------------------

def compute_peak_forces(df_moments, mass, lever_arm=0.4):
	"""Estimate peak joint forces (N) from joint moments (Nm) using an assumed lever arm.
	Also compute bodyweight multiples (force / (mass * g)).
	This is an approximation for visualization only.
	"""
	g = 9.81
	peak_knee_moment = float(df_moments['knee_Nm'].abs().max())
	peak_hip_moment = float(df_moments['hip_Nm'].abs().max())

	# avoid divide-by-zero
	lever = lever_arm if lever_arm > 0 else 0.4
	peak_knee_force = peak_knee_moment / lever
	peak_hip_force = peak_hip_moment / lever

	bw = mass * g
	peak_knee_bw = peak_knee_force / bw
	peak_hip_bw = peak_hip_force / bw

	return {
		'peak_knee_moment_Nm': peak_knee_moment,
		'peak_hip_moment_Nm': peak_hip_moment,
		'peak_knee_force_N': peak_knee_force,
		'peak_hip_force_N': peak_hip_force,
		'peak_knee_bw': peak_knee_bw,
		'peak_hip_bw': peak_hip_bw,
	}


def energy_j_per_kg_per_m(mass, speed, cadence, stiffness, damping, foot_type):
	"""Convert estimated metabolic cost (kcal/min) into J/kg/m.

	Calculation:
	  kcal/min -> J/min = kcal/min * 4184
	  distance per minute = speed (m/s) * 60
	  J per meter = (J/min) / (distance per minute)
	  J/kg/m = (J per meter) / mass
	"""
	kcal_per_min = estimate_metabolic_cost(mass, speed, cadence, stiffness, damping, foot_type)
	j_per_min = kcal_per_min * 4184.0
	distance_per_min = max(0.0001, speed) * 60.0
	j_per_m = j_per_min / distance_per_min
	j_per_kg_per_m = j_per_m / max(0.0001, mass)
	return j_per_kg_per_m


def risk_category_label(risk):
	if risk < 25:
		return 'Low', '🟢', '#2ecc71'
	elif risk < 60:
		return 'Moderate', '🟡', '#f1c40f'
	else:
		return 'High', '🔴', '#e74c3c'

def render_risk_meter(risk):
	"""Render a simple color-coded horizontal risk bar using HTML/CSS."""
	pct = float(max(0.0, min(100.0, risk)))
	theme = _get_active_theme_from_config()

	# gauge colors and template per theme
	if theme == 'light':
		template = 'plotly_white'
		bgcolor = 'rgba(0,0,0,0)'
		font_color = '#0b132b'
	else:
		template = 'plotly_dark'
		bgcolor = 'rgba(0,0,0,0)'
		font_color = '#e6eef8'

	fig = go.Figure(go.Indicator(
		mode='gauge+number',
		value=pct,
		gauge={
			'shape': 'angular',
			'axis': {'range': [0, 100], 'tickcolor': font_color},
			'bar': {'color': '#0f4c75'},
			'steps': [
				{'range': [0, 25], 'color': '#2ecc71'},
				{'range': [25, 60], 'color': '#f1c40f'},
				{'range': [60, 100], 'color': '#e74c3c'},
			],
			'threshold': {
				'value': pct,
				'line': {'color': '#000', 'width': 4},
			},
		},
		number={'suffix': '%', 'font': {'color': font_color, 'size': 20}},
		domain={'x': [0, 1], 'y': [0, 1]}
	))

	fig.update_layout(margin={'t': 10, 'b': 10, 'l': 10, 'r': 10}, template=template, paper_bgcolor=bgcolor, height=260)

	# category label under the gauge
	cat, icon, color = risk_category_label(pct)
	st.markdown(f"**Risk:** {icon} <span style='color:{color}'>{cat}</span> ({pct:.1f}%)", unsafe_allow_html=True)
	st.plotly_chart(fig, use_container_width=True)


def summarize_risk_factors(params, peaks, risk_val):
	"""Return a short text-based explanation and recommendations based on parameters and peaks."""
	msgs = []
	# stiffness/damping messages
	stiffness = params.get('stiffness', 60)
	damping = params.get('damping', 20)
	speed = params.get('speed', 1.2)

	if stiffness < 30:
		msgs.append('Stiffness is too low → increased instability risk.')
	elif stiffness > 180:
		msgs.append('Stiffness is very high → reduced shock absorption and higher peak loads.')

	if damping < 10:
		msgs.append('Low damping → abrupt loading expected; consider increasing damping.')
	elif damping > 70:
		msgs.append('High damping → may reduce energy return and increase metabolic cost.')

	# speed recommendation
	if speed > 1.8:
		msgs.append('High walking speed detected → consider reducing speed to lower joint loads.')
	elif speed < 0.6:
		msgs.append('Very slow speed → may increase instability and energy cost per distance.')

	# joint load messages
	if peaks['peak_knee_bw'] > 1.5 or peaks['peak_hip_bw'] > 1.5:
		msgs.append('High joint load detected → consider reducing walking speed or adjusting prosthetic stiffness.')

	# risk-level specific
	cat, _, _ = risk_category_label(risk_val)
	if cat == 'High':
		msgs.append('Overall predicted injury risk is High — evaluate prosthetic tuning and gait training.')
	elif cat == 'Moderate':
		msgs.append('Moderate risk — monitor symptoms and consider minor adjustments.')

	# Only add 'Low predicted risk...' if no other recommendations
	if cat == 'Low' and not msgs:
		msgs.append('Low predicted risk under current settings.')

	# deduplicate and return
	seen = set()
	final = []
	for m in msgs:
		if m not in seen:
			final.append(m)
			seen.add(m)
	return final


def plot_joint_load_vs_stiffness(mass, speed, cadence, stiffness, damping, foot_type):
	stiffs = np.linspace(max(10, stiffness - 100), min(300, stiffness + 100), 40)
	vals = []
	for s in stiffs:
		df = simulate_joint_moments(mass, speed, cadence, s, damping, foot_type)
		peaks = compute_peak_forces(df, mass)
		vals.append(peaks['peak_knee_force_N'])
	dfp = pd.DataFrame({'stiffness': stiffs, 'peak_knee_force_N': vals})
	fig = px.line(dfp, x='stiffness', y='peak_knee_force_N', title='Joint load vs Prosthetic stiffness', template='plotly_dark')
	fig.update_layout(xaxis_title='Stiffness (N·m/rad)', yaxis_title='Peak knee force (N)')
	return fig, dfp


def plot_energy_vs_speed(mass, cadence, stiffness, damping, foot_type, speed_range=None):
	if speed_range is None:
		speeds = np.linspace(0.4, 2.5, 30)
	else:
		speeds = np.array(speed_range)
	vals = [energy_j_per_kg_per_m(mass, s, cadence, stiffness, damping, foot_type) for s in speeds]
	dfe = pd.DataFrame({'speed_m_s': speeds, 'J_per_kg_per_m': vals})
	fig = px.line(dfe, x='speed_m_s', y='J_per_kg_per_m', title='Energy cost vs Walking speed', template='plotly_dark')
	fig.update_layout(xaxis_title='Speed (m/s)', yaxis_title='Energy (J/kg/m)')
	return fig, dfe


def optimize_stiffness_damping(mass, speed, cadence, stiffness, damping, foot_type, s_span=50, d_span=30, s_steps=21, d_steps=13):
	"""Simple grid search around current stiffness/damping to minimize peak knee moment (Nm).

	s_span/d_span define +/- search range around current values; s_steps/d_steps control resolution.
	Returns best_stiffness, best_damping, best_peak_knee, results_df
	"""
	s_min = max(10.0, stiffness - s_span)
	s_max = min(300.0, stiffness + s_span)
	d_min = max(0.0, damping - d_span)
	d_max = min(100.0, damping + d_span)

	s_vals = np.linspace(s_min, s_max, s_steps)
	d_vals = np.linspace(d_min, d_max, d_steps)

	rows = []
	best = (None, None, 1e12)
	for s in s_vals:
		for d in d_vals:
			df = simulate_joint_moments(mass, speed, cadence, s, d, foot_type)
			peak_knee = float(df['knee_Nm'].abs().max())
			rows.append({'stiffness': s, 'damping': d, 'peak_knee_Nm': peak_knee})
			if peak_knee < best[2]:
				best = (s, d, peak_knee)

	results = pd.DataFrame(rows)
	best_s, best_d, best_peak = best
	return best_s, best_d, best_peak, results


def design_summary_and_recommendations(df_moments, mass, speed, cadence, stiffness, damping, foot_type, foot_length=26, heel_height=2.0, toe_stiffness=30, alignment=0, prosthesis_weight=1.2, material_type='Carbon Fiber'):
	"""Compute a compact design-oriented summary table and simple recommendations.

	Returns (summary_df, recommendations_list)
	"""
	# basic stats (more intuitive / human-friendly names and units)
	peak_knee = float(df_moments['knee_Nm'].abs().max())
	peak_hip = float(df_moments['hip_Nm'].abs().max())
	mean_knee = float(df_moments['knee_Nm'].mean())
	mean_hip = float(df_moments['hip_Nm'].mean())
	rms_knee = float(np.sqrt((df_moments['knee_Nm'] ** 2).mean()))
	rms_hip = float(np.sqrt((df_moments['hip_Nm'] ** 2).mean()))
	p90_knee = float(np.percentile(df_moments['knee_Nm'].abs(), 90))
	p90_hip = float(np.percentile(df_moments['hip_Nm'].abs(), 90))
	phase_knee_peak = float(df_moments.loc[df_moments['knee_Nm'].abs().idxmax(), 'phase_rad'])
	phase_hip_peak = float(df_moments.loc[df_moments['hip_Nm'].abs().idxmax(), 'phase_rad'])

	# forces (approx) and bodyweight multiples for clearer designer interpretation
	lever_arm = 0.4
	forces = compute_peak_forces(df_moments, mass, lever_arm=lever_arm)
	peak_knee_force = forces['peak_knee_force_N']
	peak_hip_force = forces['peak_hip_force_N']
	peak_knee_bw = forces['peak_knee_bw']
	peak_hip_bw = forces['peak_hip_bw']

	# normalize by body mass for designer perspective (moments per kg)
	peak_knee_per_kg = peak_knee / max(0.0001, mass)
	peak_hip_per_kg = peak_hip / max(0.0001, mass)

	# convert phase (radians) into percent of gait cycle for easier reading
	phase_knee_pct = (phase_knee_peak / (2.0 * np.pi)) * 100.0
	phase_hip_pct = (phase_hip_peak / (2.0 * np.pi)) * 100.0

	summary = pd.DataFrame({
		'metric': [
			'Peak moment (N·m)',
			'Peak force (N)',
			'Peak force (×BW)',
			'Mean moment (N·m)',
			'RMS moment (N·m)',
			'90th percentile moment (N·m)',
			'Phase of peak (% gait)',
			'Peak moment per kg (N·m/kg)'
		],
		'knee': [
			peak_knee,
			peak_knee_force,
			peak_knee_bw,
			mean_knee,
			rms_knee,
			p90_knee,
			phase_knee_pct,
			peak_knee_per_kg
		],
		'hip': [
			peak_hip,
			peak_hip_force,
			peak_hip_bw,
			mean_hip,
			rms_hip,
			p90_hip,
			phase_hip_pct,
			peak_hip_per_kg
		]
	})

	# sensitivity to stiffness (finite difference)
	delta = max(5.0, stiffness * 0.1)
	s_low = max(10.0, stiffness - delta)
	s_high = min(300.0, stiffness + delta)
	df_low = simulate_joint_moments(mass, speed, cadence, s_low, damping, foot_type, foot_length, heel_height, toe_stiffness, alignment, prosthesis_weight, material_type)
	df_high = simulate_joint_moments(mass, speed, cadence, s_high, damping, foot_type, foot_length, heel_height, toe_stiffness, alignment, prosthesis_weight, material_type)
	pk_low = float(df_low['knee_Nm'].abs().max())
	pk_high = float(df_high['knee_Nm'].abs().max())
	stiffness_slope = (pk_high - pk_low) / (s_high - s_low) if (s_high - s_low) != 0 else 0.0

	recs = []
	# simple heuristics
	if peak_knee_per_kg > 1.5 or peak_hip_per_kg > 1.5:
		recs.append('High joint loads relative to body mass — consider reducing walking speed or lowering stiffness.')
	# Only show if effect is more pronounced
	if stiffness_slope > 0.2:
		recs.append(f'Increasing stiffness increases peak knee moment by ~{stiffness_slope:.2f} Nm per unit stiffness; reducing stiffness may lower peak loads.')
	elif stiffness_slope < -0.2:
		recs.append('Increasing stiffness reduces peak knee moment in this regime; increasing stiffness may improve load distribution.')

	# energy-related note
	energy = energy_j_per_kg_per_m(mass, speed, cadence, stiffness, damping, foot_type)
	if energy > 0.8:
		recs.append('Energy cost is relatively high — consider energy-return foot designs or tuning damping.')

	# comfort/stability hints
	if stiffness < 30:
		recs.append('Low stiffness may increase instability — consider a stiffer foot if stability is a priority.')
	if damping < 10:
		recs.append('Low damping may cause abrupt loading — consider increasing damping to smooth gait loads.')

	# New prosthetic variable recommendations
	if foot_length > 32:
		recs.append('Foot length is above typical range — may affect toe clearance and gait symmetry.')
	if heel_height > 4.0:
		recs.append('High heel height may increase forefoot loading and alter gait mechanics.')
	if toe_stiffness > 80:
		recs.append('Toe stiffness is high — may reduce push-off efficiency and comfort.')
	if abs(alignment) > 7:
		recs.append('Significant alignment deviation detected — may increase risk of joint overload or instability.')
	if prosthesis_weight > 2.5:
		recs.append('Prosthesis weight is high — may increase energy cost and reduce comfort for long-term use.')
	if material_type == 'Plastic':
		recs.append('Plastic material selected — may have lower durability and energy return compared to composites or metals.')
	elif material_type == 'Carbon Fiber':
		recs.append('Carbon fiber material selected — offers high energy return and low weight, suitable for active users.')
	elif material_type == 'Titanium':
		recs.append('Titanium material selected — durable and lightweight, good for long-term use.')
	elif material_type == 'Aluminum':
		recs.append('Aluminum material selected — lightweight but less durable than titanium.')
	elif material_type == 'Composite':
		recs.append('Composite material selected — balanced energy return and comfort.')

	if not recs:
		recs.append('No major design concerns detected for current settings.')

	return summary, recs


# ------------------------- UI -------------------------

def build_sidebar():
	st.sidebar.header('Patient Parameters')

	# If a reset was requested on the previous run, apply defaults before creating widgets
	if st.session_state.get('reset_requested', False):
		st.session_state['mass'] = 75.0
		st.session_state['speed'] = 1.2
		st.session_state['cadence'] = 110
		st.session_state['stiffness'] = 60
		st.session_state['damping'] = 20
		st.session_state['foot_type'] = 'Default'
		st.session_state['reset_requested'] = False

	# If an apply_suggestion flag is present, apply suggested_values before widget creation
	if st.session_state.get('apply_suggestion', False):
		sv = st.session_state.get('suggested_values', {})
		if 'stiffness' in sv:
			st.session_state['stiffness'] = sv['stiffness']
		if 'damping' in sv:
			st.session_state['damping'] = sv['damping']
		st.session_state['apply_suggestion'] = False
		# clear suggested_values after applying
		st.session_state['suggested_values'] = {}

	mass = st.sidebar.slider('Body mass (kg)', min_value=30.0, max_value=150.0, value=st.session_state['mass'] if 'mass' in st.session_state else 75.0, step=1.0, key='mass')
	speed = st.sidebar.slider('Walking speed (m/s)', min_value=0.4, max_value=2.5, value=st.session_state['speed'] if 'speed' in st.session_state else 1.2, step=0.05, key='speed')
	cadence = st.sidebar.slider('Cadence (steps/min)', min_value=60, max_value=160, value=st.session_state['cadence'] if 'cadence' in st.session_state else 110, step=1, key='cadence')

	st.sidebar.header('Prosthetic Parameters')
	stiffness = st.sidebar.slider('Stiffness (N·m/rad)', min_value=10, max_value=300, value=st.session_state.get('stiffness', 60), step=1, key='stiffness')
	damping = st.sidebar.slider('Damping (N·m·s/rad)', min_value=0, max_value=100, value=st.session_state.get('damping', 20), step=1, key='damping')
	foot_length = st.sidebar.slider('Foot Length (cm)', min_value=20, max_value=35, value=st.session_state.get('foot_length', 26), step=1, key='foot_length')
	heel_height = st.sidebar.slider('Heel Height (cm)', min_value=0.0, max_value=5.0, value=float(st.session_state.get('heel_height', 2.0)), step=0.1, key='heel_height')
	toe_stiffness = st.sidebar.slider('Toe Stiffness (N·m/rad)', min_value=5, max_value=100, value=st.session_state.get('toe_stiffness', 30), step=1, key='toe_stiffness')
	alignment = st.sidebar.slider('Alignment (deg)', min_value=-10, max_value=10, value=st.session_state.get('alignment', 0), step=1, key='alignment')
	prosthesis_weight = st.sidebar.slider('Prosthesis Weight (kg)', min_value=0.5, max_value=3.5, value=st.session_state.get('prosthesis_weight', 1.2), step=0.1, key='prosthesis_weight')
	material_type = st.sidebar.selectbox('Material Type', options=['Carbon Fiber', 'Titanium', 'Aluminum', 'Composite', 'Plastic'], index=0, key='material_type')

	# More intuitive foot type display labels mapped to internal keys used by simulation
	display_to_internal = {
		'Standard (everyday)': 'Default',
		'Energy-return (springy)': 'Energy-Return',
		'Stiff blade (performance)': 'Stiff Blade',
		'Flexible roll-over (comfort)': 'Flexible Roll-over',
	}
	internal_to_display = {v: k for k, v in display_to_internal.items()}

	current_internal = st.session_state.get('foot_type', 'Default')
	current_label = internal_to_display.get(current_internal, list(display_to_internal.keys())[0])
	display_options = list(display_to_internal.keys())
	selected_label = st.sidebar.selectbox('Foot Type', options=display_options, index=display_options.index(current_label), key='foot_type_label')
	foot_type = display_to_internal.get(selected_label, 'Default')

	st.sidebar.markdown('---')

	if st.sidebar.button('Reset to defaults'):
		# mark for reset and request a rerun so defaults are applied before widget creation
		st.session_state['reset_requested'] = True
		try:
			if hasattr(st, 'experimental_rerun'):
				st.experimental_rerun()
			elif hasattr(st, 'rerun'):
				st.rerun()
			else:
				st.info('Defaults queued. Please reload the app to see defaults.')
		except Exception:
			st.info('Defaults queued. Please reload the app to see defaults.')

	return dict(
		mass=mass,
		speed=speed,
		cadence=cadence,
		stiffness=stiffness,
		damping=damping,
		foot_type=foot_type,
		foot_length=foot_length,
		heel_height=heel_height,
		toe_stiffness=toe_stiffness,
		alignment=alignment,
		prosthesis_weight=prosthesis_weight,
		material_type=material_type
	)


def main():
	# top nav menu (small)
	with st.container():
		st.title('Smart Prosthetic Gait Load Optimizer')
		st.caption('Educational simulation tool — visualize prosthetic design trade-offs')
	params = build_sidebar()

	# no custom CSS — use Streamlit's default theme

	# run simulation
	df_moments = simulate_joint_moments(
		params['mass'], params['speed'], params['cadence'], params['stiffness'], params['damping'], params['foot_type'],
		foot_length=params['foot_length'], heel_height=params['heel_height'], toe_stiffness=params['toe_stiffness'], alignment=params['alignment'], prosthesis_weight=params['prosthesis_weight'], material_type=params['material_type']
	)
	metabolic = estimate_metabolic_cost(params['mass'], params['speed'], params['cadence'], params['stiffness'], params['damping'], params['foot_type'])
	risk, peak_knee, peak_hip = compute_injury_risk(df_moments, params['mass'], params['speed'], params['cadence'])

	# layout results
	left, right = st.columns([2, 1])

	with left:

		# Primary outputs (large metrics / summaries)
		peaks = compute_peak_forces(df_moments, params['mass'], lever_arm=0.4)
		energy_jkgm = energy_j_per_kg_per_m(params['mass'], params['speed'], params['cadence'], params['stiffness'], params['damping'], params['foot_type'])
		risk_val = risk
		cat = risk_category_label(risk_val)

		
		col_a, col_b, col_c = st.columns(3)


		# --- Make primary outputs font smaller for compact view ---
		st.markdown("""
			<style>
			div[data-testid='metric-container'] {
				font-size: 0.85em !important;
				min-height: 60px;
			}
			div[data-testid='metric-container'] label, div[data-testid='metric-container'] span {
				font-size: 0.85em !important;
			}
			div[data-testid='metric-container'] [data-testid='stMetricValue'] {
				font-size: 1.1em !important;
			}
			</style>
		""", unsafe_allow_html=True)

		# Move recommendations above risk meter

		st.markdown('**Contributing risk factors & recommendations**')
		recs = summarize_risk_factors(params, peaks, risk_val)
		_, design_recs = design_summary_and_recommendations(
			df_moments,
			params['mass'], params['speed'], params['cadence'], params['stiffness'], params['damping'], params['foot_type'],
			params['foot_length'], params['heel_height'], params['toe_stiffness'], params['alignment'], params['prosthesis_weight'], params['material_type']
		)
		all_recs = recs + design_recs
		for r in all_recs:
			st.markdown(f"- {r}")

		render_risk_meter(risk_val)

		st.markdown('---')
		st.markdown("<h4 style='margin-bottom:0.2em'>Primary Outputs</h4>", unsafe_allow_html=True)
		st.caption("Key metrics for current settings. Hover for details.")
		st.markdown('')
		risk_cat, risk_icon, risk_color = risk_category_label(risk_val)
		col_knee, col_hip, col_misc = st.columns([1,1,1])
		with col_knee:
			st.markdown('🦵 **Knee**', unsafe_allow_html=True)
			st.metric(
				label="Peak Force",
				value=f"{peaks['peak_knee_force_N']:.1f} N",
				help="Maximum estimated knee joint force during gait cycle"
			)
			st.metric(
				label="Peak Load (×BW)",
				value=f"{peaks['peak_knee_bw']:.2f} ×BW",
				help="Peak knee force as a multiple of bodyweight"
			)
			st.metric(
				label="Peak Moment",
				value=f"{peaks['peak_knee_moment_Nm']:.1f} N·m",
				help="Maximum estimated knee joint moment during gait cycle"
			)
		with col_hip:
			st.markdown('🦶 **Hip**', unsafe_allow_html=True)
			st.metric(
				label="Peak Force",
				value=f"{peaks['peak_hip_force_N']:.1f} N",
				help="Maximum estimated hip joint force during gait cycle"
			)
			st.metric(
				label="Peak Load (×BW)",
				value=f"{peaks['peak_hip_bw']:.2f} ×BW",
				help="Peak hip force as a multiple of bodyweight"
			)
			st.metric(
				label="Peak Moment",
				value=f"{peaks['peak_hip_moment_Nm']:.1f} N·m",
				help="Maximum estimated hip joint moment during gait cycle"
			)
		with col_misc:
			st.markdown('⚡ **Other**', unsafe_allow_html=True)
			st.metric(
				label="Energy Cost",
				value=f"{energy_jkgm:.3f} J/kg/m",
				help="Estimated energy cost per meter walked (Joules per kg per meter)"
			)
			st.metric(
				label=f"Injury Risk {risk_icon}",
				value=f"{risk_cat} ({risk_val:.1f}%)",
				help=f"Predicted injury risk category and score (higher = more risk)"
			)
		st.markdown('---')

		# ...existing code...

		# ...existing code...


	with right:
		# Custom graph: user selects X and Y variables
		import plotly.express as px
		st.markdown("### Explore Prosthetic Design Relationships")
		param_options = {
			'Stiffness (N·m/rad)': ('stiffness', np.linspace(10, 300, 30)),
			'Damping (N·m·s/rad)': ('damping', np.linspace(0, 100, 30)),
			'Foot Length (cm)': ('foot_length', np.linspace(20, 35, 30)),
			'Heel Height (cm)': ('heel_height', np.linspace(0, 5, 30)),
			'Toe Stiffness (N·m/rad)': ('toe_stiffness', np.linspace(5, 100, 30)),
			'Alignment (deg)': ('alignment', np.linspace(-10, 10, 30)),
			'Prosthesis Weight (kg)': ('prosthesis_weight', np.linspace(0.5, 3.5, 30)),
		}
		output_options = {
			'Peak Knee Force (N)': 'peak_knee_force_N',
			'Peak Hip Force (N)': 'peak_hip_force_N',
			'Energy Cost (J/kg/m)': 'energy_cost',
			'Peak Joint Moment (N·m)': 'peak_joint_moment',
		}
		x_label = st.selectbox('X Axis Variable', list(param_options.keys()))
		y_label = st.selectbox('Y Axis Output', list(output_options.keys()))
		x_key, x_range = param_options[x_label]
		y_key = output_options[y_label]
		y_vals = []
		for x in x_range:
			args = dict(params)
			args[x_key] = x
			df = simulate_joint_moments(
				args['mass'], args['speed'], args['cadence'], args['stiffness'], args['damping'], args['foot_type'],
				foot_length=args['foot_length'], heel_height=args['heel_height'], toe_stiffness=args['toe_stiffness'], alignment=args['alignment'], prosthesis_weight=args['prosthesis_weight'], material_type=args['material_type']
			)
			peaks = compute_peak_forces(df, args['mass'])
			if y_key == 'peak_knee_force_N':
				y_vals.append(peaks['peak_knee_force_N'])
			elif y_key == 'peak_hip_force_N':
				y_vals.append(peaks['peak_hip_force_N'])
			elif y_key == 'energy_cost':
				energy = energy_j_per_kg_per_m(args['mass'], args['speed'], args['cadence'], args['stiffness'], args['damping'], args['foot_type'])
				# Optionally, scale energy by prosthesis weight factor
				energy *= (1.0 + (args['prosthesis_weight'] - 1.2) / 10.0)
				y_vals.append(energy)
			elif y_key == 'peak_joint_moment':
				y_vals.append(float(df['knee_Nm'].abs().max() + df['hip_Nm'].abs().max()))
		df_custom = pd.DataFrame({x_key: x_range, y_key: y_vals})
		fig = px.line(df_custom, x=x_key, y=y_key, title=f'{y_label} vs {x_label}', template='plotly_dark')
		fig.add_vline(x=params[x_key], line_dash='dash', line_color='#0f4c75', annotation_text='Current', annotation_position='top')
		fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
		st.plotly_chart(fig, use_container_width=True)
		# Dynamic description for graph relevance
		relevance_map = {
			('Stiffness (N·m/rad)', 'Peak Knee Force (N)'): "Stiffness affects shock absorption and joint loading. Higher stiffness can increase peak knee force, impacting comfort and injury risk.",
			('Stiffness (N·m/rad)', 'Peak Hip Force (N)'): "Stiffness influences force transmission to the hip. Balancing stiffness is key for both knee and hip protection.",
			('Stiffness (N·m/rad)', 'Energy Cost (J/kg/m)'): "Increasing stiffness may raise energy cost due to reduced shock absorption, affecting user fatigue.",
			('Stiffness (N·m/rad)', 'Peak Joint Moment (N·m)'): "Stiffness directly impacts joint moments; optimal tuning can reduce overload risk.",

			('Damping (N·m·s/rad)', 'Peak Knee Force (N)'): "Damping smooths force peaks. Too little can cause abrupt loading, too much may reduce efficiency and increase knee force.",
			('Damping (N·m·s/rad)', 'Peak Hip Force (N)'): "Damping affects energy dissipation and hip force transmission, influencing comfort and safety.",
			('Damping (N·m·s/rad)', 'Energy Cost (J/kg/m)'): "Higher damping can increase energy cost by reducing gait efficiency.",
			('Damping (N·m·s/rad)', 'Peak Joint Moment (N·m)'): "Damping modulates joint moments, balancing comfort and risk of overload.",

			('Foot Length (cm)', 'Peak Knee Force (N)'): "Longer feet may improve stability but can increase knee force due to altered gait mechanics.",
			('Foot Length (cm)', 'Peak Hip Force (N)'): "Foot length affects hip force by changing stride and leverage.",
			('Foot Length (cm)', 'Energy Cost (J/kg/m)'): "Longer feet may increase energy cost due to less efficient gait.",
			('Foot Length (cm)', 'Peak Joint Moment (N·m)'): "Foot length influences joint moments through leverage and stride changes.",

			('Heel Height (cm)', 'Peak Knee Force (N)'): "Higher heel height increases forefoot loading and knee force, impacting comfort and risk.",
			('Heel Height (cm)', 'Peak Hip Force (N)'): "Heel height alters hip force by changing gait posture and loading patterns.",
			('Heel Height (cm)', 'Energy Cost (J/kg/m)'): "High heels may increase energy cost due to less efficient gait mechanics.",
			('Heel Height (cm)', 'Peak Joint Moment (N·m)'): "Heel height affects joint moments by shifting loading patterns during stance and push-off.",

			('Toe Stiffness (N·m/rad)', 'Peak Knee Force (N)'): "Toe stiffness affects push-off and force transmission. Excessive stiffness may increase knee loading.",
			('Toe Stiffness (N·m/rad)', 'Peak Hip Force (N)'): "Toe stiffness influences hip force during push-off and propulsion.",
			('Toe Stiffness (N·m/rad)', 'Energy Cost (J/kg/m)'): "Stiffer toes may increase energy cost by reducing push-off efficiency.",
			('Toe Stiffness (N·m/rad)', 'Peak Joint Moment (N·m)'): "Toe stiffness modulates joint moments during late stance and push-off.",

			('Alignment (deg)', 'Peak Knee Force (N)'): "Alignment deviations can increase knee force, raising risk of overload and instability.",
			('Alignment (deg)', 'Peak Hip Force (N)'): "Poor alignment increases hip force and risk of joint overload.",
			('Alignment (deg)', 'Energy Cost (J/kg/m)'): "Misalignment may increase energy cost due to inefficient gait mechanics.",
			('Alignment (deg)', 'Peak Joint Moment (N·m)'): "Alignment deviations can dramatically increase joint moments, raising risk of overload and instability.",

			('Prosthesis Weight (kg)', 'Peak Knee Force (N)'): "Heavier prostheses may increase knee force, affecting comfort and injury risk.",
			('Prosthesis Weight (kg)', 'Peak Hip Force (N)'): "Prosthesis weight impacts hip force, especially during swing phase.",
			('Prosthesis Weight (kg)', 'Energy Cost (J/kg/m)'): "Heavier prostheses increase energy cost, affecting user fatigue and long-term comfort.",
			('Prosthesis Weight (kg)', 'Peak Joint Moment (N·m)'): "Prosthesis weight can increase joint moments, especially during swing and stance transitions.",
		}
		desc = relevance_map.get((x_label, y_label), f"This graph shows how changes in {x_label.lower()} affect {y_label.lower()}. Use this relationship to guide prosthetic selection and tuning for your patient's needs.")
		st.caption(f"{y_label} vs {x_label}: {desc} Use the dashed line to see your current setting.")

	# injury risk heatmap section (secondary plots are shown in the right column now)

	# educational panel
	with st.expander('How parameters influence gait mechanics (educational)'):
		st.markdown(
			"""
			- **Stiffness**: Higher stiffness increases peak joint moments and reduces shock absorption, raising risk and discomfort.
			- **Damping**: Dissipates energy; low damping causes abrupt loading, high damping can reduce efficiency.
			- **Foot Type**:
				- *Standard (everyday)*: Balanced for daily use, moderate energy return and comfort.
				- *Energy-return (springy)*: Lowers metabolic cost, improves push-off, but may change force distribution and stability.
				- *Stiff blade (performance)*: Maximizes energy return and propulsion, but concentrates forces and may reduce comfort for casual users.
				- *Flexible roll-over (comfort)*: Enhances shock absorption and comfort, reduces peak forces, but may increase energy cost for active users.
			- **Speed & Cadence**: Faster speeds and non-optimal cadence raise metabolic cost and joint moments.
			- **Foot Length**: Longer feet may improve stability but can affect toe clearance and gait symmetry.
			- **Heel Height**: Higher heels increase forefoot loading and alter gait mechanics.
			- **Toe Stiffness**: Stiffer toes reduce push-off efficiency and comfort, but may aid propulsion for some users.
			- **Alignment**: Poor alignment increases risk of joint overload and instability.
			- **Prosthesis Weight**: Heavier prostheses increase energy cost and may reduce comfort, especially for long-term use.
			- **Material Type**:
				- *Carbon Fiber*: High energy return, very lightweight, ideal for active users and performance.
				- *Titanium*: Extremely durable, lightweight, excellent for long-term use and reliability.
				- *Aluminum*: Lightweight, less durable than titanium, suitable for moderate activity levels.
				- *Composite*: Balanced energy return and comfort, good for a wide range of users.
				- *Plastic*: Lower durability and energy return, best for temporary or low-activity use.
			"""
		)

	# download outputs
	results = {
		'params': params,
		'summary': {'metabolic_kcal_min': metabolic, 'injury_risk': risk, 'peak_knee_Nm': peak_knee, 'peak_hip_Nm': peak_hip},
	}

	csv_buffer = df_moments.to_csv(index=False)
	json_buffer = json.dumps(results)

	# Generate metabolic trend DataFrame for download
	_, dfe = plot_metabolic_over_params(
		params['mass'], np.linspace(0.4, 2.5, 30), params['cadence'], params['stiffness'], params['damping'], params['foot_type']
	)

	c1, c2, c3 = st.columns(3)
	with c1:
		st.download_button('Download moments CSV', csv_buffer, file_name='moments.csv', mime='text/csv')
	with c2:
		st.download_button('Download summary JSON', json_buffer, file_name='summary.json', mime='application/json')
	with c3:
		st.download_button('Download metabolic trend CSV', dfe.to_csv(index=False), file_name='metabolic_trend.csv', mime='text/csv')


if __name__ == '__main__':
	main()

