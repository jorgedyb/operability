#!/usr/bin/env python3
"""
Streamlit App: RAO Operability Dashboard
- Compute operability metrics on demand
- Sidebar: Limits and offsets + calculate button
- Main: switch among plots without recalculation
- Contour plots for each draft
- Cache and reuse previously computed results (per configuration)
- Progress bar for calculations
Parallelized using multiple CPU cores
"""
import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import re
import joblib
from pathlib import Path
from collections import defaultdict
import concurrent.futures

import logging

logging.getLogger("streamlit.runtime.scriptrunner_utils").setLevel(logging.ERROR)
# or logging.CRITICAL to hide absolutely everything


# Constants
DATA_DIR = 'rao_outputs_fine'
RESULTS_DIR = 'Results - Operability - 20 workers'
os.makedirs(RESULTS_DIR, exist_ok=True)

HULL_PATTERN = re.compile(
    r"(?P<hull>[A-Za-z]+)_L(?P<L>\d+(?:\.\d+)?)_B(?P<B>\d+(?:\.\d+)?)_T(?P<T>\d+(?:\.\d+)?)_rao\.csv",
    re.I,
)

# Helper functions
def load_rao_data(df: pd.DataFrame):
    df['RAO_complex'] = df['RAO'].apply(lambda s: complex(str(s).strip('() ')))
    df = df.sort_values('omega')
    rao = defaultdict(dict)
    for (dof, hdg), grp in df.groupby(['radiating_dof', 'wave_direction']):
        rao[dof.strip().lower()][float(hdg)] = grp['RAO_complex'].to_numpy()
    return rao, np.sort(df['omega'].unique())

def combined_z_motion(rao_data, hdg, x=0.0, y=0.0):
    """Compute vertical motion RAO at gangway tip using offset (x, y)."""
    eta3 = rao_data.get('heave', {}).get(hdg)
    eta4 = rao_data.get('roll', {}).get(hdg)
    eta5 = rao_data.get('pitch', {}).get(hdg)

    if eta3 is None or eta4 is None or eta5 is None:
        return None  # or raise an error

    # Tip motion RAO = heave - x*pitch + y*roll
    return eta3 - x * eta5 + y * eta4


def jonswap(omega, Hs, Tp):
    g = 9.81
    gamma = 3.3
    alpha = 5.061 * (Hs**2 / Tp**4) * (1 - 0.287 * np.log(gamma))  # From VERES
    wp = 2 * np.pi / Tp
    sigma = np.where(omega < wp, 0.07, 0.09)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        S_pm = alpha * g**2 * omega**-5 * np.exp(-1.25 * (wp / omega)**4)
        r = np.exp(-((omega - wp)**2) / (2 * sigma**2 * wp**2))
        S = S_pm * gamma**r

    return np.nan_to_num(S)

def rms_motion(rao, omega, S):
    val = np.trapz(np.abs(rao)**2 * S, omega)
    return float(np.sqrt(val)) if val >= 0 else 0.0

def proc_config(cfg, df_scatter, limits_center, limits_offset, offsets, scatter_key, limits_key):
    """Compute operability for one hull configuration, with per-config caching."""
    L, B, T, hull = cfg['L'], cfg['B'], cfg['T'], cfg['hull']
    cfg_key = f"{hull}_L{L}_B{B}_T{T}"
    cache_path = Path(RESULTS_DIR) / f"{scatter_key}_{limits_key}_{cfg_key}_oper.pkl"
    if cache_path.exists():
        cfg_cached, joint, rms, annual = joblib.load(cache_path)

        # —— auto-upgrade old pickles ——
        if 'hull' not in cfg_cached:
            cfg_cached['hull'] = hull          # add the missing key
            joblib.dump((cfg_cached, joint, rms, annual), cache_path)

        return (cfg_cached, joint, rms, annual)
    df_rao = pd.read_csv(Path(DATA_DIR) / cfg['file'])
    rao_data, omega = load_rao_data(df_rao)
    joint, rms_list = [], []
    for hdg in sorted(rao_data['heave']):
        P_ok = 0.0
        sum_r = 0.0
        for Hs, row in df_scatter.iterrows():
            for Tp, p in row.items():
                if p <= 0: continue
                S = jonswap(omega, Hs, Tp)
                ok = True
                for dof, lim in limits_center.items():
                    if dof not in rao_data or hdg not in rao_data[dof]:
                        st.warning(f"Missing RAO data for DOF '{dof}' at heading {hdg}°")
                        ok = False
                        break
                    if dof == 'heave' and any((xo, yo)):  # only apply offset if x/y != 0
                        arr = combined_z_motion(rao_data, hdg, xo, yo)
                    else:
                        arr = rao_data[dof][hdg]

                    if arr is None:
                        ok = False
                        break

                    lim_use = limits_offset.get(dof, lim)

                    if rms_motion(arr, omega, S) > lim_use:
                        ok = False
                        break
                if ok:
                    P_ok += p
                sum_r += p * rms_motion(rao_data['heave'][hdg], omega, S)
        # P_ok now a fraction <=1
        joint.append((hdg * 180 / math.pi, P_ok))
        rms_list.append((hdg * 180 / math.pi, sum_r))
        #print(rms_list[-1]) 
    # annual days = mean fraction * 365
    calm_prob = df_scatter.loc[df_scatter.index <= 1.0].values.sum()
    st.write(f"Calm-sea probability = {calm_prob:.2%}")
    annual = np.mean([v for _, v in joint]) * 365
    result = (cfg, joint, rms_list, annual)
    joblib.dump(result, cache_path)
    return result

def compute_all(scatter_path, limits_center, limits_offset, offsets, max_workers):
    try:
        df_scatter = pd.read_excel(scatter_path, index_col=0)
    except Exception as e:
        st.error(f"Error reading scatter file. Close it and retry. {e}")
        return [], pd.DataFrame()
    df_scatter.columns = pd.to_numeric(df_scatter.columns, errors='coerce')
    df_scatter = df_scatter.dropna(axis=1)
    # normalize to true probabilities
    total_prob = df_scatter.values.sum()
    #print(f"Total probability in scatter: {total_prob:.4f}")
    df_scatter = df_scatter / total_prob

    scatter_key = Path(scatter_path).stem
    limits_key = f"H{limits_center['heave']}_R{math.degrees(limits_center['roll'])}_P{math.degrees(limits_center['pitch'])}_OH{limits_offset.get('heave',0)}"

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('_rao.csv')]
    configs = []
    for f in sorted(files):
        m = HULL_PATTERN.match(f)
        if m:
            configs.append({
                'hull': m['hull'].lower(),
                'L':    float(m['L']),     # <— use the name
                'B':    float(m['B']),
                'T':    float(m['T']),
                'file': f,
            })

    total = len(configs)
    progress = st.sidebar.progress(0)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(proc_config, cfg, df_scatter, limits_center,
                                   limits_offset, offsets, scatter_key, limits_key)
                   for cfg in configs]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(fut.result())
            progress.progress(i / total)

    df_ops = pd.DataFrame([{
        'hull':   r[0]['hull'],      
        'L':      r[0]['L'],
        'B':      r[0]['B'],
        'T':      r[0]['T'],
        'annual': r[3],
    } for r in results])
    summary_csv = Path(RESULTS_DIR) / f"{scatter_key}_{limits_key}_summary.csv"
    df_ops.to_csv(summary_csv, index=False)
    summary_pkl = Path(RESULTS_DIR) / f"{scatter_key}_{limits_key}_summary.pkl"
    joblib.dump((results, df_ops), summary_pkl)

    results.sort(key=lambda x: (x[0]['hull'], x[0]['T'], x[0]['L'], x[0]['B']))
    return results, df_ops

# ------------------------------------------------------------------
# === Helpers for tornado (single-hull sensitivity) ================
# ------------------------------------------------------------------
def single_config_oper(cfg, df_scatter, limits_c, limits_off,
                       omega, rao_data):
    """Annual operable days for ONE hull / ONE limits set."""
    P_ok = 0.0
    for hdg in sorted(rao_data['heave']):
        for Hs, row in df_scatter.iterrows():
            for Tp, p in row.items():
                if p <= 0:
                    continue
                S = jonswap(omega, Hs, Tp)
                ok = True
                for dof, lim in limits_c.items():
                    lim_use = limits_off.get(dof, lim)

                    if dof == 'heave' and any((xo, yo)):
                        arr = combined_z_motion(rao_data, hdg, xo, yo)
                    else:
                        arr = rao_data[dof][hdg]

                    if arr is None:
                        ok = False
                        break

                    if rms_motion(arr, omega, S) > lim_use:
                        ok = False
                        break

                if ok:
                    P_ok += p
    # divide by headings, then scale to days / yr
    return (P_ok / len(rao_data['heave'])) * 365.0


def build_tornado(cfg, df_scatter, base_c, base_off):
    """Return baseline days and list of (label, low, high) bars."""
    # Load RAO once
    df_rao         = pd.read_csv(Path(DATA_DIR) / cfg['file'])
    rao_data, omg  = load_rao_data(df_rao)

    base_days      = single_config_oper(cfg, df_scatter,
                                        base_c, base_off,
                                        omg, rao_data)

    sweep = {
        'Heave lim':    ('heave',      0.8, 1.2),
        'Roll lim':     ('roll',       0.8, 1.2),
        'Pitch lim':    ('pitch',      0.8, 1.2),
        'Heave offset': ('heave_off',  0.8, 1.2),
    }
    bars = []
    for label, (key, lo_fac, hi_fac) in sweep.items():
        # copy dicts so we can mutate
        lim_c   = base_c.copy()
        lim_off = base_off.copy()

        if key == 'heave_off':                 # vary offset version
            lim_off['heave'] = base_off['heave'] * lo_fac
            low_days  = single_config_oper(cfg, df_scatter,
                                           base_c, lim_off,
                                           omg, rao_data)

            lim_off['heave'] = base_off['heave'] * hi_fac
            high_days = single_config_oper(cfg, df_scatter,
                                           base_c, lim_off,
                                           omg, rao_data)
        else:                                  # vary centre limit
            lim_c[key] = base_c[key] * lo_fac
            low_days   = single_config_oper(cfg, df_scatter,
                                            lim_c, base_off,
                                            omg, rao_data)

            lim_c[key] = base_c[key] * hi_fac
            high_days  = single_config_oper(cfg, df_scatter,
                                            lim_c, base_off,
                                            omg, rao_data)

        bars.append((label, low_days, high_days))

    return base_days, bars




# Streamlit App
st.set_page_config(layout='wide')
st.title('RAO Operability Dashboard')
if 'plot_configs' not in st.session_state:
    st.session_state.plot_configs = []  # will hold tuples of (hull, L, B, T)

# Sidebar inputs
st.sidebar.header('Data & Scatter')
default_sc = Path('scatter-haltenbanken.xlsx')
scatter_path = default_sc if default_sc.exists() else st.sidebar.file_uploader('Scatter Excel', type=['xlsx','xls'])

st.sidebar.header('Limits & Offsets')
safety = st.sidebar.number_input("Safety Factor (×)", 1.0, 2.0, 1.2, 0.05)
raw_off_h = st.sidebar.number_input('Offset heave lim (m)', value=3.0, min_value=0.0, max_value=10.0, step=0.5)
raw_h = st.sidebar.number_input('Heave lim (m)', value=3.0, min_value=0.0, max_value=10.0, step=0.5)
raw_r = st.sidebar.number_input('Roll lim (deg)', value=3.0, min_value=0.0, max_value=90.0, step=0.5)
raw_p = st.sidebar.number_input('Pitch lim (deg)', value=3.0, min_value=0.0, max_value=90.0, step=0.5)
xo = st.sidebar.number_input('X offset (m)', 0.0)
yo = st.sidebar.number_input('Y offset (m)', 30.0)
zo = st.sidebar.number_input('Z offset (m)', 10.0)

st.sidebar.header('Number of Workers, Parallelization')
workers = st.sidebar.slider('Workers', 1, os.cpu_count(), os.cpu_count())
if st.sidebar.button("Calculate Operability") and scatter_path:
    # apply safety factor here
    limits_center = {
      'heave': raw_h / safety,  
      'roll' : math.radians(raw_r / safety),
      'pitch': math.radians(raw_p / safety)
    }

    # offsets can also be factored if needed:
    limits_offset = {'heave': raw_off_h / safety}
    offsets = (xo, yo, zo)

    st.session_state.limits_center = limits_center
    st.session_state.limits_offset = limits_offset
    st.session_state.offsets       = offsets 

    results, df_ops = compute_all(scatter_path, limits_center, limits_offset, offsets, workers)
    st.session_state.results = results
    st.session_state.df_ops = df_ops


# Main plots
if 'results' in st.session_state:
    res = st.session_state.results
    df_ops = st.session_state.df_ops
    hulls = sorted({cfg['hull'] for cfg, *_ in res})
    lengths = sorted({cfg['L'] for cfg, *_ in res})
    beams   = sorted({cfg['B'] for cfg, *_ in res})
    drafts  = sorted({cfg['T'] for cfg, *_ in res})

    # — Compare two hulls side-by-side —
    st.header("Compare Configurations")

    # layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vessel 1")
        v1_h = st.selectbox("Hull Type", hulls, key="v1_h")
        v1_L = st.selectbox("Select Length (m)", lengths, key="v1_L")
        v1_B = st.selectbox("Select Beam   (m)", beams,   key="v1_B")
        v1_T = st.selectbox("Select Draft  (m)", drafts,  key="v1_T")
    with col2:
        st.subheader("Vessel 2")
        v2_h = st.selectbox("Hull Type", hulls, key="v2_h")
        v2_L = st.selectbox("Select Length (m)", lengths, key="v2_L")
        v2_B = st.selectbox("Select Beam   (m)", beams,   key="v2_B")
        v2_T = st.selectbox("Select Draft  (m)", drafts,  key="v2_T")

    def get_annual(hull, L, B, T):
        # build a boolean Series mask of exactly the same length as df_ops
        mask = (
            (df_ops['hull'] == hull) &
            np.isclose(df_ops['L'], L) &
            np.isclose(df_ops['B'], B) &
            np.isclose(df_ops['T'], T)
        )
        sub = df_ops.loc[mask, 'annual']
        if sub.empty:
            return None
        return float(sub.iloc[0])

    # compute
    v1_days = get_annual(v1_h, v1_L, v1_B, v1_T)
    v2_days = get_annual(v2_h, v2_L, v2_B, v2_T)

    if v1_days is not None and v2_days is not None:
        diff = v2_days - v1_days
        pct  = (diff / v1_days) * 100 if v1_days else 0
        verb = "increase" if diff >= 0 else "decrease"

        st.markdown("**Increase/reduction in operability:**")
        st.write(f"From **{v1_days:.1f} days** to **{v2_days:.1f} days** → "
                f"{abs(pct):.1f}% {verb}")
    else:
        st.warning("One of the selected configurations wasn’t found in the results.")

    sel_h = st.selectbox('Select Hull Type', hulls)
    sel_L = st.selectbox('Select Length (m)', lengths)
    sel_B = st.selectbox('Select Beam   (m)', beams)
    sel_T = st.selectbox('Select Draft  (m)', drafts)

    # the new “add” checkbox
    # add / clear controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Add configuration'):
            new_cfg = (sel_h, sel_L, sel_B, sel_T)
            if new_cfg not in st.session_state.plot_configs:
                st.session_state.plot_configs.append(new_cfg)
    with col2:
        if st.button('Clear configurations'):
            st.session_state.plot_configs = []

    # optional: show user what’s queued
    #st.write('Configs in plot:', st.session_state.plot_configs)

    # filter your results down to only the “added” configs
    filtered = [
        (cfg, joint, rms, ann)
        for cfg, joint, rms, ann in res
        if (cfg['hull'], cfg['L'], cfg['B'], cfg['T']) in st.session_state.plot_configs
    ]
    
    sel_h_cont = st.selectbox("Hull type for contour", hulls)
    sel_contour_lim = st.select_slider("Select highlighted contour line [Days]: ", options=np.arange(0,366,1), value=300)


    plot_type = st.radio('Select Plot', ['Polar Oper.', 'Tornado', 'Contour'])

    if plot_type == 'Polar Oper.':
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        for cfg, joint, _, _ in filtered:
            ang, op = zip(*joint)
            ax.plot(
                np.radians(ang),
                [o * 100 for o in op],
                label=f"L={cfg['L']},B={cfg['B']},T={cfg['T']}"
            )
        ax.set_rlim(0, 100)
        ax.set_title('Operability vs Heading (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # show legend
        st.pyplot(fig)
        
    
    elif plot_type == 'Tornado':
        # --- pick exactly ONE hull (reuse the Single-select widgets) ----
        # ---- find a cfg that really exists ---------------------------------
        cfg_match = next(
            (cfg for cfg, *_ in res
            if (cfg['hull'] == sel_h)
            and np.isclose(cfg['L'], sel_L)
            and np.isclose(cfg['B'], sel_B)
            and np.isclose(cfg['T'], sel_T)),
            None
        )

        if cfg_match is None:
            st.warning("That L–B–T combination wasn’t part of the results. "
                    "Pick one you added with “Add configuration” or run "
                    "Calculate Operability again.")
            st.stop()

        cfg = cfg_match # this is the one we want to plot

        # guard: need scatter + limits (they live in session state only
        #        *after* "Calculate Operability" has been pressed)
        if 'df_ops' not in st.session_state or 'results' not in st.session_state:
            st.info("Run “Calculate Operability” first.")
        else:
            # rebuild scatter & limits exactly like compute_all() did
            limits_center = st.session_state.limits_center
            limits_offset = st.session_state.limits_offset

            df_scatter = pd.read_excel(scatter_path, index_col=0)
            df_scatter.columns = pd.to_numeric(df_scatter.columns,
                                            errors='coerce')
            df_scatter = df_scatter.dropna(axis=1)
            df_scatter /= df_scatter.values.sum()     # normalise to prob.

            base_days, bars = build_tornado(cfg, df_scatter,
                                            limits_center, limits_offset)

            # --------- draw horizontal-bar “tornado” -------------
            labels, lows, highs = zip(*bars)
            y = np.arange(len(labels))

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(y, np.array(highs) - base_days,
                    left=base_days, height=0.4, label='+20 %')
            ax.barh(y, np.array(lows)  - base_days,
                    left=base_days, height=0.4, label='-20 %')
            ax.axvline(base_days, color='k')
            ax.set_yticks(y, labels)
            ax.set_xlabel('Operable days / year')
            ax.set_title(
                f'Sensitivity – L={sel_L}, B={sel_B}, T={sel_T}\n'
                f'Baseline = {base_days:.1f} days/yr')
            ax.legend()
            st.pyplot(fig)



    else:
                # Contour: same contour levels across all drafts
        drafts = sorted(df_ops['T'].unique())
        # let user pick number of levels
        #n_contours = st.sidebar.number_input(
        #    'Number of contour levels', min_value=3, max_value=100, value=10
        #)
        # compute global min/max operability over all drafts
        vmin = df_ops['annual'].min()
        vmax = df_ops['annual'].max()
        diff = int(vmax) - int(vmin)
        levels = np.linspace(vmin, vmax, diff+2)

        fig, axes = plt.subplots(1, len(drafts), figsize=(6 * len(drafts), 4))

        for ax, Tv in zip(axes, drafts):
            # Filter the data for the selected hull and draft
            sub = df_ops[(df_ops["hull"] == sel_h_cont) & (df_ops["T"] == Tv)]
            piv = sub.pivot(index="B", columns="L", values="annual")
            
            X, Y = np.meshgrid(piv.columns, piv.index)

            # Create the filled contour plot
            cs = ax.contourf(X, Y, piv.values, levels=levels)  # Use your predefined levels

            # Add contour lines on top of the filled contours
            contour = ax.contour(X, Y, piv.values, levels=[sel_contour_lim], colors='red', linewidths=1)

            # Set the axis labels and title
            ax.set(
                title=f'T = {Tv} (m)',
                xlabel='Length (m)',
                ylabel='Beam (m)'
            )

            # Add the colorbar
            cbar = plt.colorbar(cs, ax=ax)
            cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins='auto'))
            cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)} days"))  # Format as integer days

            # Manually add a line to the colorbar at value 230
            line_position = 230  # The level where you want to add a line

            # Get the colorbar limits from the mappable (cs)
            cbar_limits = [vmin, vmax]

            # Normalize the position of the line based on the colorbar limits
            normalized_position = (line_position - cbar_limits[0]) / (cbar_limits[1] - cbar_limits[0])

        # Display the plot in Streamlit
        st.pyplot(fig)
        if st.button("Save Contour Plot to File"):
            fig_path = Path(RESULTS_DIR) / f"{sel_h_cont}_contour_plot.png"
            fig.savefig(fig_path, bbox_inches='tight')
            st.success(f"Contour plot saved to {fig_path}")

        

        # Save the figure to a file button


#"""
#    , 'Heave RMS', 'Annual Bar'
#    elif plot_type == 'Heave RMS':
#        fig, ax = plt.subplots()
#        for cfg, _, rms_list, _ in res:
#            hdg, rvals = zip(*rms_list)
#            ax.plot(hdg, rvals, label=f"L={cfg['L']},B={cfg['B']},T={cfg['T']}" )
#        ax.set(title='Heave RMS vs Heading', xlabel='Heading (deg)', ylabel='RMS')
#        st.pyplot(fig)
#
#    elif plot_type == 'Annual Bar':
#        fig, ax = plt.subplots()
#        labels = [f"L={r[0]['L']},B={r[0]['B']},T={r[0]['T']}" for r in res]
#        vals = [r[3] for r in res]
#        ax.bar(labels, vals)
#        ax.set(title='Annual Operable Days', ylabel='Days/yr')
#        plt.xticks(rotation=45)
#        st.pyplot(fig)
#"""
#python -m streamlit run Operability-App-V5.py