import io
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import sympy as sp

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Hispanic Heritage Historical Data Explorer + Function Lab
st.set_page_config(page_title="Hispanic Heritage Data & Function Lab", page_icon="🌎", layout="wide")

COUNTRIES = {
    "Mexico": "MEX", "Brazil": "BRA", "Argentina": "ARG", "Colombia": "COL",
    "Chile": "CHL", "Peru": "PER", "Venezuela": "VEN", "Ecuador": "ECU", "Uruguay": "URY"
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Years a newborn is expected to live if current mortality patterns persist."),
    "Unemployment rate (national estimate)": ("SL.UEM.TOTL.NE.ZS", "% of total labor force", "Share of the labor force without work but available for and seeking employment; national estimate."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by population. This is not average household income or wealth."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people", "Annual live births per 1,000 people; not a count of births."),
    "Net migration": ("SM.POP.NETM", "people", "Net number of migrants (immigration minus emigration) over the source period; not the number leaving."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "homicides per 100,000 people", "Intentional homicides per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in consumer prices."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP, not all public-sector debt."),
    "Total reserves, including gold": ("FI.RES.TOTL.CD", "current US$", "Total reserves including gold, in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a percentage of GDP; not the current account."),
}
CONTEXTS = {
    "Business": ("Revenue", "dollars", "time", "years", "Revenue changes over time."),
    "Engineering": ("Position", "meters", "time", "seconds", "Position of an object or mechanism."),
    "Medical": ("Drug concentration", "mg/L", "time", "hours", "A simplified concentration model; not medical advice or a dosing tool."),
    "Sports science": ("Position", "meters", "time", "seconds", "Position during a movement."),
    "Biology": ("Population", "organisms", "time", "days", "Population or cell-colony change."),
    "Bioengineering": ("Cell count", "cells", "time", "hours", "A simplified cell-growth model."),
    "Economic forecasting": ("Price index", "index points", "time", "years", "A simplified economic indicator."),
    "Weather": ("Temperature", "°C", "time", "hours", "A simplified temperature pattern."),
    "Custom": ("Output", "units", "input", "units", "Describe your own context."),
}
SUGGESTED = {
    "Business": "0.08*x**2 + 2*x + 10",
    "Engineering": "-4.9*x**2 + 20*x + 1",
    "Medical": "8*exp(-0.4*x)",
    "Sports science": "2*sin(x) + 3",
    "Biology": "100*exp(0.12*x)",
    "Bioengineering": "50*exp(0.08*x)",
    "Economic forecasting": "0.5*x**2 + 2*x + 100",
    "Weather": "10*sin(2*pi*x/24) + 15",
    "Custom": "x**2 + 2*x + 1",
}
WORLD_BANK = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=1000"

st.title("🌎 Hispanic Heritage Historical Data Explorer")
st.caption("Explore authentic World Bank indicators, fit polynomial models, compare countries, and investigate functions.")

with st.sidebar:
    st.header("Historical data")
    country = st.selectbox("Main country", list(COUNTRIES.keys()), index=0)
    indicator_name = st.selectbox("Indicator", list(INDICATORS.keys()), index=1)
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Model marker interval (years)", 1, 10, 1)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 5)
    compare_countries = st.multiselect("Compare same indicator with", [c for c in COUNTRIES if c != country], default=["Colombia"])
    printer_view = st.toggle("Printer-friendly view", value=False)
    st.divider()
    st.header("Function lab")
    course = st.selectbox("Math pathway", ["Honors Algebra 2", "Precalculus", "Calculus"], index=2)
    context = st.selectbox("Context", list(CONTEXTS.keys()), index=2)
    use_fit = st.checkbox("Use historical-data curve of best fit", value=False)
    use_suggested = st.checkbox("Use suggested function", value=True)
    custom_desc = st.text_input("Context description", value=CONTEXTS[context][4])
    var_x = st.text_input("Independent variable name", value=CONTEXTS[context][2])
    unit_x = st.text_input("Independent variable units", value=CONTEXTS[context][3])
    var_y = st.text_input("Dependent variable name", value=CONTEXTS[context][0])
    unit_y = st.text_input("Dependent variable units", value=CONTEXTS[context][1])
    st.caption("Function examples use x as the independent variable.")
    function_text = st.text_input("Function f(x)", value=SUGGESTED[context])
    interval = st.slider("Analysis interval", 0.0, 20.0, (0.0, 10.0), 0.5)
    current_x = st.slider("Current input x", 0.0, 20.0, 3.0, 0.25)
    aroc_left = st.slider("Average-rate interval start", 0.0, 20.0, 2.0, 0.25)
    aroc_right = st.slider("Average-rate interval end", 0.0, 20.0, 6.0, 0.25)
    show_f = st.checkbox("Show f(x)", True)
    show_fp = st.checkbox("Show f′(x)", True)
    show_fpp = st.checkbox("Show f″(x)", False)
    animate = st.checkbox("Animation mode (step through x)", False)
    speed = st.select_slider("Animation speed", options=["Slow", "Medium", "Fast"], value="Medium")

# Retrieve data
country_code = COUNTRIES[country]
indicator_code, units, definition = INDICATORS[indicator_name]
end_year = datetime.now().year - 1
start_year = end_year - 69  # 70 calendar years inclusive
source_url = WORLD_BANK.format(country=country_code, indicator=indicator_code, start=start_year, end=end_year)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(cc, code, start, end):
    url = WORLD_BANK.format(country=cc, indicator=code, start=start, end=end)
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        return pd.DataFrame(columns=["Year", "Value"]), url
    rows = [{"Year": int(row["date"]), "Value": float(row["value"])}
            for row in payload[1] if row.get("date") and row.get("value") is not None]
    df = pd.DataFrame(rows, columns=["Year", "Value"]).drop_duplicates("Year").sort_values("Year")
    return df, url

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_compare(cc, code, start, end):
    return fetch_world_bank(cc, code, start, end)[0]

st.subheader("1. Historical data")
st.markdown("Brazil is included as a **Latin American comparison**; it is not a Spanish-speaking Hispanic country.")
try:
    with st.spinner(f"Retrieving official World Bank data for {country}…"):
        source_df, source_url = fetch_world_bank(country_code, indicator_code, start_year, end_year)
    if source_df.empty:
        st.warning("The World Bank API returned no observations for this country/indicator/window. No replacement values have been fabricated.")
    else:
        st.success(f"Retrieved {len(source_df)} observations.")
except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
    source_df = pd.DataFrame(columns=["Year", "Value"])
    st.error(f"Could not retrieve live World Bank data: {exc}. Check the connection or try again later; no mock data are substituted.")

meta_cols = st.columns(4)
meta_cols[0].metric("Indicator code", indicator_code)
meta_cols[1].metric("Available observations", len(source_df))
meta_cols[2].metric("First observed year", str(int(source_df.Year.min())) if not source_df.empty else "—")
meta_cols[3].metric("Last observed year", str(int(source_df.Year.max())) if not source_df.empty else "—")
st.write(f"**Definition:** {definition}  \n**Units:** {units}  \n**Source:** [World Bank indicator page](https://data.worldbank.org/indicator/{indicator_code}) · [API request]({source_url})  \n**Retrieved:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Editable data copy stored by data signature
signature = f"{country}|{indicator_code}|{start_year}|{end_year}"
if st.session_state.get("data_signature") != signature:
    st.session_state["data_signature"] = signature
    st.session_state["working_data"] = source_df.copy()
    st.session_state.pop("single_result", None)
    st.session_state.pop("batch_result", None)

st.markdown("#### Editable working table")
st.caption("Edit values to explore a scenario. The source observations remain unchanged; use Reset to source to restore them.")
edit_col1, edit_col2 = st.columns([1, 5])
with edit_col1:
    if st.button("Reset to source"):
        st.session_state["working_data"] = source_df.copy()
        st.session_state["data_editor"] = source_df.copy()
        st.session_state.pop("single_result", None)
        st.session_state.pop("batch_result", None)
with edit_col2:
    st.caption("Years must be unique integers; values must be numeric and finite.")
edited = st.data_editor(st.session_state.get("working_data", source_df), num_rows="dynamic", use_container_width=True, key="data_editor", disabled=source_df.empty)
working = edited.copy() if isinstance(edited, pd.DataFrame) else source_df.copy()
if not working.empty:
    working["Year"] = pd.to_numeric(working["Year"], errors="coerce")
    working["Value"] = pd.to_numeric(working["Value"], errors="coerce")
    before = len(working)
    working = working.dropna(subset=["Year", "Value"])
    working = working[np.isfinite(working["Year"]) & np.isfinite(working["Value"])]
    working = working[working["Year"] == np.floor(working["Year"])]
    working["Year"] = working["Year"].astype(int)
    working = working.drop_duplicates("Year").sort_values("Year")
    if len(working) != before:
        st.info(f"Validation removed {before-len(working)} row(s) with missing/invalid fields, non-integer years, or duplicate years.")
    if indicator_name in ["Life expectancy at birth", "Unemployment rate (national estimate)", "Crude birth rate", "Intentional homicide rate"] and (working["Value"] < 0).any():
        st.warning("Negative values are not valid for this indicator. Remove or correct them before fitting.")
        working = working[working["Value"] >= 0]
is_edited = not working.reset_index(drop=True).equals(source_df.reset_index(drop=True))

# Fit polynomial on centered time
fit_ok = len(working) >= degree + 1 and working["Year"].nunique() >= degree + 1
model = None
fit_info = {}
if fit_ok:
    base_year = int(working["Year"].mean())
    t = working["Year"].to_numpy(dtype=float) - base_year
    y = working["Value"].to_numpy(dtype=float)
    try:
        coeffs = np.polynomial.polynomial.polyfit(t, y, degree)
        model = np.polynomial.Polynomial(coeffs)
        pred = model(t)
        residuals = y - pred
        ss_tot = float(np.sum((y-y.mean())**2))
        r2 = float(1 - np.sum(residuals**2)/ss_tot) if ss_tot else float("nan")
        mae = float(np.mean(np.abs(residuals)))
        fit_info = {"base": base_year, "coeffs": coeffs, "r2": r2, "mae": mae, "residuals": residuals,
                    "mean": float(np.mean(y)), "min_year": int(working.Year.min()), "max_year": int(working.Year.max())}
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        st.error(f"Model fitting failed: {exc}")
else:
    st.warning(f"Degree {degree} needs at least {degree+1} distinct valid years; currently there are {working['Year'].nunique() if not working.empty else 0}.")

def predict_year(year):
    if model is None:
        return np.nan
    return float(model(float(year) - fit_info["base"]))

def classify_year(year):
    if year < fit_info["min_year"] or year > fit_info["max_year"]:
        return "Extrapolation (outside fitted range)"
    if int(year) in set(working["Year"].tolist()):
        return "Fitted value at an observed year"
    return "Within-range estimate"

st.subheader("2. Polynomial model and visualization")
if model is not None:
    st.caption("Metrics below describe the fit to the same observations (in-sample); they do not prove forecasting accuracy.")
    mcols = st.columns(5)
    mcols[0].metric("Degree", degree)
    mcols[1].metric("R² (in-sample)", f"{fit_info['r2']:.4f}")
    mcols[2].metric("MAE", f"{fit_info['mae']:.4g} {units}")
    mcols[3].metric("Mean", f"{fit_info['mean']:.4g} {units}")
    mcols[4].metric("Fitted range", f"{fit_info['min_year']}–{fit_info['max_year']}")
    eq = " + ".join([f"({c:.6g})·t^{i}" for i, c in enumerate(fit_info["coeffs"])])
    st.code(f"t = year − {fit_info['base']}\nŷ = {eq}", language="text")
    st.info("Residual error/MAE is a summary of historical fit error. This app does not provide validated 95% prediction intervals.")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.scatter(working["Year"], working["Value"], label="Observed / working data", s=22)
    xlo, xhi = fit_info["min_year"], fit_info["max_year"]
    xcurve = np.linspace(xlo, xhi + horizon, max(250, (xhi + horizon-xlo+1)*2))
    ycurve = np.array([predict_year(x) for x in xcurve])
    inside = xcurve <= xhi
    ax.plot(xcurve[inside], ycurve[inside], label="Fitted curve (historical range)")
    if horizon > 0:
        ax.plot(xcurve[~inside], ycurve[~inside], linestyle="--", label="Extrapolated display segment")
    markers = np.arange(xlo, xhi + horizon + 1, sample_step)
    ax.scatter(markers, [predict_year(x) for x in markers], marker="x", label=f"Model markers every {sample_step} year(s)")
    ax.axvline(xhi, linestyle=":", linewidth=1.5, label="End of historical fit")
    for comp in compare_countries:
        try:
            comp_df = fetch_compare(COUNTRIES[comp], indicator_code, start_year, end_year)
            if not comp_df.empty:
                ax.plot(comp_df["Year"], comp_df["Value"], marker=".", linewidth=1, label=f"{comp} observed")
                st.caption(f"{comp}: {int(comp_df.Year.min())}–{int(comp_df.Year.max())}, {len(comp_df)} observations; same indicator and units.")
            else:
                st.caption(f"{comp}: no observations available for this indicator/window.")
        except Exception as exc:
            st.caption(f"{comp}: retrieval unavailable ({exc}).")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{indicator_name} ({units})")
    ax.set_title(f"{country}: {indicator_name}")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
else:
    fig = None

# Single / batch estimates
st.subheader("3. Estimate values")
if model is not None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Single year")
        single_year = st.number_input("Year to estimate (1950–2100)", min_value=1950, max_value=2100, value=min(2100, max(1950, fit_info["max_year"]+1)), step=1)
        if st.button("Calculate value"):
            st.session_state["single_result"] = {"Year": int(single_year), "Estimate": predict_year(single_year), "Classification": classify_year(single_year)}
        result = st.session_state.get("single_result")
        if result:
            st.metric(f"Estimated value in {result['Year']}", f"{result['Estimate']:.6g} {units}")
            st.write(f"**Type:** {result['Classification']}")
            actual = working.loc[working.Year == result["Year"], "Value"]
            if not actual.empty:
                st.write(f"**Observed value:** {actual.iloc[0]:.6g} {units}")
            if abs(result["Estimate"]) > max(1, abs(fit_info["mean"])) * 100:
                st.warning("This polynomial estimate is unusually large relative to the historical mean. Review extrapolation limits; it is not clipped.")
    with c2:
        st.markdown("#### Batch years")
        batch_mode = st.radio("Input type", ["Year range", "Custom years"], horizontal=True)
        valid_years = []
        if batch_mode == "Year range":
            b1, b2, b3 = st.columns(3)
            start_b = b1.number_input("Start", 1950, 2100, max(1950, fit_info["min_year"]), key="batch_start")
            end_b = b2.number_input("End", 1950, 2100, min(2100, fit_info["max_year"]+5), key="batch_end")
            step_b = b3.number_input("Step", 1, 10, 1, key="batch_step")
            if end_b <= start_b:
                st.warning("End year must be greater than start year.")
            else:
                valid_years = list(range(int(start_b), int(end_b)+1, int(step_b)))
                if valid_years and valid_years[-1] != int(end_b):
                    st.caption("The step does not land exactly on the end year, so the end year is not included.")
        else:
            raw_years = st.text_input("Comma-separated years", value=f"{fit_info['min_year']}, {fit_info['max_year']}, {fit_info['max_year']+5}")
            parsed, invalid = [], []
            for token in raw_years.split(","):
                token = token.strip()
                try:
                    val = int(token)
                    if 1950 <= val <= 2100:
                        parsed.append(val)
                    else:
                        invalid.append(token)
                except ValueError:
                    if token:
                        invalid.append(token)
            valid_years = list(dict.fromkeys(parsed))
            if invalid:
                st.warning(f"Ignored invalid/out-of-range entries: {', '.join(invalid)}")
            if len(parsed) != len(valid_years):
                st.caption("Duplicate years were removed.")
        st.write("**Valid years:**", ", ".join(map(str, valid_years)) if valid_years else "None")
        if st.button("Calculate batch", disabled=not valid_years):
            st.session_state["batch_result"] = pd.DataFrame([{"Year": yr, "Estimate": predict_year(yr), "Units": units, "Classification": classify_year(yr)} for yr in valid_years])
        if "batch_result" in st.session_state:
            st.dataframe(st.session_state["batch_result"], use_container_width=True)

# Function lab: parse and analyze symbolic function
st.subheader("4. Function lab: rates, derivatives, and behavior")
if use_fit and model is not None:
    # Convert centered polynomial coefficients into symbolic expression in x
    x = sp.symbols("x", real=True)
    fx = sum(sp.Float(float(c), 12) * (x - fit_info["base"])**i for i, c in enumerate(fit_info["coeffs"]))
    st.caption("Function lab is using the centered polynomial fitted to the selected historical data.")
else:
    x = sp.symbols("x", real=True)
    try:
        parsed = sp.sympify(function_text, locals={"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                                                   "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "pi": sp.pi})
        if parsed.free_symbols - {x}:
            raise ValueError("Use x as the only variable.")
        fx = parsed
    except Exception as exc:
        fx = None
        st.error(f"Could not read the function. Use notation such as x**2, sin(x), exp(x), or log(x). Details: {exc}")

if fx is not None:
    fprime = sp.diff(fx, x)
    fsecond = sp.diff(fprime, x)
    st.write(f"**Context:** {custom_desc}")
    st.latex("f(x) = " + sp.latex(fx))
    if course == "Calculus":
        st.latex("f'(x) = " + sp.latex(fprime))
        st.latex("f''(x) = " + sp.latex(fsecond))
    elif course == "Precalculus":
        st.write("Precalculus focus: compare function shape, turning points, end behavior, and changing average rates. Derivatives are shown only in Calculus mode.")
    else:
        st.write("Honors Algebra 2 focus: domain/range, end behavior, trends, and average rates of change.")
    def safe_callable(expr):
        return sp.lambdify(x, expr, modules=["numpy"])
    ff = safe_callable(fx)
    fpf = safe_callable(fprime)
    fppf = safe_callable(fsecond)
    def eval_scalar(fn, val):
        try:
            ans = fn(float(val))
            return float(np.asarray(ans))
        except Exception:
            return float("nan")
    fval = eval_scalar(ff, current_x)
    st.metric(f"{var_y} at {var_x}={current_x:g} {unit_x}", f"{fval:.6g} {unit_y}" if np.isfinite(fval) else "Undefined here")
    st.write(f"**Interpretation:** {custom_desc} The function value represents {var_y}; its rate of change is measured in {unit_y} per {unit_x}.")
    if aroc_right == aroc_left:
        st.warning("Choose different endpoints for average rate of change.")
    else:
        fa, fb = eval_scalar(ff, aroc_left), eval_scalar(ff, aroc_right)
        aroc = (fb-fa)/(aroc_right-aroc_left) if np.isfinite(fa) and np.isfinite(fb) else np.nan
        st.write(f"**Average rate of change** from x={aroc_left:g} to x={aroc_right:g}: {aroc:.6g} {unit_y}/{unit_x}" if np.isfinite(aroc) else "**Average rate of change:** undefined because an endpoint is outside the function's real domain.")
    inst = eval_scalar(fpf, current_x)
    if course == "Calculus":
        st.write(f"**Instantaneous rate of change:** f′({current_x:g}) = {inst:.6g} {unit_y}/{unit_x}" if np.isfinite(inst) else "**Instantaneous rate of change:** undefined at this point.")
        st.write(f"**Second derivative:** f″({current_x:g}) = {eval_scalar(fppf, current_x):.6g} {unit_y}/{unit_x}²")
    lo, hi = interval
    grid = np.linspace(lo, hi, 700)
    def eval_array(fn, xs):
        with np.errstate(all="ignore"):
            vals = np.asarray(fn(xs), dtype=float)
            if vals.ndim == 0:
                vals = np.full_like(xs, float(vals), dtype=float)
            return np.where(np.isfinite(vals) & (np.abs(vals) < 1e12), vals, np.nan)
    y0, y1, y2 = eval_array(ff, grid), eval_array(fpf, grid), eval_array(fppf, grid)
    fig2, ax2 = plt.subplots(figsize=(10, 4.8))
    if show_f: ax2.plot(grid, y0, label="f(x)")
    if show_fp: ax2.plot(grid, y1, label="f′(x)")
    if show_fpp: ax2.plot(grid, y2, label="f″(x)")
    if np.isfinite(fval) and show_f: ax2.scatter([current_x], [fval], marker="o", s=55, label="Current point")
    if np.isfinite(inst) and show_f:
        tangent_y = fval + inst*(grid-current_x)
        tangent_y[(grid < current_x-2) | (grid > current_x+2)] = np.nan
        ax2.plot(grid, tangent_y, linestyle="--", label="Local tangent line")
    ax2.axvline(current_x, linestyle=":", alpha=.6)
    ax2.set_xlabel(f"{var_x} ({unit_x})")
    ax2.set_ylabel("Function / derivative values (scales may differ)")
    ax2.set_title("Function and derivative graph")
    ax2.grid(True, alpha=.25)
    ax2.legend()
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
    if course == "Calculus":
        # numerical critical points and inflection candidates, confirmed by sign changes
        def sign_change_roots(vals, xs):
            roots = []
            for i in range(len(vals)-1):
                if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i] == 0:
                    roots.append(xs[i])
                elif np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i]*vals[i+1] < 0:
                    roots.append((xs[i]+xs[i+1])/2)
            return roots
        crit = sign_change_roots(y1, grid)
        infl = sign_change_roots(y2, grid)
        st.markdown("#### Numerical behavior on the selected interval")
        if crit:
            rows = []
            for cp in crit:
                left_s, right_s = eval_scalar(fpf, cp-0.01), eval_scalar(fpf, cp+0.01)
                kind = "local maximum" if left_s > 0 and right_s < 0 else ("local minimum" if left_s < 0 and right_s > 0 else "stationary point; no sign-change extremum")
                rows.append({"x (approx.)": round(cp, 5), "f(x)": eval_scalar(ff, cp), "classification": kind})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.write("No first-derivative sign-change roots detected on this sampled interval.")
        if infl:
            rows = []
            for ip in infl:
                left_s, right_s = eval_scalar(fppf, ip-0.01), eval_scalar(fppf, ip+0.01)
                changes = left_s*right_s < 0
                rows.append({"x (approx.)": round(ip, 5), "f(x)": eval_scalar(ff, ip), "concavity changes": "yes" if changes else "not confirmed"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.write("No second-derivative sign-change candidates detected on this sampled interval.")
        st.caption("Critical/inflection-point results are numerical approximations from a grid, not symbolic proofs. Discontinuities and very narrow features may be missed.")
        # closed interval extrema on sampled points
        finite = np.isfinite(y0)
        if finite.any():
            ix = np.where(finite)[0]
            imin, imax = ix[np.argmin(y0[finite])], ix[np.argmax(y0[finite])]
            st.write(f"**Approximate absolute extrema on [{lo:g}, {hi:g}]:** minimum ≈ ({grid[imin]:.4g}, {y0[imin]:.4g}); maximum ≈ ({grid[imax]:.4g}, {y0[imax]:.4g}). Endpoints are included in the numerical scan.")
    if animate:
        if "anim_x" not in st.session_state: st.session_state.anim_x = current_x
        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("▶ Step / play"):
                st.session_state.anim_x = min(20.0, st.session_state.anim_x + {"Slow": .25, "Medium": .5, "Fast": 1.0}[speed])
        with a2:
            if st.button("⏸ Pause"):
                st.session_state.anim_x = st.session_state.anim_x
        with a3:
            if st.button("↺ Reset"):
                st.session_state.anim_x = 0.0
        st.write(f"Simulation x = {st.session_state.anim_x:.2f}. Press **Step / play** repeatedly to move the point; speed controls step size.")
        av = eval_scalar(ff, st.session_state.anim_x)
        if np.isfinite(av):
            st.write(f"At x={st.session_state.anim_x:.2f}, the simulated {var_y.lower()} is {av:.5g} {unit_y}.")
        if context == "Medical":
            st.caption("Visual indicator: concentration level. This simplified model is for mathematics learning only, not treatment or dosing guidance.")
        elif context in ["Engineering", "Sports science"]:
            st.caption("Visual indicator: moving object/position. The plotted y-value represents the selected function, not a real-world calibrated simulation.")
        elif context in ["Biology", "Bioengineering"]:
            st.caption("Visual indicator: population/cell-colony level. Values are mathematical outputs, not measured biological forecasts.")
        elif context == "Weather":
            st.caption("Visual indicator: thermometer level. This is a sample mathematical model, not a weather forecast.")
        else:
            st.caption("Visual indicator: contextual output level; scale follows the selected function.")
else:
    st.info("Enter a valid function to enable the function lab.")

# Model comparisons for degrees
st.subheader("5. Compare polynomial degrees")
if not working.empty and working.Year.nunique() >= 6:
    comp_rows = []
    years = working.Year.to_numpy(dtype=float)
    vals = working.Value.to_numpy(dtype=float)
    base = float(years.mean())
    for d in sorted(set([3, min(5, len(working)-1), degree])):
        if d < 1 or len(working) < d+1: continue
        try:
            co = np.polynomial.polynomial.polyfit(years-base, vals, d)
            pv = np.polynomial.polynomial.polyval(years-base, co)
            resid = vals-pv
            ss = np.sum((vals-vals.mean())**2)
            comp_rows.append({"Degree": d, "R² (in-sample)": 1-np.sum(resid**2)/ss if ss else np.nan, "MAE (in-sample)": np.mean(np.abs(resid))})
        except Exception:
            pass
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows).drop_duplicates("Degree"), use_container_width=True)
        st.caption("A higher-degree model can fit historical points more closely while extrapolating less plausibly. Compare residual patterns and, where possible, test later years separately.")

# Exports
st.subheader("6. Download reports")
data_out = working.copy()
data_out["Source status"] = "User-modified scenario" if is_edited else "Source data"
csv_bytes = data_out.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV data", data=csv_bytes, file_name=f"{country}_{indicator_code}_data.csv", mime="text/csv")

summary = pd.DataFrame([{
    "Country": country, "Indicator": indicator_name, "Indicator code": indicator_code, "Units": units,
    "Source URL": source_url, "Retrieved": datetime.now().isoformat(timespec="minutes"),
    "Coverage": f"{int(working.Year.min())}–{int(working.Year.max())}" if not working.empty else "No observations",
    "Observations": len(working), "Degree": degree if model is not None else "Not fitted",
    "Base year": fit_info.get("base", ""), "Equation": f"ŷ = {eq}" if model is not None else "Not fitted",
    "R² in-sample": fit_info.get("r2", ""), "MAE": fit_info.get("mae", ""),
    "Data status": "User-modified scenario" if is_edited else "Source data"
}])
excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
    summary.to_excel(writer, index=False, sheet_name="Summary")
    data_out.to_excel(writer, index=False, sheet_name="Working Data")
    if "batch_result" in st.session_state:
        st.session_state["batch_result"].to_excel(writer, index=False, sheet_name="Calculated Years")
st.download_button("Download Excel workbook", data=excel_buf.getvalue(), file_name=f"{country}_{indicator_code}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def make_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=0.45*inch, leftMargin=0.45*inch, topMargin=0.45*inch, bottomMargin=0.45*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]),
             Paragraph(f"{country} — {indicator_name}", styles["Heading2"]),
             Paragraph(f"Indicator code: {indicator_code} | Units: {units} | Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["BodyText"]),
             Paragraph(f"Definition: {definition}", styles["BodyText"]), Spacer(1, 8)]
    if model is not None:
        story += [Paragraph(f"Polynomial degree {degree}; base year {fit_info['base']}; fitted range {fit_info['min_year']}–{fit_info['max_year']}.", styles["BodyText"]),
                  Paragraph(f"R-squared (in-sample): {fit_info['r2']:.5g}; MAE: {fit_info['mae']:.5g} {units}.", styles["BodyText"]),
                  Paragraph("These are in-sample fit metrics, not proof of forecast accuracy. No validated 95% prediction interval is provided.", styles["BodyText"]), Spacer(1, 8)]
        if fig is not None:
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format="png", dpi=120, bbox_inches="tight")
            img_buf.seek(0)
            story.append(RLImage(img_buf, width=8.5*inch, height=3.3*inch))
    story.append(Paragraph("Data preview (first 25 rows)", styles["Heading3"]))
    table_data = [list(data_out.columns[:2])] + data_out.iloc[:25, :2].astype(str).values.tolist() if not data_out.empty else [["Year", "Value"], ["No observations", ""]]
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8eef7")),("GRID",(0,0),(-1,-1),0.4,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("PADDING",(0,0),(-1,-1),3)]))
    story.append(table)
    if "single_result" in st.session_state:
        r = st.session_state["single_result"]
        story += [Spacer(1, 8), Paragraph(f"Single estimate: year {r['Year']}, {r['Estimate']:.6g} {units} ({r['Classification']}).", styles["BodyText"])]
    if "batch_result" in st.session_state:
        story += [Spacer(1, 8), Paragraph("Batch results", styles["Heading3"])]
        br = st.session_state["batch_result"]
        story.append(Table([list(br.columns)] + br.astype(str).values.tolist(), repeatRows=1))
    story += [Spacer(1, 8), Paragraph("Limitations: gaps in official data, measurement comparability, model assumptions, and polynomial extrapolation can affect interpretation. Correlation/trend does not establish cause.", styles["BodyText"])]
    doc.build(story)
    return buffer.getvalue()

st.download_button("Download PDF report", data=make_pdf(), file_name=f"{country}_{indicator_code}_report.pdf", mime="application/pdf")
st.caption("Source organization: World Bank. Compare definitions and coverage before interpreting country differences. Use historical and cultural research sources to explain context; a single indicator cannot measure a country's worth or cultural identity.")
st.markdown("---")
st.markdown("**Run locally:** `pip install -r requirements.txt` then `python -m streamlit run app.py`.")
st.markdown("**Streamlit Community Cloud:** push `app.py` and `requirements.txt` to GitHub, choose the repository/branch and `app.py`, then deploy.")
st.markdown("**Render:** build command `pip install -r requirements.txt`; start command `python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT`.")
