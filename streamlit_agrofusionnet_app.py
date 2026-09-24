import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import streamlit as st

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_data
def load_data():
    return pd.read_csv("data/agri_multimodal_dataset_engineered.csv")


@st.cache_resource
def load_artifacts():
    metadata = joblib.load("saved_models/agrofusionnet_metadata.pkl")
    scaler = joblib.load("saved_models/agrofusionnet_scaler.pkl")
    district_map = pd.read_csv("saved_models/district_mapping.csv")
    return metadata, scaler, district_map


class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim**-0.5

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        return attn @ v + x


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x):
        g = self.gate(x)
        return x * g


class AgroFusionNet(nn.Module):
    def __init__(self, num_districts, climate_cols, soil_cols, interaction_cols, stress_cols, temporal_cols):
        super().__init__()

        self.rain_lstm = nn.LSTM(1, 32, batch_first=True)
        self.ndvi_lstm = nn.LSTM(1, 16, batch_first=True)

        self.climate = nn.Sequential(
            nn.Linear(len(climate_cols), 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        self.soil = nn.Sequential(
            nn.Linear(len(soil_cols), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.inter = nn.Sequential(nn.Linear(len(interaction_cols), 16), nn.ReLU())
        self.stress = nn.Sequential(nn.Linear(len(stress_cols), 16), nn.ReLU())
        self.temp = nn.Sequential(nn.Linear(len(temporal_cols), 16), nn.ReLU())

        self.district_embed = nn.Embedding(num_districts, 16)

        fusion_size = 160

        self.cross_attn = CrossModalAttention(fusion_size)
        self.gate = GatedFusion(fusion_size)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.yield_head = nn.Linear(32, 1)
        self.stress_head = nn.Linear(32, 3)

    def forward(self, rain, ndvi, climate, soil, inter, stress, temp, district):
        rain, _ = self.rain_lstm(rain)
        rain = rain[:, -1, :]

        ndvi, _ = self.ndvi_lstm(ndvi)
        ndvi = ndvi[:, -1, :]

        climate = self.climate(climate)
        soil = self.soil(soil)
        inter = self.inter(inter)
        stress = self.stress(stress)
        temp = self.temp(temp)

        district = self.district_embed(district)

        x = torch.cat([rain, ndvi, climate, soil, inter, stress, temp, district], dim=1)
        x = self.cross_attn(x.unsqueeze(1)).squeeze(1)
        x = self.gate(x)
        x = self.fusion(x)

        yield_pred = self.yield_head(x)
        stress_pred = self.stress_head(x)
        return yield_pred, stress_pred


@st.cache_resource
def load_model(metadata, _district_map):
    model = AgroFusionNet(
        num_districts=_district_map["district_id"].nunique(),
        climate_cols=metadata["CLIMATE"],
        soil_cols=metadata["SOIL"],
        interaction_cols=metadata["INTERACTION"],
        stress_cols=metadata["STRESS"],
        temporal_cols=metadata["TEMPORAL"],
    )

    checkpoint = torch.load("saved_models/agrofusionnet_model.pt", map_location=DEVICE)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model


def soil_health_score(n_val, p_val, k_val):
    return round(min((n_val + p_val + k_val) / 300, 1), 2)


def climate_impact(row):
    dev = abs(row["rainfall_deviation"])
    return round(max(min(1 - dev, 1), 0), 2)


def stress_level(prob):
    if prob > 0.75:
        return "High"
    if prob > 0.45:
        return "Moderate"
    return "Low"


def sustainability_index(yield_score, stress_prob, soil_score, climate_score):
    si = 0.4 * yield_score + 0.3 * (1 - stress_prob) + 0.2 * soil_score + 0.1 * climate_score
    return round(si, 2)


def risk_level(si):
    if si >= 0.75:
        return "Low Risk"
    if si >= 0.60:
        return "Moderate Risk"
    if si >= 0.45:
        return "High Risk"
    return "Severe Risk"


def analyze_causes(row, n_val, p_val, k_val):
    causes = []
    rainfall_dev = row["rainfall_deviation"]
    ndvi = row["ndvi"]

    if rainfall_dev < -0.15:
        causes.append("Rainfall deficit detected")
    elif rainfall_dev > 0.35:
        causes.append("Excess rainfall detected")

    if n_val < 60:
        causes.append("Low nitrogen detected")
    elif n_val > 120:
        causes.append("Excess nitrogen detected")

    if p_val < 35:
        causes.append("Low phosphorus detected")

    if k_val < 40:
        causes.append("Low potassium detected")

    if ndvi < 0.45:
        causes.append("Vegetation stress detected")

    return causes


def recommendations(causes):
    recs = []
    for c in causes:
        if "Rainfall deficit" in c:
            recs.append("Increase irrigation frequency")
        if "Excess rainfall" in c:
            recs.append("Improve drainage management")
        if "Low nitrogen" in c:
            recs.append("Apply nitrogen fertilizer")
        if "Excess nitrogen" in c:
            recs.append("Reduce nitrogen application")
        if "Low phosphorus" in c:
            recs.append("Apply phosphorus fertilizer")
        if "Low potassium" in c:
            recs.append("Apply potassium fertilizer")
        if "Vegetation stress" in c:
            recs.append("Inspect crop for pest or disease stress")

    if not recs:
        recs.append("Current conditions appear optimal")

    return recs


def model_predict(sample, n_val, p_val, k_val, ph_val, district_id, model, scaler, metadata, stress_floor=0.1):
    rain_cols = metadata["RAIN_COLS"]
    ndvi_cols = metadata["NDVI_COLS"]
    climate_cols = metadata["CLIMATE"]
    soil_cols = metadata["SOIL"]
    interaction_cols = metadata["INTERACTION"]
    stress_cols = metadata["STRESS"]
    temporal_cols = metadata["TEMPORAL"]
    all_features = metadata["ALL_FEATURES"]

    sim = sample.copy()
    sim["n"] = n_val
    sim["p"] = p_val
    sim["k"] = k_val
    sim["ph"] = ph_val

    sim_scaled = sim.copy()
    sim_scaled[all_features] = sim_scaled[all_features].astype(float)
    sim_scaled.loc[:, all_features] = scaler.transform(sim_scaled[all_features])

    rain = torch.tensor(sim_scaled[rain_cols].values.reshape(1, 12, 1), dtype=torch.float32).to(DEVICE)
    ndvi = torch.tensor(sim_scaled[ndvi_cols].values.reshape(1, 3, 1), dtype=torch.float32).to(DEVICE)
    climate = torch.tensor(sim_scaled[climate_cols].values, dtype=torch.float32).to(DEVICE)
    soil = torch.tensor(sim_scaled[soil_cols].values, dtype=torch.float32).to(DEVICE)
    inter = torch.tensor(sim_scaled[interaction_cols].values, dtype=torch.float32).to(DEVICE)
    stress = torch.tensor(sim_scaled[stress_cols].values, dtype=torch.float32).to(DEVICE)
    temp = torch.tensor(sim_scaled[temporal_cols].values, dtype=torch.float32).to(DEVICE)
    district = torch.tensor([district_id], dtype=torch.long).to(DEVICE)

    runs = 10
    yield_preds = []
    stress_probs = []

    with torch.no_grad():
        for _ in range(runs):
            y_pred, s_pred = model(rain, ndvi, climate, soil, inter, stress, temp, district)
            yield_preds.append(float(y_pred.cpu().numpy()[0][0]))
            stress_probs.append(float(torch.softmax(s_pred, 1).cpu().numpy()[0][2]))

    raw_yield_pred = float(np.mean(yield_preds))
    if not np.isfinite(raw_yield_pred):
        raw_yield_pred = 0.0
    yield_pred = max(raw_yield_pred, 0.0)
    raw_stress_prob = float(np.mean(stress_probs))
    if not np.isfinite(raw_stress_prob):
        raw_stress_prob = stress_floor
    stress_prob = max(raw_stress_prob, stress_floor)
    return yield_pred, stress_prob


def fill_missing_features(sample, crop_df, global_df, all_features):
    filled = sample.copy()

    crop_medians = crop_df[all_features].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    global_medians = global_df[all_features].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)

    for col in all_features:
        if col not in filled.columns:
            continue

        val = pd.to_numeric(filled[col], errors="coerce").iloc[0]
        if pd.isna(val):
            replacement = crop_medians.get(col, np.nan)
            if pd.isna(replacement):
                replacement = global_medians.get(col, np.nan)
            if pd.isna(replacement):
                replacement = 0.0
            filled.loc[filled.index[0], col] = float(replacement)

    return filled


def get_ideal_sample(crop_df):
    scored = crop_df.copy()

    req_cols = ["yield", "ndvi", "rainfall_deviation"]
    for col in req_cols:
        if col not in scored.columns:
            return crop_df.iloc[[-1]].copy()

    scored = scored.dropna(subset=req_cols)
    if scored.empty:
        return crop_df.iloc[[-1]].copy()

    y = scored["yield"].astype(float)
    ndvi = scored["ndvi"].astype(float)
    rain_pen = scored["rainfall_deviation"].astype(float).abs()

    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
    ndvi_norm = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-9)
    rain_norm = 1.0 - (rain_pen - rain_pen.min()) / (rain_pen.max() - rain_pen.min() + 1e-9)

    score = 0.6 * y_norm + 0.3 * ndvi_norm + 0.1 * rain_norm
    best_idx = score.idxmax()

    return crop_df.loc[[best_idx]].copy()


def sensitivity_analysis(sample, n_val, p_val, k_val, ph_val, district_id, model, scaler, metadata):
    rain_cols = metadata["RAIN_COLS"]
    ndvi_cols = metadata["NDVI_COLS"]

    base_yield, _ = model_predict(sample, n_val, p_val, k_val, ph_val, district_id, model, scaler, metadata)
    base_real = base_yield * 350

    rows = []

    def dominant_direction_result(name, up_tuple, down_tuple):
        up_label, up_yield_real = up_tuple
        down_label, down_yield_real = down_tuple
        up_change = up_yield_real - base_real
        down_change = down_yield_real - base_real

        if abs(up_change) >= abs(down_change):
            return {
                "Scenario": f"{name} {up_label}",
                "Predicted Yield (t/ha)": round(up_yield_real, 6),
                "Change (t/ha)": round(up_change, 6),
            }

        return {
            "Scenario": f"{name} {down_label}",
            "Predicted Yield (t/ha)": round(down_yield_real, 6),
            "Change (t/ha)": round(down_change, 6),
        }

    y_n_up, _ = model_predict(sample, n_val * 1.25, p_val, k_val, ph_val, district_id, model, scaler, metadata)
    y_n_down, _ = model_predict(sample, max(n_val * 0.75, 0.0), p_val, k_val, ph_val, district_id, model, scaler, metadata)
    rows.append(dominant_direction_result("Nitrogen", ("+25%", y_n_up * 350), ("-25%", y_n_down * 350)))

    y_p_up, _ = model_predict(sample, n_val, p_val * 1.25, k_val, ph_val, district_id, model, scaler, metadata)
    y_p_down, _ = model_predict(sample, n_val, max(p_val * 0.75, 0.0), k_val, ph_val, district_id, model, scaler, metadata)
    rows.append(dominant_direction_result("Phosphorus", ("+25%", y_p_up * 350), ("-25%", y_p_down * 350)))

    y_k_up, _ = model_predict(sample, n_val, p_val, k_val * 1.25, ph_val, district_id, model, scaler, metadata)
    y_k_down, _ = model_predict(sample, n_val, p_val, max(k_val * 0.75, 0.0), ph_val, district_id, model, scaler, metadata)
    rows.append(dominant_direction_result("Potassium", ("+25%", y_k_up * 350), ("-25%", y_k_down * 350)))

    ph_up = min(ph_val + 0.5, 10.0)
    ph_down = max(ph_val - 0.5, 3.0)
    y_ph_up, _ = model_predict(sample, n_val, p_val, k_val, ph_up, district_id, model, scaler, metadata)
    y_ph_down, _ = model_predict(sample, n_val, p_val, k_val, ph_down, district_id, model, scaler, metadata)
    rows.append(dominant_direction_result("Soil pH", ("+0.5", y_ph_up * 350), ("-0.5", y_ph_down * 350)))

    sim_rain_up = sample.copy()
    sim_rain_up[rain_cols] *= 1.10
    y_rain_up, _ = model_predict(sim_rain_up, n_val, p_val, k_val, ph_val, district_id, model, scaler, metadata)

    sim_rain_down = sample.copy()
    sim_rain_down[rain_cols] *= 0.90
    y_rain_down, _ = model_predict(sim_rain_down, n_val, p_val, k_val, ph_val, district_id, model, scaler, metadata)
    rows.append(dominant_direction_result("Rainfall", ("+10%", y_rain_up * 350), ("-10%", y_rain_down * 350)))

    sim_ndvi_up = sample.copy()
    sim_ndvi_up[ndvi_cols] *= 1.10
    y_ndvi_up, _ = model_predict(sim_ndvi_up, n_val, p_val, k_val, ph_val, district_id, model, scaler, metadata)

    sim_ndvi_down = sample.copy()
    sim_ndvi_down[ndvi_cols] *= 0.90
    y_ndvi_down, _ = model_predict(sim_ndvi_down, n_val, p_val, k_val, ph_val, district_id, model, scaler, metadata)
    rows.append(dominant_direction_result("NDVI", ("+10%", y_ndvi_up * 350), ("-10%", y_ndvi_down * 350)))

    out_df = pd.DataFrame(rows)
    top = out_df.iloc[np.argmax(np.abs(out_df["Change (t/ha)"].values))]["Scenario"]
    return out_df, top


def top_feature_importance(sample, n_val, p_val, k_val, ph_val, district_id, model, scaler, metadata):
    rain_cols = metadata["RAIN_COLS"]
    ndvi_cols = metadata["NDVI_COLS"]

    base, _ = model_predict(sample, n_val, p_val, k_val, ph_val, district_id, model, scaler, metadata)
    base_real = base * 350

    def simulate_impact(n_test, p_test, k_test, ph_test, rain_mult=1.0, ndvi_mult=1.0):
        sim = sample.copy()
        sim[rain_cols] *= rain_mult
        sim[ndvi_cols] *= ndvi_mult
        y, _ = model_predict(sim, n_test, p_test, k_test, ph_test, district_id, model, scaler, metadata)
        return (y * 350) - base_real

    rows = []

    # Use two-sided perturbations for more reliable local sensitivity.
    n_up = n_val * 1.25
    n_down = max(n_val * 0.75, 0.0)
    n_imp = max(abs(simulate_impact(n_up, p_val, k_val, ph_val)), abs(simulate_impact(n_down, p_val, k_val, ph_val)))
    rows.append({"Feature": "Nitrogen", "Impact (abs t/ha)": round(n_imp, 6)})

    p_up = p_val * 1.25
    p_down = max(p_val * 0.75, 0.0)
    p_imp = max(abs(simulate_impact(n_val, p_up, k_val, ph_val)), abs(simulate_impact(n_val, p_down, k_val, ph_val)))
    rows.append({"Feature": "Phosphorus", "Impact (abs t/ha)": round(p_imp, 6)})

    k_up = k_val * 1.25
    k_down = max(k_val * 0.75, 0.0)
    k_imp = max(abs(simulate_impact(n_val, p_val, k_up, ph_val)), abs(simulate_impact(n_val, p_val, k_down, ph_val)))
    rows.append({"Feature": "Potassium", "Impact (abs t/ha)": round(k_imp, 6)})

    ph_up = min(ph_val + 0.5, 10.0)
    ph_down = max(ph_val - 0.5, 3.0)
    ph_imp = max(abs(simulate_impact(n_val, p_val, k_val, ph_up)), abs(simulate_impact(n_val, p_val, k_val, ph_down)))
    rows.append({"Feature": "Soil pH", "Impact (abs t/ha)": round(ph_imp, 6)})

    rain_imp = max(
        abs(simulate_impact(n_val, p_val, k_val, ph_val, rain_mult=1.10, ndvi_mult=1.0)),
        abs(simulate_impact(n_val, p_val, k_val, ph_val, rain_mult=0.90, ndvi_mult=1.0)),
    )
    rows.append({"Feature": "Rainfall", "Impact (abs t/ha)": round(rain_imp, 6)})

    ndvi_imp = max(
        abs(simulate_impact(n_val, p_val, k_val, ph_val, rain_mult=1.0, ndvi_mult=1.10)),
        abs(simulate_impact(n_val, p_val, k_val, ph_val, rain_mult=1.0, ndvi_mult=0.90)),
    )
    rows.append({"Feature": "NDVI", "Impact (abs t/ha)": round(ndvi_imp, 6)})

    out_df = pd.DataFrame(rows).sort_values("Impact (abs t/ha)", ascending=False).reset_index(drop=True)
    return out_df


def apply_custom_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.75rem;
            padding-bottom: 1.5rem;
            max-width: 1200px;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 650;
            margin-top: 0.5rem;
            margin-bottom: 0.65rem;
        }
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            height: 2.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="AgroFusionNet", layout="wide")
    apply_custom_css()

    st.title("AgroFusionNet Decision Intelligence System")
    st.caption("Crop yield and stress intelligence with aligned inputs and scenario analysis.")

    df = load_data()
    metadata, scaler, district_map = load_artifacts()
    model = load_model(metadata, district_map)

    states = sorted(df["state"].dropna().astype(str).unique())
    st.sidebar.header("Selection")
    state = st.sidebar.selectbox("State", states)

    state_df = df[df["state"].str.lower() == state.lower()]
    districts = sorted(state_df["district"].dropna().astype(str).unique())
    district = st.sidebar.selectbox("District", districts)

    district_df = state_df[state_df["district"].str.lower() == district.lower()]
    crops = sorted(district_df["crop"].dropna().astype(str).unique())
    crop = st.sidebar.selectbox("Crop", crops)

    st.sidebar.markdown("---")
    st.sidebar.caption("Choose location and crop, then update inputs and run prediction.")

    crop_df = district_df[district_df["crop"] == crop].copy()
    sample = crop_df.iloc[[-1]].copy()
    district_id = int(sample["district_id"].values[0])

    default_n = float(sample["n"].values[0])
    default_p = float(sample["p"].values[0])
    default_k = float(sample["k"].values[0])
    default_ph = float(sample["ph"].values[0])
    default_annual_rainfall = float(sample["annual_rainfall"].values[0])
    default_rainfall_deviation = float(sample["rainfall_deviation"].values[0])
    default_ndvi = float(sample["ndvi"].values[0])
    default_ndvi_lag1 = float(sample["ndvi_lag1"].values[0])
    default_ndvi_lag2 = float(sample["ndvi_lag2"].values[0])
    default_year = float(sample["year"].values[0])
    selection_key = f"{state}|{district}|{crop}"

    st.caption("Defaults are loaded from the same base row used by CLI (latest selected crop record). You can edit any value.")

    st.markdown('<div class="section-title">Input Configuration</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            with st.container(border=True):
                st.markdown("**Soil and Rainfall Inputs**")
                n_val = st.number_input("Soil Nitrogen (N)", value=default_n, key=f"n_{selection_key}")
                p_val = st.number_input("Soil Phosphorus (P)", value=default_p, key=f"p_{selection_key}")
                k_val = st.number_input("Soil Potassium (K)", value=default_k, key=f"k_{selection_key}")
                ph_val = st.number_input("Soil pH", value=default_ph, key=f"ph_{selection_key}")
                annual_rainfall = st.number_input("Annual Rainfall", value=default_annual_rainfall, key=f"ar_{selection_key}")
                rainfall_deviation = st.number_input("Rainfall Deviation", value=default_rainfall_deviation, key=f"rd_{selection_key}")

        with col2:
            with st.container(border=True):
                st.markdown("**Vegetation and Time Inputs**")
                ndvi = st.number_input("NDVI", value=default_ndvi, key=f"ndvi_{selection_key}")
                ndvi_lag1 = st.number_input("NDVI Lag1", value=default_ndvi_lag1, key=f"ndvi1_{selection_key}")
                ndvi_lag2 = st.number_input("NDVI Lag2", value=default_ndvi_lag2, key=f"ndvi2_{selection_key}")
                year = st.number_input("Year", value=default_year, step=1.0, key=f"year_{selection_key}")

        run_clicked = st.form_submit_button("Run Prediction", type="primary")

    sample["annual_rainfall"] = annual_rainfall
    sample["rainfall_deviation"] = rainfall_deviation
    sample["ndvi"] = ndvi
    sample["ndvi_lag1"] = ndvi_lag1
    sample["ndvi_lag2"] = ndvi_lag2
    sample["year"] = year

    sample = fill_missing_features(sample, crop_df, df, metadata["ALL_FEATURES"])

    if run_clicked:
        yield_pred, stress_prob = model_predict(sample, n_val, p_val, k_val, ph_val, district_id, model, scaler, metadata)

        yield_real = yield_pred * 350
        prev_yield = pd.to_numeric(sample["yield_last_year"], errors="coerce").iloc[0] if "yield_last_year" in sample.columns else np.nan
        if pd.isna(prev_yield):
            prev_yield = float(pd.to_numeric(crop_df.get("yield_last_year"), errors="coerce").median()) if "yield_last_year" in crop_df.columns else 0.0
            if not np.isfinite(prev_yield):
                prev_yield = 0.0

        trend = yield_real - float(prev_yield)

        soil_score = soil_health_score(n_val, p_val, k_val)
        climate_score = climate_impact(sample.iloc[0])
        max_yield = float(pd.to_numeric(df["yield"], errors="coerce").max())
        if not np.isfinite(max_yield) or max_yield <= 0:
            max_yield = 1.0
        yield_score = yield_pred / max_yield

        si = sustainability_index(yield_score, stress_prob, soil_score, climate_score)
        risk = risk_level(si)

        causes = analyze_causes(sample.iloc[0], n_val, p_val, k_val)
        recs = recommendations(causes)

        st.markdown('<div class="section-title">Model Results</div>', unsafe_allow_html=True)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Predicted Yield (t/ha)", f"{yield_real:.3f}")
        metric_cols[1].metric("Yield Trend (t/ha)", f"{trend:.3f}")
        metric_cols[2].metric("Stress Probability (%)", f"{stress_prob * 100:.1f}")

        score_cols = st.columns(4)
        score_cols[0].metric("Stress Level", stress_level(stress_prob))
        score_cols[1].metric("Soil Health Score", f"{soil_score:.2f}")
        score_cols[2].metric("Climate Impact Score", f"{climate_score:.2f}")
        score_cols[3].metric("Sustainability Index", f"{si:.2f}")

        st.info(f"Risk Level: {risk}")

        tab1, tab2, tab3 = st.tabs(["Cause Analysis", "Sensitivity", "Feature Impact"])

        with tab1:
            details_col1, details_col2 = st.columns(2, gap="large")
            with details_col1:
                st.markdown("**Detected Causes**")
                if causes:
                    for c in causes:
                        st.write(f"- {c}")
                else:
                    st.write("- No major risk factors detected")
            with details_col2:
                st.markdown("**Recommendations**")
                for r in recs:
                    st.write(f"- {r}")

        with tab2:
            sens_df, top_factor = sensitivity_analysis(
                sample,
                n_val,
                p_val,
                k_val,
                ph_val,
                district_id,
                model,
                scaler,
                metadata,
            )
            st.dataframe(sens_df, use_container_width=True, hide_index=True)
            st.success(f"Most Influential Factor: {top_factor}")

        with tab3:
            imp_df = top_feature_importance(sample, n_val, p_val, k_val, ph_val, district_id, model, scaler, metadata)
            st.dataframe(imp_df.head(5), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
