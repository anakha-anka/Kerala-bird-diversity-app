"""Kerala Seasonal Bird Diversity - Streamlit demo.
Serves the pre-trained model (best_model.joblib) and saved evaluation outputs.
Run:  streamlit run app.py
"""
import warnings
from pathlib import Path

import json
import numpy as np
import pandas as pd
import plotly.express as px
import xgboost as xgb
import streamlit as st

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
ASSETS = HERE / "assets"
CLASSES = ["Low", "Medium", "High"]
COLORS = {"Low": "#d95f02", "Medium": "#e6ab02", "High": "#1b9e77"}

st.set_page_config(page_title="Kerala Bird Diversity ML", page_icon="🐦", layout="wide")


class PortableModel:
    """Loads the XGBoost booster from JSON (version-tolerant, unlike pickle)."""

    def __init__(self, model_path, imputer_path):
        cfg = json.load(open(imputer_path))
        self.features = cfg["features"]
        self.medians = dict(zip(self.features, cfg["medians"]))
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_path))

    def predict_proba(self, X):
        X = X[self.features].astype(float).fillna(self.medians)
        return self.booster.predict(xgb.DMatrix(X, feature_names=self.features))


@st.cache_resource
def load_model():
    m = PortableModel(HERE / "xgb_model.json", HERE / "imputer.json")
    return m, m.features


@st.cache_data
def load_grid():
    return pd.read_csv(HERE / "grid_season_features.csv")


@st.cache_data
def predict_all(_model, feats, df):
    proba = _model.predict_proba(df[feats])
    out = df.copy()
    out["pred"] = [CLASSES[i] for i in proba.argmax(1)]
    out["confidence"] = proba.max(1)
    return out


model, FEATS = load_model()
grid = load_grid()
scored = predict_all(model, FEATS, grid)

LABELS = {
    "n_checklists": "Number of checklists (survey effort)",
    "mean_observers_per_list": "Mean observers per checklist",
    "elevation": "Elevation (m)",
    "lat": "Latitude",
    "lon": "Longitude",
    "prop_water": "Share of waterbird species",
    "prop_raptor": "Share of raptor species",
    "prop_forest": "Share of forest-specialist species",
    "d_resident_status_Resident": "Share of resident species",
    "d_iucn_redlist_status_Least Concern": "Share of IUCN Least Concern species",
    "d_distribution_status_Large": "Share with 'Large' range",
    "d_distribution_status_Moderate": "Share with 'Moderate' range",
    "d_distribution_status_Very Large": "Share with 'Very Large' range",
    "d_soib_status_High": "Share with SoIB 'High' concern",
    "d_soib_status_Low": "Share with SoIB 'Low' concern",
    "d_endemicity_Not endemic": "Share of non-endemic species",
}

st.title("🐦 Seasonal Bird Diversity in Kerala")
st.caption("Machine-learning classification of grid cells into Low / Medium / High Shannon diversity, "
           "built on the Kerala Bird Atlas (Dryad). Model: XGBoost, 16 selected features, "
           "validated with spatial hold-out.")

tab_pred, tab_map, tab_explain, tab_eval = st.tabs(
    ["🔮 Predict", "🗺️ Map", "🔍 Explain", "📊 Model evidence"])

# ------------------------------------------------------------------ PREDICT
with tab_pred:
    st.subheader("Predict diversity class for a grid cell")
    left, right = st.columns([2, 1])

    with left:
        mode = st.radio("Start from", ["A real atlas grid cell", "Dataset median (custom)"],
                        horizontal=True)
        if mode == "A real atlas grid cell":
            c1, c2 = st.columns(2)
            season = c1.selectbox("Season", sorted(grid["season"].unique()))
            sub = grid[grid["season"] == season].reset_index(drop=True)
            idx = c2.selectbox("Grid cell", sub.index,
                               format_func=lambda i: f"Cell {i} ({sub.lat[i]:.2f}°N, {sub.lon[i]:.2f}°E)")
            base = sub.loc[idx, FEATS].astype(float)
            actual = sub.loc[idx, "diversity_class"]
        else:
            base = grid[FEATS].median().astype(float)
            actual = None

        st.markdown("**Adjust inputs** (what-if analysis)")
        vals = {}
        cols = st.columns(2)
        for k, f in enumerate(FEATS):
            lo, hi = float(grid[f].min()), float(grid[f].max())
            v = float(np.clip(base[f], lo, hi))
            vals[f] = cols[k % 2].slider(LABELS.get(f, f), lo, hi, v,
                                         step=max((hi - lo) / 200, 1e-3),
                                         key=f"{mode}-{f}-{base.name if hasattr(base, 'name') else ''}-{v:.4f}")

    with right:
        X = pd.DataFrame([vals])[FEATS]
        proba = model.predict_proba(X)[0]
        pred = CLASSES[int(proba.argmax())]
        st.metric("Predicted class", pred)
        if actual is not None:
            st.caption(f"Observed class for this cell: **{actual}**")
        fig = px.bar(x=CLASSES, y=proba, color=CLASSES, color_discrete_map=COLORS,
                     labels={"x": "", "y": "Probability"}, range_y=[0, 1])
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.info("Inputs are species-composition and survey-effort summaries for a grid cell, "
                "so the tool is a *what-if explorer*, not a field forecast.")

# ---------------------------------------------------------------------- MAP
with tab_map:
    st.subheader("Predicted vs observed diversity across Kerala grid cells")
    c1, c2, c3 = st.columns(3)
    s = c1.selectbox("Season ", ["All"] + sorted(grid["season"].unique()))
    what = c2.radio("Show", ["Predicted", "Observed"], horizontal=True)
    split = c3.selectbox("Cells", ["All", "test (unseen regions)", "train"])
    m = scored.copy()
    if s != "All":
        m = m[m["season"] == s]
    if split != "All":
        m = m[m["split"] == split.split()[0]]
    col = "pred" if what == "Predicted" else "diversity_class"
    fig = px.scatter_map(m, lat="lat", lon="lon", color=col, color_discrete_map=COLORS,
                         category_orders={col: CLASSES}, hover_data={"season": True, "shannon": ":.2f",
                                                                      "confidence": ":.2f", "lat": False, "lon": False},
                         zoom=6.2, height=620, map_style="open-street-map")
    fig.update_traces(marker=dict(size=12, opacity=0.8))
    st.plotly_chart(fig, width="stretch")
    te = m[m["split"] == "test"]
    if len(te):
        st.caption(f"Accuracy on the {len(te)} shown test cells (never seen in training): "
                   f"**{(te['pred'] == te['diversity_class']).mean():.1%}**")
    with st.expander("Static map from the training pipeline"):
        st.image(str(ASSETS / "predicted_diversity_map.png"))

# ------------------------------------------------------------------ EXPLAIN
with tab_explain:
    st.subheader("What drives the predictions? (SHAP)")
    imp = pd.read_csv(HERE / "shap_importance_main.csv", index_col=0)
    imp.index = [LABELS.get(i, i) for i in imp.index]
    top = imp["mean_abs_shap"].sort_values().tail(10)
    st.plotly_chart(px.bar(top, orientation="h", labels={"value": "Mean |SHAP|", "index": ""})
                    .update_layout(showlegend=False, height=420), width="stretch")
    c1, c2 = st.columns(2)
    c1.image(str(ASSETS / "shap_bar_main.png"), caption="Global importance by class")
    c2.image(str(ASSETS / "shap_beeswarm_high_main.png"), caption="Effect direction for the High class")
    st.markdown(
        "- Species-composition features (conservation status, residency, raptor share) dominate.\n"
        "- Latitude and survey effort add smaller but consistent signal.\n"
        "- **Caution:** composition is derived from the same checklists as the diversity label, "
        "so see the leakage-free baseline under *Model evidence*."
    )

# --------------------------------------------------------------------- EVAL
with tab_eval:
    st.subheader("Validation summary")
    fs = pd.read_csv(HERE / "final_summary.csv")
    main, base = fs.iloc[0], fs.iloc[1]
    ci = pd.read_csv(HERE / "test_metrics_bootstrap_ci.csv")
    rep = pd.read_csv(HERE / "repeated_spatial_holdout_summary.csv").set_index("model")

    c = st.columns(4)
    c[0].metric("Test accuracy", f"{main['accuracy']:.1%}")
    c[1].metric("Macro F1", f"{main['macro_f1']:.3f}")
    c[2].metric("ROC-AUC (OvR)", f"{main['roc_auc_ovr']:.3f}")
    c[3].metric("Repeated spatial hold-out F1",
                f"{rep.loc['main', 'macro_f1_mean']:.3f} ± {rep.loc['main', 'macro_f1_std']:.3f}")

    st.markdown("**Main model vs leakage-free baseline (effort + environment only)**")
    comp = pd.DataFrame({"Model": ["Main (composition + effort + environment)", "Baseline (no species-derived inputs)"],
                         "Accuracy": [main["accuracy"], base["accuracy"]],
                         "Macro F1": [main["macro_f1"], base["macro_f1"]],
                         "ROC-AUC": [main["roc_auc_ovr"], base["roc_auc_ovr"]]})
    st.dataframe(comp.style.format({c_: "{:.3f}" for c_ in comp.columns[1:]}), hide_index=True,
                 width="stretch")

    st.markdown("**Bootstrap 95% intervals (main model, test set)**")
    t = ci[(ci["model"] == "main") & ci["metric"].isin(["accuracy", "macro_f1", "auc_macro", "ece"])]
    st.dataframe(t[["metric", "estimate", "ci_low", "ci_high"]].round(3), hide_index=True,
                 width="stretch")

    st.markdown("**Algorithm comparison (spatial CV, macro F1)**")
    mc = pd.read_csv(HERE / "model_comparison.csv")
    st.plotly_chart(px.bar(mc.sort_values("f1"), x="f1", y="model", color="features", barmode="group",
                           error_x="f1_std", height=520, labels={"f1": "Macro F1", "model": ""}),
                    width="stretch")

    g1, g2 = st.columns(2)
    g1.image(str(ASSETS / "confusion_matrices_side_by_side.png"), caption="Confusion matrices")
    g2.image(str(ASSETS / "roc_curves.png"), caption="ROC curves")
    g3, g4 = st.columns(2)
    g3.image(str(ASSETS / "calibration_curves.png"), caption="Calibration")
    g4.image(str(ASSETS / "ablation.png"), caption="Feature-group ablation")

    st.warning("**Limitations:** Kerala Bird Atlas only (one region, 2 seasons); classes are tertiles of "
               "Shannon index; composition features are derived from the same checklists as the label; "
               "the effort+environment baseline reaches only ~56% accuracy, so predictions without "
               "species information are weak. Not a substitute for field surveys.")
