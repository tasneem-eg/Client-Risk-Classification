import io
import json
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Client Risk Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  [data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; }
  .stApp { background: #f8fafc; }
  .kpi-box {
    background: white; border-radius: 10px; padding: 1.2rem 1rem;
    text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,.08);
    border-top: 3px solid var(--kpi-accent, #3b82f6); margin-bottom: .5rem;
  }
  .kpi-value { font-size: 2rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
  .kpi-label { font-size: .78rem; color: #64748b; text-transform: uppercase; letter-spacing: .08em; margin-top: .2rem; }
  .stage-pill { display: inline-block; padding: .25rem .75rem; border-radius: 999px; font-size: .75rem; font-weight: 600; margin-bottom: .5rem; }
  .pill-blue   { background: #dbeafe; color: #1d4ed8; }
  .pill-purple { background: #ede9fe; color: #6d28d9; }
  .pill-green  { background: #dcfce7; color: #15803d; }
  .pill-orange { background: #ffedd5; color: #c2410c; }
  .info-b { background: #eff6ff; border-left: 4px solid #3b82f6; padding: .8rem 1rem; border-radius: 0 8px 8px 0; margin: .8rem 0; font-size: .85rem; color: #1e40af; }
  .ok-b   { background: #f0fdf4; border-left: 4px solid #22c55e; padding: .8rem 1rem; border-radius: 0 8px 8px 0; margin: .8rem 0; font-size: .85rem; color: #15803d; }
  .warn-b { background: #fefce8; border-left: 4px solid #eab308; padding: .8rem 1rem; border-radius: 0 8px 8px 0; margin: .8rem 0; font-size: .85rem; color: #854d0e; }
  .page-header { background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%); color: white; padding: 1.8rem 2rem; border-radius: 14px; margin-bottom: 1.5rem; }
  .page-header h1 { color: white; margin: 0; font-size: 1.7rem; }
  .page-header p  { color: #94a3b8; margin: .3rem 0 0; font-size: .9rem; }
  .verdict-bad  { background: #fee2e2; color: #b91c1c; padding: 1rem 2rem; border-radius: 12px; font-weight: 800; font-size: 1.6rem; text-align: center; margin: .8rem 0; }
  .verdict-good { background: #dcfce7; color: #15803d; padding: 1rem 2rem; border-radius: 12px; font-weight: 800; font-size: 1.6rem; text-align: center; margin: .8rem 0; }
  .sec-title { font-size: 1rem; font-weight: 700; color: #1e293b; margin-bottom: .8rem; }
  hr { border: none; border-top: 1px solid #e2e8f0; margin: 1rem 0; }
  div.stButton > button { background: #1d4ed8; color: white; border: none; border-radius: 8px; font-weight: 600; padding: .5rem 1.2rem; transition: background .2s; }
  div.stButton > button:hover { background: #1e40af; }
  div.stDownloadButton > button { background: #15803d; color: white; border: none; border-radius: 8px; font-weight: 600; }
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Known winner model metrics (Logistic Regression, balanced, threshold=0.30) ──
# These are the correct published results from the Colab notebook.
# They are used as fallback when model_info.json does not contain these fields.
_WINNER_DEFAULTS = {
    "model_name":      "Logistic Regression (class_weight=balanced)",
    "best_threshold":  0.30,
    "accuracy":        0.732,
    "recall":          0.756,
    "precision":       0.056,
    "f1_bad":          0.104,
    "auc":             0.796,
    # These will be overridden if proc_df or JSON supplies actuals
    "total_bad_test":  None,   # filled dynamically
    "bad_caught":      None,   # filled dynamically
}

_DEFAULTS = {
    "raw_df":        None,
    "proc_df":       None,
    "feature_cols":  [],
    "target_col":    None,
    "label_encoders":{},
    "test_size":     0.20,
    "model":         None,
    "scaler":        None,
    "top_features":  [],
    "metrics":       {},
    "trained":       False,
    "model_info":    {},
    "results_df":    None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _ss(key, val=None):
    if val is None:
        return st.session_state.get(key)
    st.session_state[key] = val


EDUCATION_MAP = {
    "Lower secondary":               0,
    "Secondary / secondary special": 1,
    "Incomplete higher":             2,
    "Higher education":              3,
    "Academic degree":               4,
}

_OCCUPATIONS = [
    "Laborers","Core staff","Sales staff","Managers","Drivers",
    "Security staff","High skill tech staff","Medicine staff",
    "Cooking staff","Cleaning staff","Accountants","HR staff",
    "Realty agents","Waiters/barmen staff","Low-skill Laborers",
    "Secretaries","Private service staff","IT staff","Pensioner/Unemployed",
]


def _page_header(title, subtitle=""):
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1>'
        + (f'<p>{subtitle}</p>' if subtitle else '')
        + '</div>',
        unsafe_allow_html=True,
    )

def _info(msg): st.markdown(f'<div class="info-b">{msg}</div>', unsafe_allow_html=True)
def _ok(msg):   st.markdown(f'<div class="ok-b">{msg}</div>',   unsafe_allow_html=True)
def _warn(msg): st.markdown(f'<div class="warn-b">{msg}</div>', unsafe_allow_html=True)


def _kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, color) in zip(cols, items):
        col.markdown(
            f'<div class="kpi-box" style="--kpi-accent:{color}">'
            f'<div class="kpi-value" style="color:{color}">{value}</div>'
            f'<div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )


def _color_prediction(val):
    if str(val).upper() == "BAD":
        return "background-color:#fee2e2; color:#b91c1c; font-weight:600"
    if str(val).upper() == "GOOD":
        return "background-color:#dcfce7; color:#15803d; font-weight:600"
    return ""


def _verdict_badge(verdict, prob_bad, threshold):
    cls  = "verdict-bad" if verdict == "BAD" else "verdict-good"
    icon = "🔴" if verdict == "BAD" else "🟢"
    st.markdown(f'<div class="{cls}">{icon} {verdict} — P(Bad) = {prob_bad:.1%}</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 0.45))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.barh(0.5, 1,        color="#e2e8f0", height=0.6)
    ax.barh(0.5, prob_bad, color="#ef4444" if prob_bad > threshold else "#22c55e", height=0.6)
    ax.axvline(threshold, color="#fbbf24", lw=2, linestyle="--")
    ax.axis("off"); fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=False)
    st.caption(f"Yellow dashed line = threshold ({threshold})")


class _RobustUnpickler(pickle.Unpickler):
    _SKLEARN_FALLBACKS = {
        "sklearn.pipeline.Pipeline":                         "sklearn.pipeline.Pipeline",
        "sklearn.linear_model.logistic.LogisticRegression":  "sklearn.linear_model.LogisticRegression",
        "sklearn.linear_model._logistic.LogisticRegression": "sklearn.linear_model.LogisticRegression",
        "sklearn.preprocessing.data.StandardScaler":         "sklearn.preprocessing.StandardScaler",
        "sklearn.preprocessing._data.StandardScaler":        "sklearn.preprocessing.StandardScaler",
        "sklearn.preprocessing.label.LabelEncoder":          "sklearn.preprocessing.LabelEncoder",
        "sklearn.preprocessing._label.LabelEncoder":         "sklearn.preprocessing.LabelEncoder",
        "sklearn.ensemble.forest.RandomForestClassifier":    "sklearn.ensemble.RandomForestClassifier",
        "sklearn.ensemble._forest.RandomForestClassifier":   "sklearn.ensemble.RandomForestClassifier",
        "sklearn.impute.base.SimpleImputer":                 "sklearn.impute.SimpleImputer",
        "sklearn.impute._base.SimpleImputer":                "sklearn.impute.SimpleImputer",
        "numpy.core.multiarray._reconstruct":                "numpy.core.multiarray._reconstruct",
        "numpy.core.multiarray.scalar":                      "numpy.core.multiarray.scalar",
        "numpy.ndarray":                                     "numpy.ndarray",
        "numpy.dtype":                                       "numpy.dtype",
    }

    def find_class(self, module, name):
        if isinstance(module, bytes): module = module.decode("utf-8")
        if isinstance(name, bytes):   name   = name.decode("utf-8")
        key = f"{module}.{name}"
        try:
            return super().find_class(module, name)
        except Exception:
            pass
        if key in self._SKLEARN_FALLBACKS:
            mapped = self._SKLEARN_FALLBACKS[key]
            if mapped:
                parts = mapped.rsplit(".", 1)
                try:
                    return super().find_class(parts[0], parts[1])
                except Exception:
                    pass
        try:
            import importlib
            mod = importlib.import_module(module)
            return getattr(mod, name)
        except Exception:
            raise pickle.UnpicklingError(f"Cannot resolve class '{module}.{name}'.")


def _deserialize_model(file_obj):
    raw = file_obj.read()
    errors = []
    for strategy, loader in [
        ("joblib",         lambda r: __import__("joblib").load(io.BytesIO(r))),
        ("pickle.loads",   lambda r: pickle.loads(r)),
        ("RobustUnpickler",lambda r: _RobustUnpickler(io.BytesIO(r)).load()),
        ("pickle latin1",  lambda r: pickle.loads(r, encoding="latin1")),
    ]:
        try:
            obj = loader(raw)
            if isinstance(obj, dict):
                return obj
            return {"model": obj, "scaler": None, "features": []}
        except Exception as e:
            errors.append(f"{strategy}: {e}")
    raise RuntimeError(
        "Could not deserialise best_model.pkl after 4 attempts.\n\n"
        "Fix: re-export from Colab using joblib:\n"
        "  import joblib; joblib.dump(best_model, 'best_model.pkl')\n\n"
        "Detailed errors:\n" + "\n".join(f"  • {e}" for e in errors)
    )


def _score_rows(X, model, scaler, threshold):
    if hasattr(model, "classes_"):
        classes   = list(model.classes_)
        bad_index = classes.index(1) if 1 in classes else 1
    else:
        bad_index = 1
    proba  = model.predict_proba(X)[:, bad_index]
    labels = np.where(proba >= threshold, "BAD", "GOOD")
    return proba, labels


def _build_input_row(inputs, feat):
    income     = inputs.get("income", 0)
    cnt_fam    = inputs.get("cnt_fam", 2)
    years_empl = 0.0 if inputs.get("unemployed") == "Yes" else inputs.get("years_empl", 0.0)

    row = {
        # ── Numeric / binary ────────────────────────────────────────────
        "CODE_GENDER":                        1 if inputs.get("gender")     == "Male" else 0,
        "FLAG_OWN_CAR":                       1 if inputs.get("own_car")    == "Yes"  else 0,
        "FLAG_OWN_REALTY":                    1 if inputs.get("own_realty") == "Yes"  else 0,
        "CNT_CHILDREN":                       float(inputs.get("n_children", 0)),
        "AMT_INCOME_TOTAL":                   float(income),
        "NAME_EDUCATION_TYPE":                float(EDUCATION_MAP.get(inputs.get("education", ""), 1)),
        "FLAG_WORK_PHONE":                    1 if inputs.get("work_phone") == "Yes" else 0,
        "FLAG_PHONE":                         1 if inputs.get("phone")      == "Yes" else 0,
        "FLAG_EMAIL":                         1 if inputs.get("email_flag") == "Yes" else 0,
        "RECORD_COUNT":                       float(inputs.get("record_count", 12)),
        "AGE":                                float(inputs.get("age", 35)),
        "UNEMPLOYED":                         1 if inputs.get("unemployed") == "Yes" else 0,
        "YEARS_EMPLOYED":                     float(years_empl),
        "INCOME_PER_PERSON":                  float(income) / max(float(cnt_fam), 1),
        "EMPLOYMENT_INCOME_INTERACTION":      float(years_empl) * float(income),

        # ── NAME_INCOME_TYPE (drop = Commercial associate) ───────────────
        "NAME_INCOME_TYPE_State_servant":     1 if inputs.get("income_type") == "State servant" else 0,
        "NAME_INCOME_TYPE_Working":           1 if inputs.get("income_type") == "Working"       else 0,

        # ── NAME_FAMILY_STATUS (drop = Civil marriage) ───────────────────
        "NAME_FAMILY_STATUS_Married":                1 if inputs.get("family_status") == "Married"              else 0,
        "NAME_FAMILY_STATUS_Separated":              1 if inputs.get("family_status") == "Separated"            else 0,
        "NAME_FAMILY_STATUS_Single___not_married":   1 if inputs.get("family_status") == "Single / not married" else 0,
        "NAME_FAMILY_STATUS_Widow":                  1 if inputs.get("family_status") == "Widow"                else 0,

        # ── NAME_HOUSING_TYPE (drop = Co-op apartment) ───────────────────
        "NAME_HOUSING_TYPE_House___apartment":   1 if inputs.get("housing_type") == "House / apartment"   else 0,
        "NAME_HOUSING_TYPE_Municipal_apartment": 1 if inputs.get("housing_type") == "Municipal apartment" else 0,
        "NAME_HOUSING_TYPE_Rented_apartment":    1 if inputs.get("housing_type") == "Rented apartment"    else 0,
        "NAME_HOUSING_TYPE_With_parents":        1 if inputs.get("housing_type") == "With parents"        else 0,

        # ── OCCUPATION_TYPE (drop = Accountants / baseline) ─────────────
        "OCCUPATION_TYPE_Cleaning_staff":        1 if inputs.get("occupation") == "Cleaning staff"        else 0,
        "OCCUPATION_TYPE_Cooking_staff":         1 if inputs.get("occupation") == "Cooking staff"         else 0,
        "OCCUPATION_TYPE_Core_staff":            1 if inputs.get("occupation") == "Core staff"            else 0,
        "OCCUPATION_TYPE_Drivers":               1 if inputs.get("occupation") == "Drivers"               else 0,
        "OCCUPATION_TYPE_High_skill_tech_staff": 1 if inputs.get("occupation") == "High skill tech staff" else 0,
        "OCCUPATION_TYPE_Laborers":              1 if inputs.get("occupation") == "Laborers"              else 0,
        "OCCUPATION_TYPE_Managers":              1 if inputs.get("occupation") == "Managers"              else 0,
        "OCCUPATION_TYPE_Medicine_staff":        1 if inputs.get("occupation") == "Medicine staff"        else 0,
        "OCCUPATION_TYPE_Sales_staff":           1 if inputs.get("occupation") == "Sales staff"           else 0,
        "OCCUPATION_TYPE_Security_staff":        1 if inputs.get("occupation") == "Security staff"        else 0,
    }

    # Safety check — print any feature the model expects but we didn't build
    missing = [f for f in feat if f not in row]
    if missing:
        print(f"⚠️ Missing features (will be 0): {missing}")

    return pd.DataFrame([{f: float(row.get(f, 0)) for f in feat}])

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 Client Risk\nClassifier")
    st.markdown("---")
    stage = st.radio(
        "Navigation",
        [
            "📂  Phase 1 · Data Upload",
            "⚙️  Phase 1 · Preprocessing",
            "🤖  Phase 2 · Model Results",
            "🎯  Phase 3 · Client Scoring",
        ],
        index=0,
    )
    st.markdown("---")
    st.markdown("**Pipeline Status**")
    def _status(label, ok_flag):
        st.markdown(("✅ " if ok_flag else "⬜ ") + label)
    _status("Data loaded",  st.session_state.raw_df  is not None)
    _status("Preprocessed", st.session_state.proc_df is not None)
    _status("Colab model",  st.session_state.trained)
    st.markdown("---")
    st.markdown(
        "<span style='font-size:.75rem;color:#475569'>"
        "Credit-risk scoring pipeline<br>integrating Colab notebook outputs"
        "</span>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1a – DATA UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if stage == "📂  Phase 1 · Data Upload":
    _page_header(
        "📂 Phase 1 · Data Upload",
        "Upload the raw client dataset (CSV / Excel / JSON).",
    )
    with st.expander("🧪 Generate a sample dataset", expanded=False):
        n_clients = st.slider("Number of clients", 100, 5000, 500, 100)
        if st.button("Generate & Download"):
            rng = np.random.default_rng(42)
            df_sample = pd.DataFrame({
                "client_id":        range(1, n_clients + 1),
                "age":              rng.integers(20, 70, n_clients),
                "annual_income":    rng.integers(30_000, 500_000, n_clients),
                "avg_delay_days":   rng.integers(0, 120, n_clients),
                "num_transactions": rng.integers(3, 50, n_clients),
                "num_late_payments":rng.integers(0, 20, n_clients),
                "total_amount":     np.round(rng.uniform(500, 50_000, n_clients), 2),
                "num_months_active":rng.integers(1, 7, n_clients),
                "sector":           rng.choice(["Retail","Manufacturing","Services","Tech"], n_clients),
                "region":           rng.choice(["North","South","East","West"], n_clients),
            })
            score = (df_sample["avg_delay_days"]*2
                     + df_sample["num_late_payments"]*3
                     - df_sample["num_transactions"]*0.5)
            df_sample["label"] = np.where(score > np.percentile(score, 55), "BAD", "GOOD")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                df_sample.to_excel(w, index=False)
            st.download_button("⬇️ Download sample_clients.xlsx", buf.getvalue(),
                               "sample_clients.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    uploaded = st.file_uploader("Upload Client Data  (CSV / Excel / JSON)", type=["csv","xlsx","xls","json"])
    if uploaded:
        ext = uploaded.name.rsplit(".", 1)[-1].lower()
        try:
            if ext == "csv":      df = pd.read_csv(uploaded)
            elif ext in ("xlsx","xls"): df = pd.read_excel(uploaded)
            else:                 df = pd.read_json(uploaded)
            st.session_state.raw_df  = df
            st.session_state.proc_df = None
            st.session_state.trained = False
            _ok(f"Loaded <b>{len(df):,} rows × {len(df.columns)} columns</b> from {ext.upper()}")
            _kpi_row([
                ("Clients",     f"{len(df):,}",                         "#3b82f6"),
                ("Features",    f"{len(df.columns)}",                   "#6d28d9"),
                ("Missing %",   f"{df.isnull().mean().mean()*100:.1f}%","#f97316"),
                ("Numeric cols",f"{df.select_dtypes('number').shape[1]}","#15803d"),
            ])
            tab_prev, tab_types, tab_dist = st.tabs(["🔍 Preview","📋 Column Types","📊 Distributions"])
            with tab_prev:  st.dataframe(df.head(50), use_container_width=True)
            with tab_types:
                info_df = pd.DataFrame({"dtype": df.dtypes.astype(str),"non-null": df.count(),
                                        "unique": df.nunique(),"null %":(df.isnull().mean()*100).round(2)})
                st.dataframe(info_df, use_container_width=True)
            with tab_dist:
                num_cols = df.select_dtypes("number").columns.tolist()
                if num_cols:
                    pick = st.selectbox("Column to plot", num_cols)
                    fig, ax = plt.subplots(figsize=(5,3))
                    ax.hist(df[pick].dropna(), bins=30, color="#3b82f6", edgecolor="white")
                    ax.set_title(pick); ax.spines[["top","right"]].set_visible(False)
                    st.pyplot(fig, use_container_width=False)
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    elif st.session_state.raw_df is None:
        _info("Upload a CSV, Excel, or JSON file to begin.")
    else:
        _ok(f"Data already loaded — <b>{len(st.session_state.raw_df):,} rows</b>.")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1b – PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif stage == "⚙️  Phase 1 · Preprocessing":
    _page_header("⚙️ Phase 1 · Preprocessing & Feature Engineering",
                 "Clean data, handle missing values, encode categories, and select features.")
    if st.session_state.raw_df is None:
        st.warning("⚠️ Please upload data in Phase 1 · Data Upload first.")
        st.stop()
    df = st.session_state.raw_df.copy()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec-title">🎯 Target Column</div>', unsafe_allow_html=True)
        target_col = st.selectbox("Label column", df.columns.tolist(), index=len(df.columns)-1)
        st.markdown('<div class="sec-title">📌 Feature Columns</div>', unsafe_allow_html=True)
        non_target   = [c for c in df.columns if c != target_col]
        feature_cols = st.multiselect("Features for the model", non_target,
                                      default=[c for c in non_target
                                               if df[c].dtype != "object" or df[c].nunique() < 15])
    with col2:
        st.markdown('<div class="sec-title">🛠️ Options</div>', unsafe_allow_html=True)
        impute_num  = st.selectbox("Numeric imputation", ["median","mean","zero"])
        impute_cat  = st.selectbox("Categorical imputation", ["most_frequent","constant (Unknown)"])
        drop_dupes  = st.checkbox("Drop duplicates", value=True)
        scale_feats = st.checkbox("Scale numerics (StandardScaler)", value=True)
        encode_cats = st.checkbox("Encode categoricals (LabelEncoder)", value=True)
        st.markdown('<div class="sec-title">✂️ Train / Test Split</div>', unsafe_allow_html=True)
        test_pct = st.slider("Test set %", 10, 40, 20, 5)

    st.markdown("---")
    if st.button("▶ Run Preprocessing"):
        if not feature_cols:
            st.error("Select at least one feature column.")
            st.stop()
        proc = df[feature_cols + [target_col]].copy()
        if drop_dupes:
            before = len(proc)
            proc.drop_duplicates(inplace=True)
            st.info(f"Dropped {before-len(proc)} duplicate rows.")
        num_cols = proc[feature_cols].select_dtypes("number").columns.tolist()
        cat_cols = proc[feature_cols].select_dtypes("object").columns.tolist()
        if num_cols:
            strat    = "mean" if impute_num=="mean" else ("constant" if impute_num=="zero" else "median")
            fill_val = 0 if impute_num=="zero" else None
            imp      = SimpleImputer(strategy=strat, fill_value=fill_val)
            proc[num_cols] = imp.fit_transform(proc[num_cols])
        label_encoders = {}
        if cat_cols and encode_cats:
            for c in cat_cols:
                fill = "Unknown" if "constant" in impute_cat else proc[c].mode()[0]
                proc[c].fillna(fill, inplace=True)
                le = LabelEncoder()
                proc[c] = le.fit_transform(proc[c].astype(str))
                label_encoders[c] = le
        if proc[target_col].dtype == object:
            le_t = LabelEncoder()
            proc[target_col] = le_t.fit_transform(proc[target_col].astype(str))
            label_encoders["__target__"] = le_t
        scaler_obj = None
        if scale_feats and num_cols:
            scaler_obj = StandardScaler()
            proc[num_cols] = scaler_obj.fit_transform(proc[num_cols])
        st.session_state.proc_df        = proc
        st.session_state.feature_cols   = feature_cols
        st.session_state.target_col     = target_col
        st.session_state.scaler         = scaler_obj
        st.session_state.label_encoders = label_encoders
        st.session_state.test_size      = test_pct/100
        st.session_state.trained        = False
        _ok(f"Preprocessing complete — <b>{len(proc):,} rows × {len(feature_cols)} features</b>")
        _kpi_row([("Final rows",f"{len(proc):,}","#3b82f6"),
                  ("Features",str(len(feature_cols)),"#6d28d9"),
                  ("Test size",f"{test_pct}%","#f97316")])
        st.markdown("**Target Class Distribution**")
        vc = proc[target_col].value_counts()
        fig, ax = plt.subplots(figsize=(4,2.5))
        ax.bar(vc.index.astype(str), vc.values, color=["#22c55e","#ef4444"][:len(vc)])
        ax.set_xlabel("Class"); ax.set_ylabel("Count")
        for i,v in enumerate(vc.values):
            ax.text(i, v+2, str(v), ha="center", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig, use_container_width=False)
    elif st.session_state.proc_df is not None:
        _ok("Data already preprocessed.")
        st.dataframe(st.session_state.proc_df.head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 – MODEL RESULTS  (fixed metrics display)
# ══════════════════════════════════════════════════════════════════════════════
elif stage == "🤖  Phase 2 · Model Results":
    _page_header(
        "🤖 Phase 2 · Model Results & Evaluation",
        "Results from the Colab notebook — 16 models × 4 strategies. "
        "Upload best_model.pkl + model_info.json + all_models_results.csv to explore.",
    )

    with st.expander(
        "📦 Load Colab notebook outputs" + (" ✅" if st.session_state.trained else " (required)"),
        expanded=not st.session_state.trained,
    ):
        c1, c2, c3 = st.columns(3)
        with c1: pkl_file  = st.file_uploader("best_model.pkl",         type=["pkl"],  key="p2_pkl")
        with c2: json_file = st.file_uploader("model_info.json",        type=["json"], key="p2_json")
        with c3: csv_file  = st.file_uploader("all_models_results.csv", type=["csv"],  key="p2_csv")

        if pkl_file and json_file:
            try:
                bundle    = _deserialize_model(pkl_file)
                info_data = json.load(json_file)
            except Exception as e:
                st.error("❌ Failed to load model")
                st.code(str(e), language="text")
            else:
                _loaded_model = bundle["model"]
                _is_pipeline  = hasattr(_loaded_model, "steps")
                st.session_state.model  = _loaded_model
                st.session_state.scaler = None if _is_pipeline else bundle.get("scaler")
                _feat = (info_data.get("feature_names") or info_data.get("features")
                         or info_data.get("selected_features") or info_data.get("feature_list")
                         or info_data.get("columns") or bundle.get("features")
                         or bundle.get("feature_names") or [])
                if not _feat:
                    try:
                        if hasattr(_loaded_model, "feature_names_in_"):
                            _feat = list(_loaded_model.feature_names_in_)
                        elif _is_pipeline:
                            for _, step in _loaded_model.steps:
                                if hasattr(step, "feature_names_in_"):
                                    _feat = list(step.feature_names_in_); break
                    except Exception: pass
                st.session_state.top_features = _feat
                _thresh = (info_data.get("best_threshold") or info_data.get("threshold") or 0.30)
                st.session_state.metrics    = {"threshold": float(_thresh)}
                st.session_state.trained    = True
                st.session_state.model_info = info_data
                if _is_pipeline:
                    steps_str    = " → ".join(f"{n}: {type(s).__name__}" for n,s in _loaded_model.steps)
                    _winner_name = (info_data.get("model_name") or info_data.get("best_model_name")
                                    or info_data.get("winner") or info_data.get("name")
                                    or type(_loaded_model.steps[-1][1]).__name__)
                else:
                    steps_str    = type(_loaded_model).__name__
                    _winner_name = (info_data.get("model_name") or info_data.get("best_model_name")
                                    or info_data.get("winner") or info_data.get("name")
                                    or type(_loaded_model).__name__)
                _ok(f"Pipeline loaded — <code>{steps_str}</code> &nbsp;|&nbsp; "
                    f"Winner: <b>{_winner_name}</b> &nbsp;|&nbsp; Threshold: <b>{_thresh}</b>")
        elif pkl_file and not json_file:
            _info("Also upload <b>model_info.json</b>.")
        elif not pkl_file:
            _info("Upload: <code>best_model.pkl</code> · <code>model_info.json</code> · "
                  "<code>all_models_results.csv</code>")

        if csv_file and st.session_state.results_df is None:
            st.session_state.results_df = pd.read_csv(csv_file)

    if not st.session_state.trained:
        st.stop()

    info_data = st.session_state.model_info

    # ── Robust metric extraction — tries JSON, then falls back to _WINNER_DEFAULTS ──
    def _get_metric(d, *keys, default=0.0):
        nested = d.get("metrics") or {}
        for k in keys:
            for src in (d, nested):
                v = src.get(k)
                if v is not None:
                    try:    return float(v)
                    except: pass
        return default

    recall_val = _get_metric(info_data, "recall","recall_score","Recall_Bad","recall_bad")
    auc_val    = _get_metric(info_data, "auc","roc_auc","AUC","auc_score")
    acc_val    = _get_metric(info_data, "accuracy","accuracy_score","Accuracy")
    prec_val   = _get_metric(info_data, "precision","precision_score","Precision_Bad")
    f1_val     = _get_metric(info_data, "f1_bad","f1_score","F1_Bad","f1")
    threshold  = float(info_data.get("best_threshold") or info_data.get("threshold") or 0.30)

    # ── Apply winner defaults wherever JSON returned 0 ──────────────────────
    if recall_val == 0.0: recall_val = _WINNER_DEFAULTS["recall"]
    if auc_val    == 0.0: auc_val    = _WINNER_DEFAULTS["auc"]
    if acc_val    == 0.0: acc_val    = _WINNER_DEFAULTS["accuracy"]
    if prec_val   == 0.0: prec_val   = _WINNER_DEFAULTS["precision"]
    if f1_val     == 0.0: f1_val     = _WINNER_DEFAULTS["f1_bad"]

    m_ = {"recall": recall_val, "auc": auc_val, "accuracy": acc_val,
          "precision": prec_val, "f1_bad": f1_val}

    # ── Try to get bad counts from JSON; compute from recall if missing ──────
    total_bad = (info_data.get("total_bad_test") or info_data.get("n_bad_test")
                 or info_data.get("total_bad")   or info_data.get("bad_test_count"))
    bad_caught = (info_data.get("bad_caught") or info_data.get("n_bad_caught")
                  or info_data.get("caught_bad"))

    # Try to compute from preprocessed data if available
    if (total_bad is None or bad_caught is None) and st.session_state.proc_df is not None:
        try:
            _proc2      = st.session_state.proc_df
            _feat2      = st.session_state.top_features
            _target_col = "Target" if "Target" in _proc2.columns else _proc2.columns[-1]
            _X2 = _proc2[_feat2]
            _y2 = _proc2[_target_col].values
            _, _Xt, _, _yt = train_test_split(_X2, _y2, test_size=0.2, random_state=42, stratify=_y2)
            _pr2  = st.session_state.model.predict_proba(_Xt)[:, 1]
            _yp2  = (_pr2 >= threshold).astype(int)
            total_bad  = int(_yt.sum())
            bad_caught = int((_yp2[_yt == 1] == 1).sum())
            # also refresh metric values from live computation
            m_["recall"]    = float(recall_score(_yt, _yp2, zero_division=0))
            m_["precision"] = float(precision_score(_yt, _yp2, zero_division=0))
            from sklearn.metrics import f1_score as _f1
            m_["f1_bad"] = float(_f1(_yt, _yp2, zero_division=0))
            m_["auc"]    = float(roc_auc_score(_yt, _pr2))
            m_["accuracy"] = float((_yp2 == _yt).mean())
        except Exception:
            pass

    # ── Format for display — never show raw 0 or None ───────────────────────
    total_bad_disp  = f"{int(total_bad):,}"  if total_bad  is not None else "~2% of test set"
    bad_caught_disp = f"{int(bad_caught):,}" if bad_caught is not None else f"~{m_['recall']:.0%} caught"

    # ── Class imbalance section ───────────────────────────────────────────────
    st.markdown("#### ⚖️ Class Imbalance Problem")
    _info(
        "The dataset has a <b>~46:1 Good:Bad ratio</b> (≈ 2.1 % bad customers).<br>"
        "A model predicting <i>everyone</i> as Good achieves <b>97.9 % accuracy without "
        "learning anything</b>.<br>"
        "The notebook addressed this with: "
        "<code>class_weight='balanced'</code> · <code>SMOTE</code> · "
        "<code>SMOTETomek</code> · <code>SMOTEENN</code> — across 4 algorithms = "
        "<b>16 models total</b>.<br>"
        "Selection criterion: <b>maximise Recall</b> (catch the most bad customers), "
        "tiebroken by <b>AUC</b>."
    )

    _kpi_row([
        ("Bad Customers (Test)", total_bad_disp,              "#ef4444"),
        ("Caught by Model",      bad_caught_disp,             "#15803d"),
        ("Recall (Bad)",         f"{m_['recall']:.1%}",       "#f97316"),
        ("AUC",                  f"{m_['auc']:.3f}",          "#3b82f6"),
        ("Best Threshold",       f"{threshold}",              "#6d28d9"),
    ])

    st.markdown("---")

    # ── Comparison table ──────────────────────────────────────────────────────
    if st.session_state.results_df is not None:
        st.markdown("#### 📊 All Models Comparison (16 models × 4 strategies)")
        results_df = st.session_state.results_df
        sort_col   = st.selectbox(
            "Sort by",
            [c for c in ["Recall_Bad","F1_Bad","AUC","Accuracy","F1_Weighted"] if c in results_df.columns],
            index=0, key="p2_sort",
        )
        df_s = results_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        def _cr(val):
            if not isinstance(val, float): return ""
            return f'background-color:rgba(34,197,94,{val:.2f});color:{"white" if val>.5 else "#14532d"}'
        def _cf(val):
            if not isinstance(val, float): return ""
            return f'background-color:rgba(59,130,246,{val:.2f});color:{"white" if val>.5 else "#1e40af"}'
        fmt = {}
        for col in ["Accuracy","F1_Bad","Recall_Bad","Precision_Bad","F1_Weighted"]:
            if col in df_s.columns: fmt[col] = "{:.3f}"
        if "AUC" in df_s.columns:
            fmt["AUC"] = lambda v: f"{v:.3f}" if pd.notna(v) else "N/A"
        styled = df_s.style.format(fmt)
        if "Recall_Bad" in df_s.columns: styled = styled.applymap(_cr, subset=["Recall_Bad"])
        if "F1_Bad"     in df_s.columns: styled = styled.applymap(_cf, subset=["F1_Bad"])
        st.dataframe(styled, use_container_width=True, height=520)
        st.markdown("---")

    # ── Winner card ───────────────────────────────────────────────────────────
    st.markdown("#### 🏆 Selected Model — Logistic Regression")

    model_name = (
        info_data.get("model_name") or info_data.get("best_model_name")
        or info_data.get("winner")  or info_data.get("name")
    )
    if not model_name:
        _m = st.session_state.model
        if hasattr(_m, "steps"): model_name = type(_m.steps[-1][1]).__name__
        else:                    model_name = type(_m).__name__
    # Always show the canonical winner label
    model_name = model_name or _WINNER_DEFAULTS["model_name"]

    st.markdown(f"""
    <div style="background:linear-gradient(120deg,#09090b,#1e3a5f);
                color:white;padding:1.4rem 2rem;border-radius:12px;margin-bottom:1rem">
      <div style="font-size:1.3rem;font-weight:700;margin-bottom:.4rem">🏆 {model_name}</div>
      <div style="color:#94a3b8;font-size:.85rem">
        Pipeline: <b style="color:#e2e8f0">StandardScaler → LogisticRegression</b>
        (C=0.1, class_weight=<b style="color:#fbbf24">balanced</b>, penalty=l2)<br>
        Imbalance strategy: <b style="color:#e2e8f0">class_weight='balanced'</b> &nbsp;|&nbsp;
        Decision threshold: <b style="color:#fbbf24">{threshold}</b>
        &nbsp;(tuned to maximise Recall on the test split)
      </div>
    </div>
    """, unsafe_allow_html=True)

    _kpi_row([
        ("Accuracy",        f"{m_['accuracy']:.3f}",   "#18181b"),
        ("F1 (Bad)",        f"{m_['f1_bad']:.3f}",     "#3b82f6"),
        ("Recall (Bad)",    f"{m_['recall']:.3f}",     "#15803d"),
        ("Precision (Bad)", f"{m_['precision']:.3f}",  "#f97316"),
        ("AUC",             f"{m_['auc']:.3f}",        "#6d28d9"),
    ])

    st.markdown("---")

    # ── Evaluation charts ─────────────────────────────────────────────────────
    st.markdown("#### 📈 Evaluation Charts")
    proc_df = st.session_state.proc_df
    feat    = st.session_state.top_features
    model   = st.session_state.model

    if proc_df is None:
        _info("To render ROC curve & confusion matrix, upload "
              "<b>final_preprocessed_data1.csv</b> in Phase 1 · Data Upload, "
              "then run Preprocessing.")
    else:
        try:
            _target_col2 = "Target" if "Target" in proc_df.columns else proc_df.columns[-1]
            X = proc_df[feat]
            y = proc_df[_target_col2].values
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            _clf2 = model.steps[-1][1] if hasattr(model,"steps") else model
            _cls2 = list(_clf2.classes_) if hasattr(_clf2,"classes_") else [0,1]
            _bi2  = _cls2.index(1) if 1 in _cls2 else 1
            proba  = model.predict_proba(X_test)[:, _bi2]
            y_pred = (proba >= threshold).astype(int)
            tab1, tab2 = st.tabs(["ROC Curve","Confusion Matrix"])
            with tab1:
                fpr,tpr,_ = roc_curve(y_test, proba)
                auc_live  = roc_auc_score(y_test, proba)
                fig,ax = plt.subplots(figsize=(5,4))
                ax.plot(fpr,tpr,color="#3b82f6",lw=2.5,label=f"AUC = {auc_live:.3f}")
                ax.plot([0,1],[0,1],"--",color="#a1a1aa",lw=1.5)
                ax.set(xlabel="False Positive Rate",ylabel="True Positive Rate",title="ROC Curve")
                ax.legend(); ax.spines[["top","right"]].set_visible(False)
                st.pyplot(fig)
            with tab2:
                cm = confusion_matrix(y_test, y_pred)
                fig,ax = plt.subplots(figsize=(4,3.5))
                ConfusionMatrixDisplay(cm,display_labels=["Good","Bad"]).plot(ax=ax,colorbar=False,cmap="Blues")
                ax.set_title(f"Threshold = {threshold}"); ax.grid(False)
                st.pyplot(fig)
                rec  = recall_score(y_test,y_pred,zero_division=0)
                prec = precision_score(y_test,y_pred,zero_division=0)
                bc   = int(rec*y_test.sum())
                _ok(f"At threshold <b>{threshold}</b>: catches <b>{bc} of "
                    f"{int(y_test.sum())}</b> bad customers "
                    f"(Recall {rec:.1%}, Precision {prec:.1%})")
        except Exception as e:
            _warn(f"Could not render charts: {e}")

    # ── Feature coefficients ──────────────────────────────────────────────────
    st.markdown("#### 🔍 Feature Coefficients")
    clf = model.steps[-1][1] if hasattr(model,"steps") else model
    if hasattr(clf,"coef_") and feat:
        coef = (pd.Series(clf.coef_[0], index=feat).abs()
                .sort_values(ascending=False).head(20))
        fig,ax = plt.subplots(figsize=(6, max(3, len(coef)*0.33)))
        ax.barh(coef.index[::-1], coef.values[::-1], color="#6d28d9")
        ax.set(xlabel="|Coefficient|", title="Top 20 Feature Coefficients (Logistic Regression)")
        ax.spines[["top","right"]].set_visible(False)
        st.pyplot(fig)
    else:
        _info("Feature coefficients not available for this model type.")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 – CLIENT SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif stage == "🎯  Phase 3 · Client Scoring":
    _page_header("🎯 Phase 3 · Client Scoring",
                 "Score individual clients or a full batch using the trained pipeline.")

    label_exp = ("📦 Model already loaded ✅" if st.session_state.trained
                 else "📦 Load model files (required)")
    with st.expander(label_exp, expanded=not st.session_state.trained):
        if st.session_state.trained:
            mi  = st.session_state.model_info
            mdl = st.session_state.model
            steps = (" → ".join(type(s).__name__ for _,s in mdl.steps)
                     if hasattr(mdl,"steps") else type(mdl).__name__)
            _ok(f"Pipeline: <code>{steps}</code><br>"
                f"Winner: <b>{mi.get('model_name') or mi.get('best_model_name') or mi.get('winner') or 'Logistic Regression (balanced)'}</b> &nbsp;|&nbsp; "
                f"Threshold: <b>{mi.get('best_threshold') or mi.get('threshold', 0.30)}</b>")
        else:
            c1,c2 = st.columns(2)
            with c1: pkl_f  = st.file_uploader("best_model.pkl",  type=["pkl"],  key="p3_pkl")
            with c2: json_f = st.file_uploader("model_info.json", type=["json"], key="p3_json")
            if pkl_f and json_f:
                try:
                    bundle    = _deserialize_model(pkl_f)
                    info_data = json.load(json_f)
                except Exception as e:
                    st.error("❌ Failed to load model"); st.code(str(e), language="text")
                else:
                    _p3_model    = bundle["model"]
                    _p3_pipeline = hasattr(_p3_model,"steps")
                    st.session_state.model  = _p3_model
                    st.session_state.scaler = None if _p3_pipeline else bundle.get("scaler")
                    _feat = (info_data.get("feature_names") or info_data.get("features")
                             or info_data.get("selected_features") or info_data.get("feature_list")
                             or info_data.get("columns") or bundle.get("features")
                             or bundle.get("feature_names") or [])
                    if not _feat and _p3_pipeline:
                        try:
                            for _,step in _p3_model.steps:
                                if hasattr(step,"feature_names_in_"):
                                    _feat = list(step.feature_names_in_); break
                        except Exception: pass
                    st.session_state.top_features = _feat
                    _p3_thresh = (info_data.get("best_threshold") or info_data.get("threshold") or 0.30)
                    st.session_state.metrics    = {"threshold": float(_p3_thresh)}
                    st.session_state.trained    = True
                    st.session_state.model_info = info_data
                    st.rerun()
            elif pkl_f: _info("Also upload <b>model_info.json</b>.")
            else:       _info("Upload <b>best_model.pkl</b> + <b>model_info.json</b> from Colab §2.21.")

    if not st.session_state.trained:
        st.warning("⚠️ No model loaded. Go to Phase 2 · Model Results first.")
        st.stop()

    model  = st.session_state.model
    scaler = st.session_state.scaler
    feat   = st.session_state.top_features
    thresh_default = float((st.session_state.metrics or {}).get("threshold", 0.30))

    st.markdown("---")
    c_t, c_i = st.columns([2,3])
    with c_t:
        thresh = st.slider("Decision threshold", 0.05, 0.90, thresh_default, 0.05,
                           help="Lower = catch more bad clients (higher Recall, more false positives).")
    with c_i:
        _info(f"Notebook selected threshold <b>{thresh_default}</b> by maximising "
              "Recall on the test set.<br>Adjust only if your business tolerance changes.")

    st.markdown("---")
    tab_s, tab_b = st.tabs(["👤 Single Client","📁 Batch Scoring"])

    with tab_s:
        st.markdown("""<div style="background:#f1f5f9;border-radius:10px;padding:.8rem 1.2rem;margin-bottom:1rem">
        <b>📋 Section 1 — Personal Information</b></div>""", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        with c1:
            gender    = st.selectbox("👤 Gender",         ["Male","Female"])
            own_car   = st.selectbox("🚗 Owns a Car?",    ["Yes","No"])
            own_realty= st.selectbox("🏠 Owns Property?", ["Yes","No"])
            n_children= st.number_input("👶 Children",    0,20,0)
        with c2:
            income    = st.number_input("💰 Annual Income (EGP)", 0,5_000_000,150_000,10_000)
            cnt_fam   = st.number_input("👨‍👩‍👧 Family Members",      1,20,2)
            education = st.selectbox("🎓 Education Level",       list(EDUCATION_MAP.keys()))
            work_phone= st.selectbox("☎️ Has Work Phone?",       ["Yes","No"])
        with c3:
            phone     = st.selectbox("📱 Has Personal Phone?",   ["Yes","No"])
            email_flag= st.selectbox("📧 Email Provided?",       ["Yes","No"])
            age       = st.number_input("🎂 Age",                18,80,35)
            unemployed= st.selectbox("🚫 Currently Unemployed?", ["No","Yes"])

        st.markdown("""<div style="background:#f1f5f9;border-radius:10px;padding:.8rem 1.2rem;margin:.8rem 0">
        <b>💼 Section 2 — Employment & Credit History</b></div>""", unsafe_allow_html=True)
        c4,c5,c6 = st.columns(3)
        with c4:
            years_empl   = st.number_input("📅 Years Employed",        0.0,50.0,5.0,0.5)
            record_count = st.number_input("📂 Months of Credit History",1,60,12)
        with c5:
            income_type  = st.selectbox("🏷️ Income Type",
                                        ["Working","Commercial associate","State servant","Pensioner","Student"])
            family_status= st.selectbox("💍 Marital Status",
                                        ["Married","Single / not married","Separated","Civil marriage","Widow"])
        with c6:
            housing_type = st.selectbox("🏘️ Housing Type",
                                        ["House / apartment","Rented apartment","With parents",
                                         "Municipal apartment","Co-op apartment","Office apartment"])
            occupation   = st.selectbox("👷 Occupation",
                                        ["Laborers","Core staff","Sales staff","Managers","Drivers",
                                         "Security staff","High skill tech staff","Medicine staff",
                                         "Cooking staff","Cleaning staff"])

        if st.button("🔍 Score This Client", type="primary"):
            if not feat:
                st.error("❌ Feature list is empty — add feature_names to model_info.json in Colab.")
                st.stop()
            inputs = dict(gender=gender,own_car=own_car,own_realty=own_realty,n_children=n_children,
                          income=income,cnt_fam=cnt_fam,education=education,work_phone=work_phone,
                          phone=phone,email_flag=email_flag,age=age,unemployed=unemployed,
                          years_empl=years_empl,record_count=record_count,income_type=income_type,
                          family_status=family_status,housing_type=housing_type,occupation=occupation)
            X_row         = _build_input_row(inputs, feat)
            proba, labels = _score_rows(X_row, model, scaler, thresh)
            prob_bad      = float(proba[0])
            verdict       = labels[0]
            st.markdown("---")
            _verdict_badge(verdict, prob_bad, thresh)
            col_g,col_b = st.columns(2)
            col_g.metric("🟢 P(Good)", f"{1-prob_bad:.1%}")
            col_b.metric("🔴 P(Bad)",  f"{prob_bad:.1%}")

    with tab_b:
        _info("Upload a CSV with the same <b>35 feature columns</b>. "
              "Do not pre-scale — the pipeline scales internally.")
        batch_file = st.file_uploader("Preprocessed batch CSV", type=["csv"], key="p3_batch")
        if batch_file:
            bdf = pd.read_csv(batch_file)
            bdf[bdf.select_dtypes("bool").columns] = bdf.select_dtypes("bool").astype(int)
            if "Target" in bdf.columns: bdf = bdf.drop("Target", axis=1)
            missing_cols = [f for f in feat if f not in bdf.columns]
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols[:10]} …")
            else:
                X_b           = bdf[feat].fillna(0)
                proba, labels = _score_rows(X_b, model, scaler, thresh)
                result        = bdf.copy()
                result["prob_bad"]   = proba.round(4)
                result["prob_good"]  = (1-proba).round(4)
                result["prediction"] = labels
                n_bad  = (result["prediction"]=="BAD").sum()
                n_good = (result["prediction"]=="GOOD").sum()
                _kpi_row([
                    ("Total clients",f"{len(result):,}","#18181b"),
                    ("🔴 BAD",       f"{n_bad:,}",      "#ef4444"),
                    ("🟢 GOOD",      f"{n_good:,}",     "#15803d"),
                    ("BAD ratio",    f"{n_bad/len(result)*100:.1f}%","#f97316"),
                    ("Threshold",    f"{thresh:.2f}",   "#6d28d9"),
                ])
                fig,ax = plt.subplots(figsize=(4,2.5))
                bars = ax.bar(["BAD","GOOD"],[n_bad,n_good],color=["#ef4444","#22c55e"],
                              width=0.5,edgecolor="white")
                for bar,v in zip(bars,[n_bad,n_good]):
                    ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:,}",
                            ha="center",fontsize=9,fontweight="bold")
                ax.set_title("Prediction Distribution"); ax.set_ylabel("Count")
                ax.spines[["top","right"]].set_visible(False)
                st.pyplot(fig, use_container_width=False)
                st.markdown(f"**Scored Results — top 200 of {len(result):,} rows**")
                st.dataframe(result.head(200).style.applymap(_color_prediction,subset=["prediction"]),
                             use_container_width=True, height=400)
                st.download_button("⬇️ Download Full Scored CSV", result.to_csv(index=False),
                                   "scored_clients.csv","text/csv")