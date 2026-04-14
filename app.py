import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

    /* Dark background */
    .stApp { background-color: #0a0e1a; }
    section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1e2d45; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 16px;
    }

    /* Headers */
    h1, h2, h3 { color: #e2e8f0 !important; }

    /* Prediction result box */
    .churn-yes {
        background: linear-gradient(135deg, #ff6b6b22, #ff6b6b11);
        border: 2px solid #ff6b6b;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 10px 0;
    }
    .churn-no {
        background: linear-gradient(135deg, #4ade8022, #4ade8011);
        border: 2px solid #4ade80;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 10px 0;
    }
    .big-result { font-size: 2.5rem; font-weight: 800; margin: 8px 0; }
    .result-sub { font-size: 1rem; color: #94a3b8; }

    /* Info boxes */
    .info-box {
        background: #111827;
        border-left: 4px solid #00d4ff;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 8px 0;
        color: #e2e8f0;
    }
    .warn-box {
        background: #111827;
        border-left: 4px solid #ff6b6b;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 8px 0;
        color: #e2e8f0;
    }

    /* Divider */
    hr { border-color: #1e2d45; }

    /* DataFrame */
    .dataframe { background: #111827 !important; }
</style>
""", unsafe_allow_html=True)

# ── PLOT THEME ────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f172a",
    "axes.facecolor":    "#1e293b",
    "axes.labelcolor":   "white",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "text.color":        "white",
    "axes.titlecolor":   "white",
    "axes.edgecolor":    "#334155",
    "grid.color":        "#1e3a5f",
    "legend.facecolor":  "#1e293b",
    "legend.edgecolor":  "#334155",
})

# ── LOAD MODELS ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("models.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_dataset():
    df = pd.read_csv("customer_churn_dataset-training-master.csv")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(columns=["CustomerID"], inplace=True)
    df["Churn"] = df["Churn"].astype(int)
    return df.sample(60000, random_state=42)

data = load_models()
MODELS       = data["models"]
SCALER       = data["scaler"]
FEATURE_NAMES = data["feature_names"]
METRICS      = data["metrics"]
MODEL_COLORS = {
    "Logistic Regression": "#60a5fa",
    "Decision Tree":       "#f97316",
    "Random Forest":       "#4ade80",
    "Gradient Boosting":   "#a78bfa",
}

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Churn Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home & Predict",
         "📊 EDA",
         "🤖 Model Performance",
         "🔵 Clustering",
         "📁 Batch Prediction"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Model Selection**")
    selected_model = st.selectbox(
        "Choose model for prediction",
        list(MODELS.keys()),
        index=2,    # Default: Random Forest
        label_visibility="collapsed"
    )
    st.markdown(f"""
    <div style='background:#1e293b;border-radius:10px;padding:12px;margin-top:8px;'>
        <div style='color:#94a3b8;font-size:0.75rem;'>Selected Model Stats</div>
        <div style='color:#4ade80;font-size:1.3rem;font-weight:800;margin:4px 0;'>{METRICS[selected_model]['accuracy']}%</div>
        <div style='color:#94a3b8;font-size:0.75rem;'>Accuracy</div>
        <div style='display:flex;gap:12px;margin-top:8px;'>
            <div><div style='color:#00d4ff;font-weight:700;'>{METRICS[selected_model]['f1']}</div><div style='color:#64748b;font-size:0.7rem;'>F1</div></div>
            <div><div style='color:#c77dff;font-weight:700;'>{METRICS[selected_model]['auc']}</div><div style='color:#64748b;font-size:0.7rem;'>AUC</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#64748b;font-size:0.75rem;'>
    📌 Dataset: 440K records<br>
    🎯 Target: Churn (Yes/No)<br>
    🧹 Preprocessing: StandardScaler<br>
    🔢 Features: 14 (after encoding)
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — HOME & PREDICT
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home & Predict":

    st.markdown("# 🧠 Customer Churn Prediction")
    st.markdown("*Enter customer details below to predict whether they will churn.*")
    st.markdown("---")

    # ── Top metric cards ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Dataset Size",  "440K Records")
    c2.metric("🎯 Best Accuracy", "99.70%",        "Decision Tree")
    c3.metric("🏆 Best AUC",      "1.0000",        "Random Forest")
    c4.metric("🔵 Clusters",      "4 Segments",    "K-Means")

    st.markdown("---")
    st.markdown("### 🔮 Single Customer Prediction")

    # ── Input form ────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        age    = st.slider("Age", 18, 65, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.slider("Tenure (months)", 1, 60, 24)

    with col2:
        st.markdown("**📞 Usage Behaviour**")
        usage_freq    = st.slider("Usage Frequency (per month)", 1, 30, 15)
        support_calls = st.slider("Support Calls", 0, 10, 3)
        last_interact = st.slider("Last Interaction (days ago)", 1, 30, 14)

    with col3:
        st.markdown("**💳 Billing & Contract**")
        payment_delay    = st.slider("Payment Delay (days)", 0, 30, 10)
        total_spend      = st.number_input("Total Spend ($)", 50, 1500, 600, step=10)
        subscription     = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract_length  = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

    st.markdown("")
    predict_btn = st.button("⚡ Predict Churn", type="primary", use_container_width=True)

    if predict_btn:
        # Build input vector
        gender_enc = 1 if gender == "Male" else 0

        input_dict = {
            "Age":                           age,
            "Gender":                        gender_enc,
            "Tenure":                        tenure,
            "Usage Frequency":               usage_freq,
            "Support Calls":                 support_calls,
            "Payment Delay":                 payment_delay,
            "Total Spend":                   total_spend,
            "Last Interaction":              last_interact,
            "Subscription Type_Basic":       1 if subscription == "Basic"    else 0,
            "Subscription Type_Premium":     1 if subscription == "Premium"  else 0,
            "Subscription Type_Standard":    1 if subscription == "Standard" else 0,
            "Contract Length_Annual":        1 if contract_length == "Annual"    else 0,
            "Contract Length_Monthly":       1 if contract_length == "Monthly"   else 0,
            "Contract Length_Quarterly":     1 if contract_length == "Quarterly" else 0,
        }
        input_df  = pd.DataFrame([input_dict])[FEATURE_NAMES]
        input_sc  = SCALER.transform(input_df)

        model     = MODELS[selected_model]
        pred      = model.predict(input_sc)[0]
        prob      = model.predict_proba(input_sc)[0]
        churn_prob = prob[1] * 100

        st.markdown("---")
        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            if pred == 1:
                st.markdown(f"""
                <div class="churn-yes">
                    <div style="font-size:3rem;">⚠️</div>
                    <div class="big-result" style="color:#ff6b6b;">LIKELY TO CHURN</div>
                    <div class="result-sub">Churn Probability: <b style="color:#ff6b6b">{churn_prob:.1f}%</b></div>
                    <div class="result-sub">Model: {selected_model}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="warn-box">
                    <b>🚨 Recommended Actions:</b><br>
                    • Offer loyalty discount or renewal incentive<br>
                    • Assign dedicated account manager<br>
                    • Resolve open support issues immediately<br>
                    • Suggest annual contract with benefits
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="churn-no">
                    <div style="font-size:3rem;">✅</div>
                    <div class="big-result" style="color:#4ade80;">LIKELY TO STAY</div>
                    <div class="result-sub">Retention Probability: <b style="color:#4ade80">{prob[0]*100:.1f}%</b></div>
                    <div class="result-sub">Model: {selected_model}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                    <b>💡 Suggested Actions:</b><br>
                    • Send appreciation / loyalty reward<br>
                    • Offer premium upgrade at discount<br>
                    • Keep engagement high via regular contact
                </div>
                """, unsafe_allow_html=True)

        with col_res2:
            # Probability gauge
            fig, ax = plt.subplots(figsize=(5, 4))
            wedge_colors = ["#4ade80", "#ff6b6b"]
            sizes = [prob[0]*100, prob[1]*100]
            wedges, _ = ax.pie(sizes, colors=wedge_colors, startangle=90,
                               wedgeprops={"edgecolor": "#0f172a", "linewidth": 3})
            ax.text(0, 0.1, f"{churn_prob:.1f}%", ha="center", va="center",
                    fontsize=22, fontweight="bold",
                    color="#ff6b6b" if pred==1 else "#4ade80")
            ax.text(0, -0.25, "Churn Prob.", ha="center", va="center",
                    fontsize=10, color="#94a3b8")
            ax.legend(["No Churn", "Churn"], loc="lower center",
                      bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9)
            ax.set_title("Prediction Probability", fontsize=12, fontweight="bold")
            st.pyplot(fig, use_container_width=True)

        # Risk factors
        st.markdown("#### 📌 Risk Factor Analysis")
        risk_cols = st.columns(5)
        factors = [
            ("Support Calls", support_calls, 3.5, "↑ High" if support_calls > 4 else "✓ Normal"),
            ("Payment Delay", payment_delay, 12,  "↑ High" if payment_delay > 15 else "✓ Normal"),
            ("Contract",      0, 0,                "⚠️ Monthly" if contract_length=="Monthly" else "✓ Long-term"),
            ("Tenure",        tenure, 30,           "⚠️ New"    if tenure < 12 else "✓ Loyal"),
            ("Total Spend",   total_spend, 600,     "↓ Low"    if total_spend < 400 else "✓ Good"),
        ]
        for col, (name, val, thresh, status) in zip(risk_cols, factors):
            color = "#ff6b6b" if "↑" in status or "⚠️" in status or "↓" in status else "#4ade80"
            col.markdown(f"""
            <div style='background:#111827;border:1px solid #1e2d45;border-radius:10px;padding:12px;text-align:center;'>
                <div style='color:#94a3b8;font-size:0.72rem;'>{name}</div>
                <div style='color:{color};font-weight:700;font-size:1rem;margin-top:4px;'>{status}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("*Understanding the dataset before modelling.*")
    st.markdown("---")

    df = load_dataset()

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",  f"{len(df):,}")
    c2.metric("Churned",        f"{df['Churn'].sum():,}", f"{df['Churn'].mean()*100:.1f}%")
    c3.metric("Avg Tenure",     f"{df['Tenure'].mean():.1f} mo")
    c4.metric("Avg Total Spend",f"${df['Total Spend'].mean():.0f}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "📦 Boxplots", "🌡️ Heatmap"])

    with tab1:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("Feature Distributions by Churn", fontsize=14, fontweight="bold")

        num_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend"]
        for ax, col in zip(axes.flatten(), num_cols):
            for label, color, nm in [(0,"#4ade80","No Churn"),(1,"#ff6b6b","Churn")]:
                ax.hist(df[df["Churn"]==label][col], bins=25, alpha=0.6,
                        color=color, label=nm, edgecolor="none")
            ax.set_title(col, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Categorical breakdown
        st.markdown("#### Contract & Subscription vs Churn")
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        for ax, col, title in [
            (axes2[0], "Contract Length",   "Contract Length vs Churn"),
            (axes2[1], "Subscription Type", "Subscription Type vs Churn"),
        ]:
            ct = df.groupby([col, "Churn"]).size().unstack(fill_value=0)
            ct.columns = ["No Churn", "Churn"]
            ct.plot(kind="bar", ax=ax, color=["#4ade80","#ff6b6b"],
                    edgecolor="white", linewidth=0.5)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.tick_params(axis="x", rotation=0)
            ax.set_xlabel("")
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    with tab2:
        fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
        fig3.suptitle("Boxplots — Feature vs Churn", fontsize=14, fontweight="bold")
        for ax, col in zip(axes3.flatten(), num_cols):
            bp = ax.boxplot(
                [df[df["Churn"]==0][col].values, df[df["Churn"]==1][col].values],
                patch_artist=True, labels=["No Churn", "Churn"],
                medianprops=dict(color="white", linewidth=2.5)
            )
            bp["boxes"][0].set_facecolor("#4ade8066")
            bp["boxes"][1].set_facecolor("#ff6b6b66")
            ax.set_title(col, fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    with tab3:
        le  = LabelEncoder()
        df2 = df.copy()
        df2["Gender"] = le.fit_transform(df2["Gender"].astype(str))
        df2 = pd.get_dummies(df2, columns=["Subscription Type","Contract Length"])
        corr = df2.select_dtypes(include=np.number).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig4, ax4 = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr, ax=ax4, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", vmin=-1, vmax=1,
                    linewidths=0.5, linecolor="#0f172a",
                    annot_kws={"size": 8})
        ax4.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
        ax4.tick_params(axis="x", rotation=45)
        st.pyplot(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("# 🤖 Model Performance")
    st.markdown("*Evaluation metrics, confusion matrices, and ROC curves.*")
    st.markdown("---")

    df = load_dataset()
    le  = LabelEncoder()
    df2 = df.copy()
    df2["Gender"] = le.fit_transform(df2["Gender"].astype(str))
    df2 = pd.get_dummies(df2, columns=["Subscription Type","Contract Length"])
    X   = df2.drop("Churn", axis=1)[FEATURE_NAMES]
    y   = df2["Churn"]
    X_sc= SCALER.transform(X)
    _, X_test, _, y_test = train_test_split(X_sc, y, test_size=0.2, random_state=42, stratify=y)

    # Metrics summary
    st.markdown("### 📊 All Models — Metrics Summary")
    cols = st.columns(4)
    for col, (name, m) in zip(cols, METRICS.items()):
        color = MODEL_COLORS[name]
        col.markdown(f"""
        <div style='background:#111827;border:1px solid #1e2d45;border-top:3px solid {color};
                    border-radius:12px;padding:16px;text-align:center;'>
            <div style='color:#94a3b8;font-size:0.8rem;margin-bottom:8px;'>{name}</div>
            <div style='color:{color};font-size:1.8rem;font-weight:800;'>{m['accuracy']}%</div>
            <div style='color:#64748b;font-size:0.72rem;margin-top:4px;'>
                F1: {m['f1']} &nbsp;|&nbsp; AUC: {m['auc']}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🔲 Confusion Matrices", "📈 ROC Curves", "📊 Metrics Bar", "📋 Report"])

    with tab1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")
        cmaps = ["Blues","Oranges","Greens","Purples"]
        for idx, (name, model) in enumerate(MODELS.items()):
            ax  = axes.flatten()[idx]
            yp  = model.predict(X_test)
            cm  = confusion_matrix(y_test, yp)
            tn, fp, fn, tp = cm.ravel()
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps[idx], ax=ax,
                        linewidths=2, linecolor="#0f172a",
                        xticklabels=["No Churn","Churn"],
                        yticklabels=["No Churn","Churn"],
                        annot_kws={"size":16,"weight":"bold"})
            ax.set_title(f"{name}\nTP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        for name, model in MODELS.items():
            ypr = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, ypr)
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, lw=2.5, color=MODEL_COLORS[name],
                     label=f"{name}  (AUC = {roc_auc:.4f})")
        ax2.plot([0,1],[0,1],"w--", lw=1.5, label="Random (AUC=0.50)")
        ax2.fill_between([0,1],[0,1], alpha=0.05, color="gray")
        ax2.set_xlim([0,1]); ax2.set_ylim([0,1.02])
        ax2.set_xlabel("False Positive Rate", fontsize=13)
        ax2.set_ylabel("True Positive Rate",  fontsize=13)
        ax2.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=11)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    with tab3:
        metric_names = ["Accuracy","Precision","Recall","F1 Score","AUC"]
        model_names  = list(MODELS.keys())
        x     = np.arange(len(metric_names))
        width = 0.2

        fig3, ax3 = plt.subplots(figsize=(14, 6))
        for i, (name, model) in enumerate(MODELS.items()):
            yp  = model.predict(X_test)
            ypr = model.predict_proba(X_test)[:,1]
            fpr,tpr,_ = roc_curve(y_test, ypr)
            vals = [
                accuracy_score(y_test, yp),
                precision_score(y_test, yp),
                recall_score(y_test, yp),
                f1_score(y_test, yp),
                auc(fpr, tpr),
            ]
            ax3.bar(x + i*width, vals, width, label=name,
                    color=MODEL_COLORS[name], alpha=0.85,
                    edgecolor="white", linewidth=0.5, zorder=3)

        ax3.set_xticks(x + width*1.5)
        ax3.set_xticklabels(metric_names, fontsize=12)
        ax3.set_ylim(0.85, 1.05)
        ax3.set_ylabel("Score")
        ax3.set_title("All Metrics Comparison", fontsize=14, fontweight="bold")
        ax3.legend(fontsize=10)
        ax3.grid(axis="y", alpha=0.3, zorder=0)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

        # Feature Importance
        st.markdown("#### 📌 Feature Importance — Random Forest")
        rf = MODELS["Random Forest"]
        fi = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values()

        fig4, ax4 = plt.subplots(figsize=(10, 7))
        threshold = fi.quantile(0.75)
        fi_colors = ["#ff6b6b" if v >= threshold else "#60a5fa" for v in fi.values]
        ax4.barh(fi.index, fi.values, color=fi_colors, edgecolor="none", height=0.7)
        for val, bar in zip(fi.values, ax4.patches):
            ax4.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                     f"{val*100:.2f}%", va="center", fontsize=9, color="white")
        ax4.set_title("Feature Importance (Random Forest)\nRed = Top 25%", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Importance Score")
        ax4.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)

    with tab4:
        sel = st.selectbox("Select model for report", list(MODELS.keys()))
        yp  = MODELS[sel].predict(X_test)
        st.code(classification_report(y_test, yp,
                target_names=["No Churn (0)", "Churn (1)"]), language="text")


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — CLUSTERING
# ══════════════════════════════════════════════════════════════
elif page == "🔵 Clustering":
    st.markdown("# 🔵 K-Means Clustering Analysis")
    st.markdown("*Unsupervised segmentation of customers into behaviour groups.*")
    st.markdown("---")

    df = load_dataset()
    le  = LabelEncoder()
    df2 = df.copy()
    df2["Gender"] = le.fit_transform(df2["Gender"].astype(str))
    df2 = pd.get_dummies(df2, columns=["Subscription Type","Contract Length"])
    X   = df2.drop("Churn", axis=1)[FEATURE_NAMES]
    X_sc= SCALER.transform(X)

    BEST_K = 4
    @st.cache_resource
    def fit_kmeans(k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sc)
        return km, labels

    km, cluster_labels = fit_kmeans(BEST_K)
    df["Cluster"] = cluster_labels

    CLUSTER_COLORS = ["#00d4ff","#ff6b6b","#ffd166","#06d6a0"]
    CLUSTER_NAMES  = {
        0: "Stable Mid-Spenders",
        1: "High-Risk Churners 🔴",
        2: "Moderate Risk",
        3: "Loyal High-Value",
    }

    # Reorder cluster labels so cluster with max churn = 1
    churn_by_cluster = df.groupby("Cluster")["Churn"].mean()
    high_risk = churn_by_cluster.idxmax()

    tab1, tab2, tab3 = st.tabs(["📍 Cluster Profiles", "📈 Elbow & Silhouette", "🗺️ PCA Scatter"])

    with tab1:
        profile_cols = ["Age","Tenure","Usage Frequency","Support Calls","Payment Delay","Total Spend","Churn"]
        profile = df.groupby("Cluster")[profile_cols].mean().round(2)
        profile["Churn %"] = (profile["Churn"]*100).round(1)
        profile["Size"]    = df["Cluster"].value_counts().sort_index()

        # Cluster cards
        c0,c1,c2,c3 = st.columns(4)
        for col_widget, cidx, cname in [
            (c0,0,"Stable Mid-Spenders"),
            (c1,1,"High-Risk Churners"),
            (c2,2,"Moderate Risk"),
            (c3,3,"Loyal High-Value"),
        ]:
            row = profile.loc[cidx]
            color  = CLUSTER_COLORS[cidx]
            churn_pct = row["Churn %"]
            col_widget.markdown(f"""
            <div style='background:#111827;border:1px solid #1e2d45;border-top:3px solid {color};
                        border-radius:14px;padding:16px;'>
                <div style='color:{color};font-weight:800;font-size:0.9rem;margin-bottom:10px;'>
                    Cluster {cidx} — {cname}
                </div>
                <div style='color:#ff6b6b;font-size:1.6rem;font-weight:800;'>{churn_pct}%</div>
                <div style='color:#64748b;font-size:0.72rem;margin-bottom:10px;'>Churn Rate</div>
                <div style='font-size:0.78rem;'>
                    <div style='display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e2d45;'>
                        <span style='color:#64748b;'>Size</span>
                        <span style='color:#e2e8f0;font-weight:600;'>{int(row["Size"]):,}</span>
                    </div>
                    <div style='display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e2d45;'>
                        <span style='color:#64748b;'>Avg Age</span>
                        <span style='color:#e2e8f0;font-weight:600;'>{row["Age"]}</span>
                    </div>
                    <div style='display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e2d45;'>
                        <span style='color:#64748b;'>Support Calls</span>
                        <span style='color:{"#ff6b6b" if row["Support Calls"]>4 else "#e2e8f0"};font-weight:600;'>{row["Support Calls"]}</span>
                    </div>
                    <div style='display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e2d45;'>
                        <span style='color:#64748b;'>Pay Delay</span>
                        <span style='color:{"#ff6b6b" if row["Payment Delay"]>14 else "#e2e8f0"};font-weight:600;'>{row["Payment Delay"]} days</span>
                    </div>
                    <div style='display:flex;justify-content:space-between;padding:3px 0;'>
                        <span style='color:#64748b;'>Total Spend</span>
                        <span style='color:#e2e8f0;font-weight:600;'>${row["Total Spend"]:.0f}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Charts
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Cluster Analysis Charts", fontsize=13, fontweight="bold")

        # Churn rate
        churn_rate = df.groupby("Cluster")["Churn"].mean()*100
        bars = axes[0].bar(range(BEST_K), churn_rate.values, color=CLUSTER_COLORS,
                           edgecolor="white", linewidth=1, zorder=3, width=0.5)
        for bar, val in zip(bars, churn_rate.values):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                         f"{val:.1f}%", ha="center", fontweight="bold", fontsize=12, color="white")
        axes[0].set_title("Churn Rate per Cluster", fontsize=11, fontweight="bold")
        axes[0].set_xticks(range(BEST_K)); axes[0].set_xticklabels([f"C{i}" for i in range(BEST_K)])
        axes[0].set_ylim(0,120); axes[0].grid(axis="y", alpha=0.3, zorder=0)

        # Support calls
        avg_sc = df.groupby("Cluster")["Support Calls"].mean()
        bars2 = axes[1].bar(range(BEST_K), avg_sc.values, color=CLUSTER_COLORS,
                            edgecolor="white", linewidth=1, zorder=3, width=0.5)
        for bar, val in zip(bars2, avg_sc.values):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                         f"{val:.2f}", ha="center", fontweight="bold", fontsize=12, color="white")
        axes[1].set_title("Avg Support Calls per Cluster", fontsize=11, fontweight="bold")
        axes[1].set_xticks(range(BEST_K)); axes[1].set_xticklabels([f"C{i}" for i in range(BEST_K)])
        axes[1].grid(axis="y", alpha=0.3, zorder=0)

        # Cluster sizes (pie)
        sizes = df["Cluster"].value_counts().sort_index()
        wedges, texts, autotexts = axes[2].pie(
            sizes.values,
            labels=[f"C{i}\n({v:,})" for i,v in zip(sizes.index,sizes.values)],
            colors=CLUSTER_COLORS, autopct="%1.1f%%", startangle=140,
            textprops={"fontsize":9,"color":"white"},
            wedgeprops={"edgecolor":"#0f172a","linewidth":2}
        )
        for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
        axes[2].set_title("Cluster Size Distribution", fontsize=11, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Profile heatmap
        st.markdown("#### 🌡️ Cluster Profile Heatmap (Normalized)")
        hm_cols = ["Age","Tenure","Usage Frequency","Support Calls","Payment Delay","Total Spend"]
        norm_p  = (profile[hm_cols] - profile[hm_cols].min()) / (profile[hm_cols].max() - profile[hm_cols].min())
        fig2, ax2 = plt.subplots(figsize=(10,4))
        sns.heatmap(norm_p.T, ax=ax2, cmap="YlOrRd", annot=True, fmt=".2f",
                    linewidths=1, linecolor="#0f172a",
                    annot_kws={"size":12,"weight":"bold"},
                    xticklabels=[f"Cluster {i}" for i in range(BEST_K)])
        ax2.set_title("Normalized Cluster Profiles", fontsize=12, fontweight="bold")
        ax2.tick_params(axis="x", rotation=0); ax2.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    with tab2:
        inertias=[]; sil_scores=[]
        K_range = range(2,9)
        for k in K_range:
            km2   = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl2  = km2.fit_predict(X_sc)
            inertias.append(km2.inertia_)
            sil_scores.append(silhouette_score(X_sc, lbl2, sample_size=3000, random_state=42))

        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
        fig3.suptitle("Optimal K Selection", fontsize=13, fontweight="bold")

        axes3[0].plot(list(K_range), inertias, "o-", color="#00d4ff", linewidth=2.5, markersize=8)
        axes3[0].axvline(x=4, color="#ff6b6b", linestyle="--", linewidth=2, label="Chosen K=4")
        axes3[0].fill_between(list(K_range), inertias, alpha=0.1, color="#00d4ff")
        axes3[0].set_title("Elbow Method (Inertia)", fontsize=12, fontweight="bold")
        axes3[0].set_xlabel("K"); axes3[0].set_ylabel("Inertia")
        axes3[0].legend(); axes3[0].grid(True, alpha=0.3)

        bar_clrs = ["#ff6b6b" if k==4 else "#00d4ff" for k in K_range]
        axes3[1].bar(list(K_range), sil_scores, color=bar_clrs, edgecolor="white", linewidth=0.8, zorder=3)
        axes3[1].set_title("Silhouette Score vs K", fontsize=12, fontweight="bold")
        axes3[1].set_xlabel("K"); axes3[1].set_ylabel("Silhouette Score")
        axes3[1].grid(axis="y", alpha=0.3, zorder=0)

        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    with tab3:
        pca    = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_sc)
        pv     = pca.explained_variance_ratio_.sum()*100

        fig4, ax4 = plt.subplots(figsize=(10,7))
        for c in range(BEST_K):
            mask = cluster_labels == c
            ax4.scatter(coords[mask,0], coords[mask,1],
                        c=CLUSTER_COLORS[c], label=f"Cluster {c}",
                        alpha=0.3, s=6, rasterized=True)
        ax4.set_title(f"PCA 2D Cluster Visualization\n(Explained Variance: {pv:.1f}%)",
                      fontsize=13, fontweight="bold")
        ax4.set_xlabel("PC1"); ax4.set_ylabel("PC2")
        ax4.legend(markerscale=4, fontsize=11)
        ax4.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — BATCH PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "📁 Batch Prediction":
    st.markdown("# 📁 Batch Prediction")
    st.markdown("*Upload a CSV file with customer data to predict churn for multiple customers at once.*")
    st.markdown("---")

    st.markdown("""
    <div class="info-box">
    <b>📌 Required CSV Columns:</b><br>
    Age, Gender (Male/Female), Tenure, Usage Frequency, Support Calls, Payment Delay,
    Subscription Type (Basic/Standard/Premium), Contract Length (Monthly/Quarterly/Annual),
    Total Spend, Last Interaction
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            st.markdown(f"✅ Loaded **{len(batch_df):,}** rows")
            st.dataframe(batch_df.head(), use_container_width=True)

            # Encode
            batch_proc = batch_df.copy()
            le2 = LabelEncoder()
            if "Gender" in batch_proc.columns:
                batch_proc["Gender"] = batch_proc["Gender"].map({"Male":1,"Female":0}).fillna(0).astype(int)
            if "CustomerID" in batch_proc.columns:
                batch_proc.drop(columns=["CustomerID"], inplace=True)
            if "Churn" in batch_proc.columns:
                batch_proc.drop(columns=["Churn"], inplace=True)

            batch_proc = pd.get_dummies(batch_proc, columns=["Subscription Type","Contract Length"])
            for col in FEATURE_NAMES:
                if col not in batch_proc.columns:
                    batch_proc[col] = 0
            batch_proc = batch_proc[FEATURE_NAMES]

            batch_sc   = SCALER.transform(batch_proc)
            model      = MODELS[selected_model]
            preds      = model.predict(batch_sc)
            probs      = model.predict_proba(batch_sc)[:,1]

            result_df  = batch_df.copy()
            result_df["Churn_Prediction"] = preds
            result_df["Churn_Probability_%"] = (probs*100).round(1)
            result_df["Risk_Level"] = pd.cut(
                probs*100, bins=[0,30,60,100],
                labels=["🟢 Low","🟡 Medium","🔴 High"]
            )

            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Customers",  f"{len(result_df):,}")
            c2.metric("Predicted Churn",  f"{preds.sum():,}", f"{preds.mean()*100:.1f}%")
            c3.metric("Predicted Stay",   f"{(preds==0).sum():,}", f"{(preds==0).mean()*100:.1f}%")

            st.dataframe(result_df, use_container_width=True)

            csv_out = result_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Predictions CSV",
                csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        # Sample download
        st.markdown("#### 📥 Download Sample CSV Template")
        sample = pd.DataFrame({
            "Age":               [25, 42, 60],
            "Gender":            ["Male","Female","Male"],
            "Tenure":            [5, 24, 48],
            "Usage Frequency":   [10, 20, 5],
            "Support Calls":     [6, 2, 8],
            "Payment Delay":     [20, 5, 25],
            "Subscription Type": ["Basic","Premium","Standard"],
            "Contract Length":   ["Monthly","Annual","Monthly"],
            "Total Spend":       [200, 800, 150],
            "Last Interaction":  [2, 20, 1],
        })
        st.dataframe(sample, use_container_width=True)
        st.download_button(
            "⬇️ Download Sample Template",
            sample.to_csv(index=False),
            file_name="sample_customers.csv",
            mime="text/csv",
        )
