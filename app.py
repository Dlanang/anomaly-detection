"""
Streamlit Anomaly Detection Dashboard — pure NumPy, no scikit-learn, no API key
Run: python -m streamlit run app.py
"""

import json
import warnings
import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config + CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]  { background:#0b0f1a; }
[data-testid="stSidebar"]           { background:#0f1520 !important; border-right:1px solid #1e2740; }
section[data-testid="stSidebar"] *  { color:#94a3b8 !important; }
section[data-testid="stSidebar"] h2 { color:#e2e8f0 !important; }
h1,h2,h3,h4                         { color:#e2e8f0 !important; }
hr                                   { border-color:#1e2740 !important; }

[data-testid="metric-container"]  { background:#111827; border:1px solid #1e2740; border-radius:12px; padding:1rem 1.25rem; }
[data-testid="stMetricLabel"]     { color:#6b7a99 !important; font-size:.75rem !important; text-transform:uppercase; letter-spacing:.07em; }
[data-testid="stMetricValue"]     { color:#e2e8f0 !important; font-size:1.8rem !important; font-weight:700 !important; }

[data-testid="stInfo"]    { background:#0f2340 !important; border-left:3px solid #3b82f6 !important; border-radius:8px; }
[data-testid="stSuccess"] { background:#0f2e1e !important; border-left:3px solid #22c55e !important; border-radius:8px; }
[data-testid="stWarning"] { background:#2a1f0a !important; border-left:3px solid #f59e0b !important; border-radius:8px; }
[data-testid="stError"]   { background:#2a0f0f !important; border-left:3px solid #ef4444 !important; border-radius:8px; }

[data-testid="stButton"]>button {
    background:linear-gradient(135deg,#3b82f6,#6366f1) !important;
    color:#fff !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important; padding:.5rem 1.2rem !important;
}
[data-testid="stButton"]>button:hover { opacity:.85 !important; }
[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
[data-testid="stExpander"]  { background:#111827 !important; border:1px solid #1e2740 !important; border-radius:10px !important; }

.s-card { background:#111827; border:1px solid #1e2740; border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:1.2rem; }
.s-card-title { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:#6b7a99; margin-bottom:.9rem; font-weight:600; }

/* report styles */
.rpt-block  { background:#0f1520; border:1px solid #1e2740; border-radius:10px; padding:1.1rem 1.4rem; margin-bottom:.75rem; }
.rpt-label  { font-size:.68rem; text-transform:uppercase; letter-spacing:.08em; color:#6b7a99; margin-bottom:.4rem; font-weight:600; }
.rpt-value  { color:#cbd5e1; font-size:.9rem; line-height:1.7; }
.rpt-badge  { display:inline-block; border-radius:6px; padding:3px 12px; font-size:.78rem; font-weight:700; }
.sev-critical { background:#3d0f0f; color:#ef4444; }
.sev-high     { background:#3d2a0a; color:#f59e0b; }
.sev-medium   { background:#0f2340; color:#3b82f6; }
.sev-low      { background:#0f2e1e; color:#22c55e; }

.find-item  { display:flex; gap:10px; align-items:flex-start; padding:.55rem 0; border-bottom:1px solid #1e2740; }
.find-icon  { width:7px; height:7px; border-radius:50%; background:#ef4444; margin-top:7px; flex-shrink:0; }
.find-text  { color:#94a3b8; font-size:.87rem; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chart palette
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG="0b0f1a"; CARD_BG="#111827"; BORDER="#1e2740"
TEXT_PRI="#e2e8f0"; TEXT_MUT="#6b7a99"
BLUE="#3b82f6"; CRIMSON="#ef4444"; AMBER="#f59e0b"; GREEN="#22c55e"

def _mpl():
    plt.rcParams.update({
        "figure.facecolor":"#0b0f1a","axes.facecolor":CARD_BG,
        "axes.edgecolor":BORDER,"axes.labelcolor":TEXT_MUT,
        "axes.titlecolor":TEXT_PRI,"xtick.color":TEXT_MUT,"ytick.color":TEXT_MUT,
        "text.color":TEXT_PRI,"grid.color":BORDER,"grid.linestyle":"--","grid.alpha":.5,
        "axes.titlesize":13,"axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10,
        "legend.facecolor":CARD_BG,"legend.edgecolor":BORDER,"legend.fontsize":10,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Pure-NumPy Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────
def _c(n):
    if n<=1: return 0.0
    return 2*(np.log(n-1)+0.5772156649)-2*(n-1)/n

class _ITree:
    __slots__=("md","sf","sv","l","r","size","leaf")
    def __init__(self,md): self.md=md;self.sf=self.sv=self.l=self.r=None;self.size=0;self.leaf=False
    def fit(self,X,d=0):
        self.size=len(X)
        if d>=self.md or self.size<=1: self.leaf=True;return self
        f=np.random.randint(0,X.shape[1]);col=X[:,f];lo,hi=col.min(),col.max()
        if lo==hi: self.leaf=True;return self
        self.sf=f;self.sv=np.random.uniform(lo,hi);m=col<self.sv
        self.l=_ITree(self.md).fit(X[m],d+1);self.r=_ITree(self.md).fit(X[~m],d+1);return self
    def path(self,x,d=0):
        if self.leaf: return d+_c(self.size)
        return(self.l if x[self.sf]<self.sv else self.r).path(x,d+1)

class IForest:
    def __init__(self,n=100,s=256,cont=0.05,seed=42):
        self.n=n;self.s=s;self.cont=cont;self.seed=seed
        self.trees=[];self.thr=None;self.cn=1.0
    def fit(self,X):
        np.random.seed(self.seed);N=len(X);s=min(self.s,N)
        self.cn=_c(s);md=int(np.ceil(np.log2(s))) if s>1 else 1
        self.trees=[_ITree(md).fit(X[np.random.choice(N,s,replace=False)]) for _ in range(self.n)]
        sc=self._raw(X);self.thr=np.percentile(sc,100*(1-self.cont));return self
    def _raw(self,X):
        avg=np.array([np.mean([t.path(x) for t in self.trees]) for x in X])
        return -avg/self.cn if self.cn else np.zeros(len(X))
    def score(self,X): return self._raw(np.array(X,float))
    def predict(self,X): return np.where(self.score(X)>=self.thr,1,-1)

class Scaler:
    def fit_transform(self,X):
        X=np.array(X,float);self.mu=X.mean(0);self.sd=X.std(0)
        self.sd[self.sd==0]=1.0;return (X-self.mu)/self.sd

def pca2(X):
    X=np.array(X,float);X-=X.mean(0)
    _,vecs=np.linalg.eigh(np.cov(X.T))
    return X@vecs[:,::-1][:,:2]

# ─────────────────────────────────────────────────────────────────────────────
# Analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_analysis(file_bytes:bytes, cont:float):
    logs=[]
    try:
        for ln in file_bytes.decode("utf-8").splitlines():
            if ln.strip(): logs.append(json.loads(ln))
        df=pd.DataFrame(logs)
    except Exception as e: return None,None,str(e)
    if df.empty: return None,None,"File produced no rows."

    for col in list(df.columns):
        if any(isinstance(x,dict) for x in df[col].dropna()):
            try:
                mask=df[col].apply(lambda x:isinstance(x,dict))
                flat=pd.json_normalize(df.loc[mask,col]).add_prefix(f"{col}_")
                flat.index=df[mask].index
                df=df.drop(columns=[col]).join(flat)
            except: pass
    for col in df.columns:
        if any(isinstance(x,list) for x in df[col].dropna()):
            df[col]=df[col].apply(lambda x:str(x) if isinstance(x,list) else x)

    excl={"timestamp","src_ip","dest_ip","flow_id","in_iface","pkt_src","app_proto","proto"}
    num=[c for c in df.select_dtypes(include=np.number).columns if c not in excl]
    cat=[c for c in df.select_dtypes(include=["object","bool"]).columns if c not in excl]
    for col in num:
        if df[col].isnull().any(): df[col]=df[col].fillna(df[col].mean())
    enc=pd.get_dummies(df[cat],dummy_na=True,drop_first=True,dtype=int) if cat else pd.DataFrame(index=df.index)
    scaler=Scaler()
    scaled=pd.DataFrame(scaler.fit_transform(df[num]),columns=num,index=df.index) if num else pd.DataFrame(index=df.index)
    X=pd.concat([scaled,enc],axis=1).replace([np.inf,-np.inf],np.nan).dropna(axis=1)
    if X.empty: return None,None,"No usable features."
    model=IForest(n=100,s=min(256,len(X)),cont=cont,seed=42)
    model.fit(X.values)
    df=df.copy()
    df["anomaly_prediction"]=model.predict(X.values)
    df["anomaly_score"]=model.score(X.values)
    df["predicted_label"]=(df["anomaly_prediction"]==-1).astype(int)
    return df,X,None

# ─────────────────────────────────────────────────────────────────────────────
# Rule-based report generator  (no API key needed)
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(df: pd.DataFrame, cont: float) -> dict:
    """Produce a structured incident report purely from statistics."""
    now   = datetime.datetime.now()
    n_tot = len(df)
    n_a   = int((df["predicted_label"]==1).sum())
    n_n   = n_tot - n_a
    ratio = n_a / n_tot if n_tot else 0

    scores     = df["anomaly_score"]
    anom_scores= df.loc[df["predicted_label"]==1,"anomaly_score"]
    top10      = df.sort_values("anomaly_score").head(10)

    # ── Severity ─────────────────────────────────────────────────────────────
    if ratio >= 0.20:
        severity, sev_class = "CRITICAL", "sev-critical"
        sev_reason = f"{ratio:.1%} of traffic flagged — far above expected baseline."
    elif ratio >= 0.10:
        severity, sev_class = "HIGH", "sev-high"
        sev_reason = f"{ratio:.1%} anomaly rate exceeds the 10% high-risk threshold."
    elif ratio >= 0.05:
        severity, sev_class = "MEDIUM", "sev-medium"
        sev_reason = f"{ratio:.1%} anomaly rate is within the moderate-risk band (5–10%)."
    else:
        severity, sev_class = "LOW", "sev-low"
        sev_reason = f"{ratio:.1%} anomaly rate is below the 5% baseline threshold."

    # ── Key findings ─────────────────────────────────────────────────────────
    findings = []
    findings.append(
        f"Isolation Forest flagged {n_a:,} of {n_tot:,} records ({ratio:.2%}) as anomalous "
        f"at contamination rate {cont:.0%}."
    )

    min_score = float(anom_scores.min()) if not anom_scores.empty else 0
    findings.append(
        f"Most anomalous record scored {min_score:.4f} — "
        f"{'significantly' if min_score < -0.3 else 'moderately'} below the decision boundary."
    )

    score_spread = float(scores.std())
    findings.append(
        f"Score standard deviation: {score_spread:.4f} — "
        f"{'high variance suggesting diverse anomaly types' if score_spread > 0.15 else 'low variance suggesting a consistent traffic pattern'}."
    )

    if "event_type" in df.columns:
        anom_events = df[df["predicted_label"]==1]["event_type"].value_counts()
        if not anom_events.empty:
            top_evt = anom_events.index[0]
            top_cnt = int(anom_events.iloc[0])
            findings.append(
                f"Event type '{top_evt}' accounts for {top_cnt} anomalies "
                f"({top_cnt/n_a:.0%} of all flagged records)."
            )

    if "src_ip" in df.columns:
        top_ips = df[df["predicted_label"]==1]["src_ip"].value_counts().head(3)
        if not top_ips.empty:
            ip_list = ", ".join(top_ips.index.tolist())
            findings.append(f"Most frequent anomalous source IPs: {ip_list}.")

    if "dest_ip" in df.columns:
        top_dst = df[df["predicted_label"]==1]["dest_ip"].value_counts().head(3)
        if not top_dst.empty:
            dst_list = ", ".join(top_dst.index.tolist())
            findings.append(f"Most targeted destination IPs: {dst_list}.")

    # ── Pattern analysis ─────────────────────────────────────────────────────
    patterns = []
    q25 = float(anom_scores.quantile(0.25)) if not anom_scores.empty else 0
    q75 = float(anom_scores.quantile(0.75)) if not anom_scores.empty else 0
    iqr = q75 - q25
    if iqr < 0.05:
        patterns.append("Anomaly scores are tightly clustered, suggesting a single dominant anomaly pattern.")
    elif iqr > 0.20:
        patterns.append("Wide IQR in anomaly scores indicates multiple distinct anomaly clusters in the data.")
    else:
        patterns.append("Moderate score spread suggests 2–3 distinguishable anomaly groups.")

    normal_mean = float(df.loc[df["predicted_label"]==0,"anomaly_score"].mean()) if n_n > 0 else 0
    anom_mean   = float(anom_scores.mean()) if not anom_scores.empty else 0
    separation  = normal_mean - anom_mean
    patterns.append(
        f"Score separation between normal ({normal_mean:.4f}) and anomalous ({anom_mean:.4f}) "
        f"records is {separation:.4f} — "
        f"{'strong' if separation > 0.3 else 'moderate' if separation > 0.15 else 'weak'} discriminability."
    )

    if "proto" in df.columns:
        anom_proto = df[df["predicted_label"]==1]["proto"].value_counts()
        if not anom_proto.empty:
            patterns.append(
                f"Protocol distribution in anomalies: "
                + ", ".join(f"{p} ({c})" for p,c in anom_proto.head(3).items()) + "."
            )

    # ── Executive summary ─────────────────────────────────────────────────────
    summary = (
        f"Anomaly detection completed on {now.strftime('%d %B %Y at %H:%M:%S')}. "
        f"The Isolation Forest model processed {n_tot:,} network log records and identified "
        f"{n_a:,} anomalies ({ratio:.2%}), resulting in a {severity} severity assessment. "
        f"The analysis used a contamination rate of {cont:.0%}, meaning the model was tuned "
        f"to expect approximately {cont:.0%} outliers in the dataset."
    )

    # ── Top anomalies table ────────────────────────────────────────────────────
    tcols = [c for c in ["timestamp","event_type","src_ip","dest_ip","proto","anomaly_score"] if c in top10.columns]
    top10_data = top10[tcols].copy()
    if "anomaly_score" in top10_data.columns:
        top10_data["anomaly_score"] = top10_data["anomaly_score"].round(4)

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "severity": severity, "sev_class": sev_class, "sev_reason": sev_reason,
        "summary": summary,
        "findings": findings,
        "patterns": patterns,
        "top10": top10_data,
        "n_tot": n_tot, "n_a": n_a, "n_n": n_n, "ratio": ratio,
        "score_min": float(scores.min()), "score_max": float(scores.max()),
        "score_mean": float(scores.mean()), "score_std": score_spread,
    }


def report_to_text(r: dict) -> str:
    lines = [
        "=" * 60,
        "  ANOMALY DETECTION INCIDENT REPORT",
        "=" * 60,
        f"Generated   : {r['generated_at']}",
        f"Total       : {r['n_tot']:,}  |  Anomalies: {r['n_a']:,}  |  Normal: {r['n_n']:,}",
        f"Ratio       : {r['ratio']:.2%}",
        f"Severity    : {r['severity']}",
        "",
        "── EXECUTIVE SUMMARY " + "─"*39,
        r["summary"],
        "",
        "── SEVERITY JUSTIFICATION " + "─"*34,
        r["sev_reason"],
        "",
        "── KEY FINDINGS " + "─"*44,
    ] + [f"  • {f}" for f in r["findings"]] + [
        "",
        "── ANOMALY PATTERN ANALYSIS " + "─"*32,
    ] + [f"  • {p}" for p in r["patterns"]] + [
        "",
        "── SCORE STATISTICS " + "─"*40,
        f"  Min  : {r['score_min']:.4f}",
        f"  Max  : {r['score_max']:.4f}",
        f"  Mean : {r['score_mean']:.4f}",
        f"  Std  : {r['score_std']:.4f}",
        "",
        "── TOP 10 ANOMALOUS RECORDS " + "─"*32,
        r["top10"].to_string(index=False),
        "",
        "=" * 60,
        "  END OF REPORT — FoxyDucky Task 3 Noctra Lupra",
        "=" * 60,
    ]
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
def fig_score_dist(df):
    _mpl(); fig,ax=plt.subplots(figsize=(8,3.8),facecolor="#0b0f1a"); ax.set_facecolor(CARD_BG)
    ax.hist(df.loc[df["predicted_label"]==0,"anomaly_score"],bins=40,color=BLUE,alpha=.75,label="Normal",edgecolor="none")
    ax.hist(df.loc[df["predicted_label"]==1,"anomaly_score"],bins=40,color=CRIMSON,alpha=.75,label="Anomaly",edgecolor="none")
    if (df["predicted_label"]==1).any():
        ax.axvline(df.loc[df["predicted_label"]==1,"anomaly_score"].max(),color=AMBER,linestyle="--",lw=1.5,label="Threshold")
    ax.set_title("Score distribution"); ax.set_xlabel("Anomaly score"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True)
    for s in ax.spines.values(): s.set_edgecolor(BORDER)
    fig.tight_layout(); return fig

def fig_top15(df):
    _mpl(); top=df.sort_values("anomaly_score").head(15)
    colors=[CRIMSON if v==1 else BLUE for v in top["predicted_label"]]
    fig,ax=plt.subplots(figsize=(8,3.8),facecolor="#0b0f1a"); ax.set_facecolor(CARD_BG)
    ax.barh([str(i) for i in range(1,len(top)+1)],top["anomaly_score"],color=colors,edgecolor="none",height=.65)
    ax.axvline(0,color=AMBER,linestyle="--",lw=1.2,alpha=.7)
    ax.set_title("Top 15 lowest scores"); ax.set_xlabel("Score"); ax.set_ylabel("Rank")
    ax.invert_yaxis()
    ax.legend(handles=[mpatches.Patch(color=CRIMSON,label="Anomaly"),mpatches.Patch(color=BLUE,label="Normal")])
    ax.grid(True,axis="x")
    for s in ax.spines.values(): s.set_edgecolor(BORDER)
    fig.tight_layout(); return fig

def fig_pca(df,X):
    if X.shape[1]<3: return None
    try:
        _mpl(); comps=pca2(X.values)
        pdf=pd.DataFrame(comps,columns=["PC1","PC2"]); pdf["lbl"]=df["predicted_label"].values
        fig,ax=plt.subplots(figsize=(8,3.8),facecolor="#0b0f1a"); ax.set_facecolor(CARD_BG)
        n=pdf[pdf["lbl"]==0]; a=pdf[pdf["lbl"]==1]
        ax.scatter(n["PC1"],n["PC2"],s=14,color=BLUE,alpha=.5,label="Normal",linewidths=0)
        ax.scatter(a["PC1"],a["PC2"],s=45,color=CRIMSON,alpha=.85,label="Anomaly",marker="X",linewidths=0)
        ax.set_title("PCA scatter"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(); ax.grid(True,alpha=.4)
        for s in ax.spines.values(): s.set_edgecolor(BORDER)
        fig.tight_layout(); return fig
    except: return None

def fig_event(df):
    if "event_type" not in df.columns: return None
    _mpl(); fig,ax=plt.subplots(figsize=(8,3.8),facecolor="#0b0f1a"); ax.set_facecolor(CARD_BG)
    sns.countplot(x="event_type",hue="anomaly_prediction",data=df,
                  palette={1:BLUE,-1:CRIMSON},ax=ax,edgecolor="none")
    ax.set_title("Event type breakdown"); ax.set_xlabel("Event type"); ax.set_ylabel("Count")
    ax.legend(handles=[mpatches.Patch(color=BLUE,label="Normal"),mpatches.Patch(color=CRIMSON,label="Anomaly")])
    ax.grid(True,axis="y"); plt.xticks(rotation=30,ha="right")
    for s in ax.spines.values(): s.set_edgecolor(BORDER)
    fig.tight_layout(); return fig

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Anomaly Detection")
    st.markdown("<p style='color:#6b7a99;font-size:.82rem;margin-top:-6px;'>Isolation Forest · Pure NumPy</p>",
                unsafe_allow_html=True)
    st.markdown("---")
    uploaded  = st.file_uploader("Upload JSONL / JSON file", type=["json","jsonl"])
    st.markdown("**Contamination rate**")
    cont_rate = st.slider("", min_value=0.01, max_value=0.50, value=0.05, step=0.01)
    st.caption(f"**{cont_rate:.0%}** of records flagged as anomalies")
    st.markdown("---")
    run_btn   = st.button("▶  Run Detection", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("<p style='color:#3a4a6a;font-size:.72rem;'>FoxyDucky · Task 3 Noctra Lupra</p>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Welcome screen
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is None:
    _,mid,_ = st.columns([1,2,1])
    with mid:
        st.markdown("""
        <div style='text-align:center;padding:4rem 2rem;background:#111827;
                    border:1px solid #1e2740;border-radius:16px;margin-top:3rem;'>
            <div style='font-size:3.5rem;margin-bottom:1rem;'>🔍</div>
            <h3 style='color:#e2e8f0;margin-bottom:.5rem;'>Upload a file to begin</h3>
            <p style='color:#6b7a99;font-size:.9rem;line-height:1.7;'>
                Select a JSONL file from the sidebar,<br>
                adjust the contamination rate,<br>
                then click <strong style='color:#3b82f6;'>▶ Run Detection</strong>.
            </p>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Run analysis on button click
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running Isolation Forest pipeline…"):
        df, X, err = run_analysis(uploaded.getvalue(), cont_rate)
    if err:
        st.error(f"**Error:** {err}"); st.stop()
    st.session_state.update({"df":df,"X":X,"cont":cont_rate,"report":None})

if "df" not in st.session_state:
    st.info("Upload a file and click **▶ Run Detection** to start.")
    st.stop()

df   = st.session_state["df"]
X    = st.session_state["X"]
cont = st.session_state["cont"]
n_a  = int((df["predicted_label"]==1).sum())
n_n  = int((df["predicted_label"]==0).sum())
tot  = len(df)
rat  = n_a/tot if tot else 0

# ─────────────────────────────────────────────────────────────────────────────
# Banner + metrics
# ─────────────────────────────────────────────────────────────────────────────
bg_c = "#2a0f0f" if rat>0.1 else "#0f2e1e"
tx_c = "#ef4444" if rat>0.1 else "#22c55e"
st.markdown(f"""
<div style='background:#111827;border:1px solid #1e2740;border-radius:14px;
            padding:1rem 1.5rem;margin-bottom:1.2rem;display:flex;align-items:center;gap:14px;'>
    <span style='font-size:1.5rem;'>🚨</span>
    <div>
        <p style='margin:0;font-size:1.05rem;font-weight:700;color:#e2e8f0;'>Analysis complete</p>
        <p style='margin:0;font-size:.82rem;color:#6b7a99;'>{tot:,} records · contamination = {cont:.0%}</p>
    </div>
    <div style='margin-left:auto;background:{bg_c};border-radius:8px;
                padding:.3rem .9rem;color:{tx_c};font-weight:700;font-size:.88rem;'>
        {rat:.1%} anomaly rate
    </div>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total records", f"{tot:,}")
c2.metric("Anomalies 🔴",  f"{n_a:,}",  delta=f"{rat:.1%} of total",   delta_color="inverse")
c3.metric("Normal 🟢",     f"{n_n:,}",  delta=f"{1-rat:.1%} of total")
c4.metric("Anomaly ratio", f"{rat:.2%}")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Charts 2×2
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='s-card'><div class='s-card-title'>📊 Visualisations</div>",
            unsafe_allow_html=True)
r1l, r1r = st.columns(2)
with r1l:
    f=fig_score_dist(df); st.pyplot(f); plt.close(f)
with r1r:
    f=fig_top15(df); st.pyplot(f); plt.close(f)
r2l, r2r = st.columns(2)
with r2l:
    f=fig_pca(df,X)
    if f: st.pyplot(f); plt.close(f)
    else: st.info("Too few features for PCA.")
with r2r:
    f=fig_event(df)
    if f: st.pyplot(f); plt.close(f)
    else: st.info("No `event_type` column for this chart.")
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Top 20 table
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='s-card'><div class='s-card-title'>📋 Top 20 anomalous records</div>",
            unsafe_allow_html=True)
dcols=[c for c in ["timestamp","event_type","src_ip","dest_ip","proto","anomaly_score","predicted_label"] if c in df.columns]
tbl=df.sort_values("anomaly_score").head(20)[dcols].copy()
if "predicted_label" in tbl.columns:
    tbl["predicted_label"]=tbl["predicted_label"].map({1:"🔴 Anomaly",0:"🟢 Normal"})
if "anomaly_score" in tbl.columns:
    tbl["anomaly_score"]=tbl["anomaly_score"].round(4)
st.dataframe(tbl,use_container_width=True,hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Incident Report (rule-based, no API key)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='s-card'><div class='s-card-title'>📄 Incident Report</div>",
            unsafe_allow_html=True)

btn_col, info_col = st.columns([1,4])
with btn_col:
    gen_btn = st.button("📄 Generate Report", key="gen_report", use_container_width=True)
with info_col:
    st.caption("Automatically generates a structured SOC incident report from detection results — no API key required.")

if gen_btn:
    with st.spinner("Building report…"):
        rpt = generate_report(df, cont)
    st.session_state["report"] = rpt

rpt = st.session_state.get("report")

if rpt:
    # header row
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Report generated", rpt["generated_at"].split(" ")[0])
    hc2.metric("Records analysed", f"{rpt['n_tot']:,}")
    hc3.metric("Anomalies found",  f"{rpt['n_a']:,}")
    hc4.metric("Score std dev",    f"{rpt['score_std']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # severity
    st.markdown(f"""
    <div class='rpt-block'>
        <div class='rpt-label'>Severity assessment</div>
        <span class='rpt-badge {rpt["sev_class"]}'>{rpt["severity"]}</span>
        <p class='rpt-value' style='margin-top:.5rem;'>{rpt["sev_reason"]}</p>
    </div>""", unsafe_allow_html=True)

    # summary
    st.markdown(f"""
    <div class='rpt-block'>
        <div class='rpt-label'>Executive summary</div>
        <p class='rpt-value'>{rpt["summary"]}</p>
    </div>""", unsafe_allow_html=True)

    # findings + patterns side by side
    fc, pc = st.columns(2)
    with fc:
        items = "".join(
            f"<div class='find-item'><div class='find-icon'></div><div class='find-text'>{f}</div></div>"
            for f in rpt["findings"]
        )
        st.markdown(f"""
        <div class='rpt-block' style='height:100%;'>
            <div class='rpt-label'>Key findings</div>{items}
        </div>""", unsafe_allow_html=True)
    with pc:
        items2 = "".join(
            f"<div class='find-item'><div class='find-icon' style='background:#3b82f6;'></div><div class='find-text'>{p}</div></div>"
            for p in rpt["patterns"]
        )
        st.markdown(f"""
        <div class='rpt-block' style='height:100%;'>
            <div class='rpt-label'>Pattern analysis</div>{items2}
        </div>""", unsafe_allow_html=True)

    # score stats
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Min score",  f"{rpt['score_min']:.4f}")
    s2.metric("Max score",  f"{rpt['score_max']:.4f}")
    s3.metric("Mean score", f"{rpt['score_mean']:.4f}")
    s4.metric("Std dev",    f"{rpt['score_std']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # top 10 table inside report
    st.markdown("<div class='rpt-block'><div class='rpt-label'>Top 10 most anomalous records</div>",
                unsafe_allow_html=True)
    st.dataframe(rpt["top10"], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # download
    st.download_button(
        "⬇️ Download report (.txt)",
        data=report_to_text(rpt),
        file_name=f"incident_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        key="dl_report"
    )

st.markdown("</div>", unsafe_allow_html=True)