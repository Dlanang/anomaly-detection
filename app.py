"""
Streamlit Anomaly Detection Dashboard
Pure NumPy · PDF report with ReportLab (text + diagrams + table only)
Run: pip install streamlit pandas numpy matplotlib seaborn reportlab
     python -m streamlit run app.py
"""

import json
import warnings
import datetime
import io

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
.s-card { background:#111827; border:1px solid #1e2740; border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:1.2rem; }
.s-card-title { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:#6b7a99; margin-bottom:.9rem; font-weight:600; }
.rpt-wrap  { background:#1a1f2e; border:1px solid #1e2740; border-radius:12px; padding:2rem; }
.rpt-hdr   { background:linear-gradient(135deg,#1e3a5f,#2563eb); border-radius:10px; padding:1.4rem 1.8rem; margin-bottom:1.4rem; }
.rpt-title { font-size:1.5rem; font-weight:800; color:#fff; margin:0 0 .2rem; }
.rpt-sub   { font-size:.85rem; color:#93c5fd; margin:0; }
.rpt-meta  { display:flex; gap:2rem; margin-top:.9rem; flex-wrap:wrap; }
.rpt-mi    { font-size:.77rem; color:#bfdbfe; }
.rpt-mi span { color:#fff; font-weight:600; }
.sev-badge { display:inline-block; padding:.3rem 1rem; border-radius:6px; font-size:.88rem; font-weight:700; float:right; margin-top:.2rem; }
.sev-CRITICAL { background:#3d0f0f; color:#ef4444; border:1px solid #ef4444; }
.sev-HIGH     { background:#3d2a0a; color:#f59e0b; border:1px solid #f59e0b; }
.sev-MEDIUM   { background:#0f2340; color:#3b82f6; border:1px solid #3b82f6; }
.sev-LOW      { background:#0f2e1e; color:#22c55e; border:1px solid #22c55e; }
.stat-row  { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:1.4rem; }
.stat-box  { background:#111827; border:1px solid #1e2740; border-radius:10px; padding:.9rem 1rem; }
.stat-top  { height:3px; border-radius:2px; margin-bottom:.7rem; }
.stat-lbl  { font-size:.67rem; text-transform:uppercase; letter-spacing:.07em; color:#6b7a99; margin-bottom:.25rem; }
.stat-val  { font-size:1.5rem; font-weight:700; color:#e2e8f0; }
.stat-sub  { font-size:.7rem; color:#475569; margin-top:.15rem; }
.sec-hdr   { display:flex; align-items:center; gap:.6rem; margin:1.2rem 0 .6rem; padding-bottom:.35rem; border-bottom:2px solid #1e3a5f; }
.sec-num   { background:#2563eb; color:#fff; border-radius:50%; width:20px; height:20px; display:flex; align-items:center; justify-content:center; font-size:.68rem; font-weight:700; flex-shrink:0; }
.sec-title { font-size:.95rem; font-weight:700; color:#e2e8f0; text-transform:uppercase; letter-spacing:.05em; }
.rpt-sum   { background:#0f1520; border-left:4px solid #2563eb; border-radius:4px; padding:.9rem 1.1rem; color:#cbd5e1; font-size:.88rem; line-height:1.75; }
.sev-box   { background:#0f1520; border:1px solid #1e3a5f; border-radius:8px; padding:.75rem 1rem; margin-top:.7rem; display:flex; gap:.7rem; }
.sev-text  { color:#94a3b8; font-size:.86rem; line-height:1.6; }
.find-list { list-style:none; padding:0; margin:0; }
.find-list li { display:flex; gap:.7rem; align-items:flex-start; padding:.5rem 0; border-bottom:1px solid #1e2740; }
.find-list li:last-child { border-bottom:none; }
.find-num  { background:#2563eb; color:#fff; border-radius:50%; width:19px; height:19px; min-width:19px; display:flex; align-items:center; justify-content:center; font-size:.65rem; font-weight:700; margin-top:.15rem; }
.find-text { color:#94a3b8; font-size:.86rem; line-height:1.65; }
.pat-list  { list-style:none; padding:0; margin:0; }
.pat-list li { display:flex; gap:.7rem; align-items:flex-start; padding:.5rem 0; border-bottom:1px solid #1e2740; }
.pat-list li:last-child { border-bottom:none; }
.pat-dot   { width:7px; height:7px; min-width:7px; background:#f59e0b; border-radius:2px; margin-top:.45rem; }
.pat-text  { color:#94a3b8; font-size:.86rem; line-height:1.65; }
.sc-grid   { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
.sc-box    { background:#0f1520; border:1px solid #1e2740; border-radius:8px; padding:.7rem; text-align:center; }
.sc-lbl    { font-size:.65rem; color:#6b7a99; text-transform:uppercase; letter-spacing:.05em; margin-bottom:.25rem; }
.sc-val    { font-size:1.1rem; font-weight:700; }
.rpt-footer { background:#0f1520; border-top:1px solid #1e2740; border-radius:0 0 10px 10px; padding:.65rem 1rem; display:flex; justify-content:space-between; margin-top:1.2rem; }
.rpt-footer p { font-size:.7rem; color:#3a4a6a; margin:0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chart palette
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG="#0b0f1a"; CARD_BG="#111827"; BORDER="#1e2740"
TEXT_PRI="#e2e8f0"; TEXT_MUT="#6b7a99"
BLUE="#3b82f6"; CRIMSON="#ef4444"; AMBER="#f59e0b"; GREEN="#22c55e"

def _mpl_dark():
    plt.rcParams.update({
        "figure.facecolor":DARK_BG,"axes.facecolor":CARD_BG,
        "axes.edgecolor":BORDER,"axes.labelcolor":TEXT_MUT,
        "axes.titlecolor":TEXT_PRI,"xtick.color":TEXT_MUT,"ytick.color":TEXT_MUT,
        "text.color":TEXT_PRI,"grid.color":BORDER,"grid.linestyle":"--","grid.alpha":.5,
        "axes.titlesize":12,"axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,
        "legend.facecolor":CARD_BG,"legend.edgecolor":BORDER,"legend.fontsize":9,
    })

def _mpl_light():
    """Light theme for PDF charts."""
    plt.rcParams.update({
        "figure.facecolor":"#ffffff","axes.facecolor":"#f8fafc",
        "axes.edgecolor":"#cbd5e1","axes.labelcolor":"#475569",
        "axes.titlecolor":"#1e293b","xtick.color":"#64748b","ytick.color":"#64748b",
        "text.color":"#1e293b","grid.color":"#e2e8f0","grid.linestyle":"--","grid.alpha":.6,
        "axes.titlesize":11,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,
        "legend.facecolor":"#ffffff","legend.edgecolor":"#cbd5e1","legend.fontsize":8,
    })

PDF_BLUE   = "#2563eb"; PDF_RED  = "#dc2626"
PDF_GREEN  = "#16a34a"; PDF_AMB  = "#d97706"

# ─────────────────────────────────────────────────────────────────────────────
# Pure-NumPy ML
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
# Report data builder
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(df, X, cont):
    now=datetime.datetime.now()
    n_tot=len(df); n_a=int((df["predicted_label"]==1).sum()); n_n=n_tot-n_a
    ratio=n_a/n_tot if n_tot else 0
    scores=df["anomaly_score"]; asc=df.loc[df["predicted_label"]==1,"anomaly_score"]

    if   ratio>=0.20: sev,sev_col="CRITICAL","#ef4444"
    elif ratio>=0.10: sev,sev_col="HIGH","#f59e0b"
    elif ratio>=0.05: sev,sev_col="MEDIUM","#3b82f6"
    else:             sev,sev_col="LOW","#22c55e"

    if   ratio>=0.20: sev_r=f"{ratio:.1%} of traffic flagged — far above expected baseline."
    elif ratio>=0.10: sev_r=f"{ratio:.1%} anomaly rate exceeds the 10% high-risk threshold."
    elif ratio>=0.05: sev_r=f"{ratio:.1%} anomaly rate is within the moderate-risk band (5–10%)."
    else:             sev_r=f"{ratio:.1%} anomaly rate is below the 5% baseline threshold."

    findings=[]
    findings.append(f"Isolation Forest flagged {n_a:,} of {n_tot:,} records ({ratio:.2%}) as anomalous at contamination rate {cont:.0%}.")
    ms=float(asc.min()) if not asc.empty else 0
    findings.append(f"Most anomalous record scored {ms:.4f} — {'significantly' if ms<-0.3 else 'moderately'} below the decision boundary.")
    ss=float(scores.std())
    findings.append(f"Score std dev: {ss:.4f} — {'high variance, diverse anomaly types' if ss>0.15 else 'low variance, consistent traffic pattern'}.")
    if "event_type" in df.columns:
        ae=df[df["predicted_label"]==1]["event_type"].value_counts()
        if not ae.empty:
            findings.append(f"Event type '{ae.index[0]}' accounts for {int(ae.iloc[0])} anomalies ({int(ae.iloc[0])/n_a:.0%} of all flagged).")
    if "src_ip" in df.columns:
        ti=df[df["predicted_label"]==1]["src_ip"].value_counts().head(3)
        if not ti.empty: findings.append(f"Top anomalous source IPs: {', '.join(ti.index.tolist())}.")
    if "dest_ip" in df.columns:
        td=df[df["predicted_label"]==1]["dest_ip"].value_counts().head(3)
        if not td.empty: findings.append(f"Top targeted destination IPs: {', '.join(td.index.tolist())}.")

    patterns=[]
    if not asc.empty:
        iqr=float(asc.quantile(.75))-float(asc.quantile(.25))
        if   iqr<0.05:  patterns.append("Scores tightly clustered — single dominant anomaly pattern.")
        elif iqr>0.20:  patterns.append("Wide IQR — multiple distinct anomaly clusters detected.")
        else:           patterns.append("Moderate spread suggests 2–3 distinguishable anomaly groups.")
    nm=float(df.loc[df["predicted_label"]==0,"anomaly_score"].mean()) if n_n>0 else 0
    am=float(asc.mean()) if not asc.empty else 0
    sep=nm-am
    patterns.append(f"Score separation: normal ({nm:.4f}) vs anomalous ({am:.4f}) = {sep:.4f} — {'strong' if sep>0.3 else 'moderate' if sep>0.15 else 'weak'} discriminability.")
    if "proto" in df.columns:
        ap=df[df["predicted_label"]==1]["proto"].value_counts()
        if not ap.empty:
            patterns.append("Protocol mix: "+", ".join(f"{p} ({c})" for p,c in ap.head(3).items())+".")

    summary=(
        f"Anomaly detection completed on {now.strftime('%d %B %Y at %H:%M:%S')}. "
        f"The Isolation Forest model processed {n_tot:,} network log records and identified "
        f"{n_a:,} anomalies ({ratio:.2%}), resulting in a {sev} severity assessment. "
        f"The model was configured with a contamination rate of {cont:.0%}."
    )
    tcols=[c for c in ["timestamp","event_type","src_ip","dest_ip","proto","anomaly_score"] if c in df.columns]
    top10=df.sort_values("anomaly_score").head(10)[tcols].copy()
    if "anomaly_score" in top10.columns: top10["anomaly_score"]=top10["anomaly_score"].round(4)

    return {
        "now":now,"sev":sev,"sev_col":sev_col,"sev_r":sev_r,
        "summary":summary,"findings":findings,"patterns":patterns,
        "top10":top10,"top10_cols":tcols,
        "n_tot":n_tot,"n_a":n_a,"n_n":n_n,"ratio":ratio,
        "smin":float(scores.min()),"smax":float(scores.max()),
        "smean":float(scores.mean()),"sstd":ss,
        "cont":cont,"df":df,"X":X,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Light-theme chart PNGs for PDF
# ─────────────────────────────────────────────────────────────────────────────
def _png(fig) -> bytes:
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0); return buf.read()

def make_pdf_charts(df, X) -> dict:
    C={}
    # score distribution
    _mpl_light()
    f,a=plt.subplots(figsize=(5.5,2.8),facecolor="#ffffff"); a.set_facecolor("#f8fafc")
    a.hist(df.loc[df["predicted_label"]==0,"anomaly_score"],bins=35,color=PDF_BLUE,alpha=.7,label="Normal",edgecolor="none")
    a.hist(df.loc[df["predicted_label"]==1,"anomaly_score"],bins=35,color=PDF_RED, alpha=.7,label="Anomaly",edgecolor="none")
    if (df["predicted_label"]==1).any():
        a.axvline(df.loc[df["predicted_label"]==1,"anomaly_score"].max(),color=PDF_AMB,ls="--",lw=1.2,label="Threshold")
    a.set_title("Figure 1 — Anomaly Score Distribution"); a.set_xlabel("Score"); a.set_ylabel("Count")
    a.legend(); a.grid(True)
    for s in a.spines.values(): s.set_edgecolor("#cbd5e1"); s.set_linewidth(0.6)
    f.tight_layout(); C["score"]=_png(f)

    # top 15 bar
    _mpl_light(); top=df.sort_values("anomaly_score").head(15)
    colors=[PDF_RED if v==1 else PDF_BLUE for v in top["predicted_label"]]
    f,a=plt.subplots(figsize=(5.5,2.8),facecolor="#ffffff"); a.set_facecolor("#f8fafc")
    a.barh([str(i) for i in range(1,len(top)+1)],top["anomaly_score"],color=colors,edgecolor="none",height=.65)
    a.axvline(0,color=PDF_AMB,ls="--",lw=1,alpha=.7); a.set_title("Figure 2 — Top 15 Lowest Scores")
    a.set_xlabel("Score"); a.invert_yaxis()
    a.legend(handles=[mpatches.Patch(color=PDF_RED,label="Anomaly"),mpatches.Patch(color=PDF_BLUE,label="Normal")])
    a.grid(True,axis="x")
    for s in a.spines.values(): s.set_edgecolor("#cbd5e1"); s.set_linewidth(0.6)
    f.tight_layout(); C["top15"]=_png(f)

    # pca
    if X.shape[1]>=3:
        try:
            _mpl_light(); comps=pca2(X.values)
            pdf2=pd.DataFrame(comps,columns=["PC1","PC2"]); pdf2["lbl"]=df["predicted_label"].values
            f,a=plt.subplots(figsize=(5.5,2.8),facecolor="#ffffff"); a.set_facecolor("#f8fafc")
            nn=pdf2[pdf2["lbl"]==0]; aa=pdf2[pdf2["lbl"]==1]
            a.scatter(nn["PC1"],nn["PC2"],s=12,color=PDF_BLUE,alpha=.45,label="Normal",linewidths=0)
            a.scatter(aa["PC1"],aa["PC2"],s=40,color=PDF_RED, alpha=.85,label="Anomaly",marker="X",linewidths=0)
            a.set_title("Figure 3 — PCA Projection"); a.set_xlabel("PC1"); a.set_ylabel("PC2")
            a.legend(); a.grid(True,alpha=.4)
            for s in a.spines.values(): s.set_edgecolor("#cbd5e1"); s.set_linewidth(0.6)
            f.tight_layout(); C["pca"]=_png(f)
        except: pass

    # event type
    if "event_type" in df.columns:
        _mpl_light(); f,a=plt.subplots(figsize=(5.5,2.8),facecolor="#ffffff"); a.set_facecolor("#f8fafc")
        sns.countplot(x="event_type",hue="anomaly_prediction",data=df,
                      palette={1:PDF_BLUE,-1:PDF_RED},ax=a,edgecolor="none")
        a.set_title("Figure 4 — Event Type Breakdown"); a.set_xlabel("Event Type"); a.set_ylabel("Count")
        a.legend(handles=[mpatches.Patch(color=PDF_BLUE,label="Normal"),mpatches.Patch(color=PDF_RED,label="Anomaly")])
        a.grid(True,axis="y"); plt.xticks(rotation=30,ha="right")
        for s in a.spines.values(): s.set_edgecolor("#cbd5e1"); s.set_linewidth(0.6)
        f.tight_layout(); C["event"]=_png(f)

    return C

# ─────────────────────────────────────────────────────────────────────────────
# PDF builder — clean A4, text + diagrams + table only
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf(r: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, KeepTogether, PageBreak,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm,
        title="Anomaly Detection Incident Report",
        author="FoxyDucky Task 3 Noctra Lupra",
    )
    W = A4[0] - 40*mm

    # colours
    C   = colors.HexColor
    NAVY  = C("#1e3a5f"); BLUE2 = C("#2563eb"); LBLUE = C("#dbeafe")
    DARK  = C("#0f172a"); MID   = C("#1e293b"); LIGHT = C("#f8fafc")
    SLATE = C("#475569"); MUTED = C("#94a3b8"); WHITE = colors.white
    RED2  = C("#dc2626"); AMB2  = C("#d97706"); GRN2  = C("#16a34a"); BLU2 = C("#2563eb")
    BORD  = C("#e2e8f0"); HBORD = C("#cbd5e1")

    SEV_C = {"CRITICAL":C("#dc2626"),"HIGH":C("#d97706"),"MEDIUM":C("#2563eb"),"LOW":C("#16a34a")}
    sev_color = SEV_C[r["sev"]]

    def P(text, sz=9, color=MID, bold=False, align=TA_LEFT, lead=None, space=2):
        return Paragraph(str(text), ParagraphStyle("x",
            fontSize=sz, textColor=color,
            fontName="Helvetica-Bold" if bold else "Helvetica",
            alignment=align, leading=lead or sz*1.45,
            spaceAfter=space, spaceBefore=0))

    def HR(color=BORD, thick=0.5, sb=4, sa=4):
        return HRFlowable(width="100%", thickness=thick, color=color, spaceBefore=sb, spaceAfter=sa)

    def img(b, w, h):
        return RLImage(io.BytesIO(b), width=w, height=h)

    def section_hdr(num, title):
        num_cell = Table([[P(str(num), 7, WHITE, bold=True, align=TA_CENTER)]],
            colWidths=[15], rowHeights=[15],
            style=TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),BLUE2),
                ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
                ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                ("ROUNDEDCORNERS",[7,7,7,7]),
            ]))
        row = Table([[num_cell, P(title, 10, NAVY, bold=True)]],
            colWidths=[20, W-20],
            style=TableStyle([
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),3),
                ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
            ]))
        return [row, HR(BLUE2, 1)]

    story = []

    # ── COVER HEADER ─────────────────────────────────────────────────────────
    hdr = Table([[
        P("ANOMALY DETECTION SYSTEM", 9, WHITE, bold=True),
        P("FOXYDUCKY · TASK 3 NOCTRA LUPRA", 8, C("#93c5fd"), align=TA_RIGHT),
    ]], colWidths=[W*0.55, W*0.45],
    style=TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),NAVY),
        ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ROUNDEDCORNERS",[5,5,5,5]),
    ]))
    story.append(hdr)
    story.append(Spacer(1,8))

    # ── TITLE + SEVERITY ─────────────────────────────────────────────────────
    sev_badge = Table([[P(r["sev"], 10, WHITE, bold=True, align=TA_CENTER)]],
        colWidths=[55], rowHeights=[20],
        style=TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),sev_color),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
            ("ROUNDEDCORNERS",[4,4,4,4]),
        ]))
    title_row = Table([[P("INCIDENT REPORT", 20, NAVY, bold=True), sev_badge]],
        colWidths=[W-65, 65],
        style=TableStyle([
            ("VALIGN",(0,0),(-1,-1),"BOTTOM"),
            ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),2),
            ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ]))
    story.append(title_row)
    story.append(P("Network Anomaly Detection — Isolation Forest Analysis", 10, SLATE))
    story.append(Spacer(1,5))

    # meta strip
    meta = Table([[
        P(f"<b>Date:</b>  {r['now'].strftime('%d %B %Y  %H:%M:%S')}", 8, MUTED),
        P(f"<b>Contamination rate:</b>  {r['cont']:.0%}", 8, MUTED),
        P(f"<b>Algorithm:</b>  Isolation Forest (Pure NumPy)", 8, MUTED),
    ]], colWidths=[W/3]*3,
    style=TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),LIGHT),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("ROUNDEDCORNERS",[4,4,4,4]),
    ]))
    story.append(meta)
    story.append(Spacer(1,10))
    story.append(HR(BLUE2, 1.5, 2, 6))

    # ── SUMMARY STATS (4 boxes) ───────────────────────────────────────────────
    sw = W/4 - 3
    stats = [
        ("TOTAL RECORDS", f"{r['n_tot']:,}", "Records processed", BLU2),
        ("ANOMALIES",     f"{r['n_a']:,}",   f"{r['ratio']:.2%} of total", RED2),
        ("NORMAL",        f"{r['n_n']:,}",   f"{1-r['ratio']:.2%} of total", GRN2),
        ("ANOMALY RATIO", f"{r['ratio']:.2%}", f"{r['sev']} severity", sev_color),
    ]
    stat_cells = []
    for lbl,val,sub,col in stats:
        bar = Table([[""]], colWidths=[sw], rowHeights=[3],
            style=TableStyle([("BACKGROUND",(0,0),(-1,-1),col)]))
        stat_cells.append([bar, Spacer(1,4), P(lbl,6.5,MUTED,bold=True), P(val,16,MID,bold=True), P(sub,7,SLATE)])

    st_tbl = Table([stat_cells], colWidths=[sw]*4,
        style=TableStyle([
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("BACKGROUND",(0,0),(-1,-1),LIGHT),
            ("BOX",(0,0),(0,0),0.4,HBORD),("BOX",(1,0),(1,0),0.4,HBORD),
            ("BOX",(2,0),(2,0),0.4,HBORD),("BOX",(3,0),(3,0),0.4,HBORD),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),8),
            ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
            ("COLPADDING",(0,0),(-1,-1),3),
        ]))
    story.append(st_tbl)
    story.append(Spacer(1,12))

    # ── 01 EXECUTIVE SUMMARY ─────────────────────────────────────────────────
    for e in section_hdr("01","EXECUTIVE SUMMARY"): story.append(e)
    story.append(Spacer(1,4))
    story.append(Table([[P(r["summary"], 9, MID, lead=15, align=TA_JUSTIFY)]],
        colWidths=[W],
        style=TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),C("#eff6ff")),
            ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
            ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
            ("LINEBEFORE",(0,0),(0,-1),4,BLUE2),
            ("ROUNDEDCORNERS",[3,3,3,3]),
        ])))
    story.append(Spacer(1,5))
    story.append(Table([[
        P(f"<b>Severity assessment ({r['sev']}):</b>  {r['sev_r']}", 9, SLATE, lead=14),
    ]], colWidths=[W],
    style=TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),LIGHT),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
        ("LINEBEFORE",(0,0),(0,-1),4,sev_color),
        ("ROUNDEDCORNERS",[3,3,3,3]),
    ])))
    story.append(Spacer(1,12))

    # ── 02 KEY FINDINGS ──────────────────────────────────────────────────────
    for e in section_hdr("02","KEY FINDINGS"): story.append(e)
    story.append(Spacer(1,4))
    for i,f in enumerate(r["findings"]):
        nb = Table([[P(str(i+1),7,WHITE,bold=True,align=TA_CENTER)]],
            colWidths=[15], rowHeights=[15],
            style=TableStyle([
                ("BACKGROUND",(0,0),(-1,-1),BLUE2),
                ("TOPPADDING",(0,0),(-1,-1),1),("BOTTOMPADDING",(0,0),(-1,-1),1),
                ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                ("ROUNDEDCORNERS",[7,7,7,7]),
            ]))
        row = Table([[nb, P(f, 9, MID, lead=14)]],
            colWidths=[20, W-20],
            style=TableStyle([
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
                ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                ("LINEBELOW",(0,0),(-1,-1),0.4,BORD),
            ]))
        story.append(row)
    story.append(Spacer(1,12))

    # ── 03 PATTERN ANALYSIS ──────────────────────────────────────────────────
    for e in section_hdr("03","ANOMALY PATTERN ANALYSIS"): story.append(e)
    story.append(Spacer(1,4))
    for p in r["patterns"]:
        dot = Table([[""]], colWidths=[7], rowHeights=[7],
            style=TableStyle([("BACKGROUND",(0,0),(-1,-1),AMB2),("ROUNDEDCORNERS",[2,2,2,2])]))
        row = Table([[dot, P(p, 9, MID, lead=14)]],
            colWidths=[13, W-13],
            style=TableStyle([
                ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
                ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                ("LINEBELOW",(0,0),(-1,-1),0.4,BORD),
            ]))
        story.append(row)
    story.append(Spacer(1,12))

    # ── 04 SCORE STATISTICS ──────────────────────────────────────────────────
    for e in section_hdr("04","SCORE STATISTICS"): story.append(e)
    story.append(Spacer(1,6))
    sc_w = W/4 - 3
    sc_data = [[
        [P("MINIMUM",7,MUTED,bold=True,align=TA_CENTER), P(f"{r['smin']:.4f}",13,RED2,bold=True,align=TA_CENTER)],
        [P("MAXIMUM",7,MUTED,bold=True,align=TA_CENTER), P(f"{r['smax']:.4f}",13,GRN2,bold=True,align=TA_CENTER)],
        [P("MEAN",   7,MUTED,bold=True,align=TA_CENTER), P(f"{r['smean']:.4f}",13,BLU2,bold=True,align=TA_CENTER)],
        [P("STD DEV",7,MUTED,bold=True,align=TA_CENTER), P(f"{r['sstd']:.4f}",13,AMB2,bold=True,align=TA_CENTER)],
    ]]
    sc_tbl = Table(sc_data, colWidths=[sc_w]*4,
        style=TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),LIGHT),
            ("BOX",(0,0),(0,0),0.4,HBORD),("BOX",(1,0),(1,0),0.4,HBORD),
            ("BOX",(2,0),(2,0),0.4,HBORD),("BOX",(3,0),(3,0),0.4,HBORD),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),9),
            ("ROUNDEDCORNERS",[4,4,4,4]),
        ]))
    story.append(sc_tbl)
    story.append(Spacer(1,12))

    # ── 05 DIAGRAMS ──────────────────────────────────────────────────────────
    story.append(PageBreak())
    for e in section_hdr("05","DIAGRAMS"): story.append(e)
    story.append(Spacer(1,6))

    charts = r["charts"]
    cw = W/2 - 4
    ch = cw * 0.52

    def chart_pair(k1, k2):
        def cell(k):
            if k in charts: return img(charts[k], cw, ch)
            return P("(chart unavailable)", 8, MUTED, align=TA_CENTER)
        row = Table([[cell(k1), cell(k2)]], colWidths=[cw, cw],
            style=TableStyle([
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),6),
                ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                ("COLPADDING",(0,0),(-1,-1),8),
            ]))
        return row

    story.append(chart_pair("score","top15"))
    story.append(chart_pair("pca","event"))
    story.append(Spacer(1,10))

    # ── 06 TOP 10 TABLE ──────────────────────────────────────────────────────
    for e in section_hdr("06","TOP 10 MOST ANOMALOUS RECORDS"): story.append(e)
    story.append(P("Records sorted by lowest anomaly score — highest anomaly likelihood.", 8, MUTED))
    story.append(Spacer(1,6))

    tbl_df = r["top10"].reset_index(drop=True)
    cols   = list(tbl_df.columns)
    cw_tbl = W / len(cols)

    tbl_data = [[P(c.upper().replace("_"," "), 7, WHITE, bold=True, align=TA_CENTER) for c in cols]]
    for _, row in tbl_df.iterrows():
        cells=[]
        for c in cols:
            val=str(row[c]) if pd.notna(row[c]) else "—"
            col_c = RED2 if c=="anomaly_score" and pd.notna(row[c]) and float(row[c])<-0.2 else MID
            cells.append(P(val[:20], 8, col_c))
        tbl_data.append(cells)

    data_tbl = Table(tbl_data, colWidths=[cw_tbl]*len(cols),
        style=TableStyle([
            ("BACKGROUND",(0,0),(-1,0),   NAVY),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT, WHITE]),
            ("GRID",(0,0),(-1,-1),0.3,HBORD),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ROUNDEDCORNERS",[4,4,4,4]),
        ]))
    story.append(data_tbl)
    story.append(Spacer(1,16))

    # ── FOOTER LINE ──────────────────────────────────────────────────────────
    story.append(HR(NAVY, 1.5, 4, 4))
    foot = Table([[
        P("FoxyDucky Task 3 Noctra Lupra  ·  Isolation Forest Anomaly Detection", 7, MUTED),
        P(f"{r['now'].strftime('%Y-%m-%d')}  ·  CONFIDENTIAL — FOR INTERNAL USE ONLY", 7, MUTED, align=TA_RIGHT),
    ]], colWidths=[W*0.6, W*0.4],
    style=TableStyle([
        ("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),0),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(foot)

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard charts (dark theme)
# ─────────────────────────────────────────────────────────────────────────────
def dash_score(df):
    _mpl_dark(); f,a=plt.subplots(figsize=(8,3.8),facecolor=DARK_BG); a.set_facecolor(CARD_BG)
    a.hist(df.loc[df["predicted_label"]==0,"anomaly_score"],bins=40,color=BLUE,alpha=.75,label="Normal",edgecolor="none")
    a.hist(df.loc[df["predicted_label"]==1,"anomaly_score"],bins=40,color=CRIMSON,alpha=.75,label="Anomaly",edgecolor="none")
    if (df["predicted_label"]==1).any():
        a.axvline(df.loc[df["predicted_label"]==1,"anomaly_score"].max(),color=AMBER,ls="--",lw=1.5,label="Threshold")
    a.set_title("Score distribution"); a.set_xlabel("Score"); a.set_ylabel("Count")
    a.legend(); a.grid(True)
    for s in a.spines.values(): s.set_edgecolor(BORDER)
    f.tight_layout(); return f

def dash_top15(df):
    _mpl_dark(); top=df.sort_values("anomaly_score").head(15)
    colors=[CRIMSON if v==1 else BLUE for v in top["predicted_label"]]
    f,a=plt.subplots(figsize=(8,3.8),facecolor=DARK_BG); a.set_facecolor(CARD_BG)
    a.barh([str(i) for i in range(1,len(top)+1)],top["anomaly_score"],color=colors,edgecolor="none",height=.65)
    a.axvline(0,color=AMBER,ls="--",lw=1.2,alpha=.7); a.set_title("Top 15 lowest scores")
    a.set_xlabel("Score"); a.invert_yaxis()
    a.legend(handles=[mpatches.Patch(color=CRIMSON,label="Anomaly"),mpatches.Patch(color=BLUE,label="Normal")])
    a.grid(True,axis="x")
    for s in a.spines.values(): s.set_edgecolor(BORDER)
    f.tight_layout(); return f

def dash_pca(df,X):
    if X.shape[1]<3: return None
    try:
        _mpl_dark(); comps=pca2(X.values)
        pdf=pd.DataFrame(comps,columns=["PC1","PC2"]); pdf["lbl"]=df["predicted_label"].values
        f,a=plt.subplots(figsize=(8,3.8),facecolor=DARK_BG); a.set_facecolor(CARD_BG)
        n=pdf[pdf["lbl"]==0]; av=pdf[pdf["lbl"]==1]
        a.scatter(n["PC1"],n["PC2"],s=14,color=BLUE,alpha=.5,label="Normal",linewidths=0)
        a.scatter(av["PC1"],av["PC2"],s=45,color=CRIMSON,alpha=.85,label="Anomaly",marker="X",linewidths=0)
        a.set_title("PCA scatter"); a.set_xlabel("PC1"); a.set_ylabel("PC2")
        a.legend(); a.grid(True,alpha=.4)
        for s in a.spines.values(): s.set_edgecolor(BORDER)
        f.tight_layout(); return f
    except: return None

def dash_event(df):
    if "event_type" not in df.columns: return None
    _mpl_dark(); f,a=plt.subplots(figsize=(8,3.8),facecolor=DARK_BG); a.set_facecolor(CARD_BG)
    sns.countplot(x="event_type",hue="anomaly_prediction",data=df,
                  palette={1:BLUE,-1:CRIMSON},ax=a,edgecolor="none")
    a.set_title("Event type breakdown"); a.set_xlabel("Event type"); a.set_ylabel("Count")
    a.legend(handles=[mpatches.Patch(color=BLUE,label="Normal"),mpatches.Patch(color=CRIMSON,label="Anomaly")])
    a.grid(True,axis="y"); plt.xticks(rotation=30,ha="right")
    for s in a.spines.values(): s.set_edgecolor(BORDER)
    f.tight_layout(); return f

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
# Welcome
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
# Run analysis
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running Isolation Forest pipeline…"):
        df,X,err = run_analysis(uploaded.getvalue(), cont_rate)
    if err:
        st.error(f"**Error:** {err}"); st.stop()
    st.session_state.update({"df":df,"X":X,"cont":cont_rate,"report":None,"pdf":None})

if "df" not in st.session_state:
    st.info("Upload a file and click **▶ Run Detection** to start."); st.stop()

df   = st.session_state["df"]
X    = st.session_state["X"]
cont = st.session_state["cont"]
n_a  = int((df["predicted_label"]==1).sum())
n_n  = int((df["predicted_label"]==0).sum())
tot  = len(df); rat = n_a/tot if tot else 0

# ─────────────────────────────────────────────────────────────────────────────
# Banner + metrics
# ─────────────────────────────────────────────────────────────────────────────
bg_c="#2a0f0f" if rat>0.1 else "#0f2e1e"
tx_c="#ef4444" if rat>0.1 else "#22c55e"
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

c1,c2,c3,c4=st.columns(4)
c1.metric("Total records", f"{tot:,}")
c2.metric("Anomalies 🔴",  f"{n_a:,}",  delta=f"{rat:.1%} of total",   delta_color="inverse")
c3.metric("Normal 🟢",     f"{n_n:,}",  delta=f"{1-rat:.1%} of total")
c4.metric("Anomaly ratio", f"{rat:.2%}")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard charts 2×2
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='s-card'><div class='s-card-title'>📊 Visualisations</div>",
            unsafe_allow_html=True)
r1l,r1r=st.columns(2)
with r1l:
    f=dash_score(df); st.pyplot(f); plt.close(f)
with r1r:
    f=dash_top15(df); st.pyplot(f); plt.close(f)
r2l,r2r=st.columns(2)
with r2l:
    f=dash_pca(df,X)
    if f: st.pyplot(f); plt.close(f)
    else: st.info("Too few features for PCA.")
with r2r:
    f=dash_event(df)
    if f: st.pyplot(f); plt.close(f)
    else: st.info("No `event_type` column.")
st.markdown("</div>",unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Top 20 table
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
st.markdown("</div>",unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Report section — text preview + PDF download
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='s-card'><div class='s-card-title'>📄 Incident Report</div>",
            unsafe_allow_html=True)

bc,ic=st.columns([1,4])
with bc:
    gen_btn=st.button("📄 Generate Report", key="gen_report", use_container_width=True)
with ic:
    st.caption("Preview as styled text below. Download as a clean A4 PDF with diagrams and table.")

if gen_btn:
    with st.spinner("Building report…"):
        rpt = generate_report(df, X, cont)
        rpt["charts"] = make_pdf_charts(df, X)
    with st.spinner("Rendering PDF…"):
        try:
            pdf_bytes = build_pdf(rpt)
            st.session_state["pdf"] = pdf_bytes
        except ImportError:
            st.warning("Run `pip install reportlab` to enable PDF download.")
            st.session_state["pdf"] = None
    st.session_state["report"] = rpt

rpt = st.session_state.get("report")

if rpt:
    sev = rpt["sev"]
    icons = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🔵","LOW":"🟢"}

    # ── TEXT PREVIEW ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="rpt-wrap">
      <div class="rpt-hdr">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
          <div>
            <p class="rpt-title">INCIDENT REPORT</p>
            <p class="rpt-sub">Network Anomaly Detection — Isolation Forest Analysis</p>
          </div>
          <span class="rpt-badge sev-{sev}">{icons.get(sev,'')} {sev}</span>
        </div>
        <div class="rpt-meta">
          <div class="rpt-mi">Generated: <span>{rpt['now'].strftime('%d %B %Y  %H:%M:%S')}</span></div>
          <div class="rpt-mi">Contamination: <span>{rpt['cont']:.0%}</span></div>
          <div class="rpt-mi">Algorithm: <span>Isolation Forest (Pure NumPy)</span></div>
        </div>
      </div>

      <div class="stat-row">
        <div class="stat-box"><div class="stat-top" style="background:#3b82f6;"></div>
          <div class="stat-lbl">Total Records</div><div class="stat-val">{rpt['n_tot']:,}</div>
          <div class="stat-sub">Records processed</div></div>
        <div class="stat-box"><div class="stat-top" style="background:#ef4444;"></div>
          <div class="stat-lbl">Anomalies</div><div class="stat-val">{rpt['n_a']:,}</div>
          <div class="stat-sub">{rpt['ratio']:.2%} of total</div></div>
        <div class="stat-box"><div class="stat-top" style="background:#22c55e;"></div>
          <div class="stat-lbl">Normal</div><div class="stat-val">{rpt['n_n']:,}</div>
          <div class="stat-sub">{1-rpt['ratio']:.2%} of total</div></div>
        <div class="stat-box"><div class="stat-top" style="background:{rpt['sev_col']};"></div>
          <div class="stat-lbl">Anomaly Ratio</div><div class="stat-val">{rpt['ratio']:.2%}</div>
          <div class="stat-sub">{sev} severity</div></div>
      </div>

      <div class="sec-hdr"><div class="sec-num">01</div><div class="sec-title">Executive Summary</div></div>
      <div class="rpt-sum">{rpt['summary']}</div>
      <div class="sev-box">
        <div class="sev-text"><b>Severity ({sev}):</b> {rpt['sev_r']}</div>
      </div>

      <div class="sec-hdr"><div class="sec-num">02</div><div class="sec-title">Key Findings</div></div>
      <ul class="find-list">
        {''.join(f'<li><div class="find-num">{i+1}</div><div class="find-text">{f}</div></li>' for i,f in enumerate(rpt['findings']))}
      </ul>

      <div class="sec-hdr"><div class="sec-num">03</div><div class="sec-title">Anomaly Pattern Analysis</div></div>
      <ul class="pat-list">
        {''.join(f'<li><div class="pat-dot"></div><div class="pat-text">{p}</div></li>' for p in rpt['patterns'])}
      </ul>

      <div class="sec-hdr"><div class="sec-num">04</div><div class="sec-title">Score Statistics</div></div>
      <div class="sc-grid">
        <div class="sc-box"><div class="sc-lbl">Minimum</div><div class="sc-val" style="color:#ef4444;">{rpt['smin']:.4f}</div></div>
        <div class="sc-box"><div class="sc-lbl">Maximum</div><div class="sc-val" style="color:#22c55e;">{rpt['smax']:.4f}</div></div>
        <div class="sc-box"><div class="sc-lbl">Mean</div><div class="sc-val" style="color:#3b82f6;">{rpt['smean']:.4f}</div></div>
        <div class="sc-box"><div class="sc-lbl">Std Dev</div><div class="sc-val" style="color:#f59e0b;">{rpt['sstd']:.4f}</div></div>
      </div>

      <div class="sec-hdr"><div class="sec-num">05</div><div class="sec-title">Diagrams</div></div>
    </div>
    """, unsafe_allow_html=True)

    # inline chart images
    charts = rpt["charts"]
    ch1,ch2=st.columns(2)
    with ch1:
        if "score"  in charts: st.image(charts["score"],  use_container_width=True)
    with ch2:
        if "top15"  in charts: st.image(charts["top15"],  use_container_width=True)
    ch3,ch4=st.columns(2)
    with ch3:
        if "pca"    in charts: st.image(charts["pca"],    use_container_width=True)
        else: st.info("PCA unavailable.")
    with ch4:
        if "event"  in charts: st.image(charts["event"],  use_container_width=True)
        else: st.info("No event_type column.")

    # top 10 table preview
    st.markdown("<div class='sec-hdr'><div class='sec-num'>06</div><div class='sec-title'>Top 10 Most Anomalous Records</div></div>",
                unsafe_allow_html=True)
    top10_show = rpt["top10"].copy()
    if "anomaly_score" in top10_show.columns:
        top10_show["anomaly_score"] = top10_show["anomaly_score"].round(4)
    st.dataframe(top10_show, use_container_width=True, hide_index=True)

    # footer
    st.markdown(f"""
    <div class="rpt-footer">
      <p>FoxyDucky Task 3 Noctra Lupra · Isolation Forest Anomaly Detection</p>
      <p>Generated: {rpt['now'].strftime('%Y-%m-%d')} · CONFIDENTIAL</p>
    </div>""", unsafe_allow_html=True)

    # ── PDF DOWNLOAD ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.get("pdf"):
        st.download_button(
            "⬇️ Download PDF",
            data=st.session_state["pdf"],
            file_name=f"incident_report_{rpt['now'].strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="dl_pdf",
        )
    else:
        st.info("Install ReportLab to enable PDF:  `pip install reportlab`")

st.markdown("</div>", unsafe_allow_html=True)