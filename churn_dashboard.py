import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Configuração ────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analytics | Gabriel Naumann",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* ── Base ─────────────────────────────────────────── */
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background-color: #06080f;
        background-image:
            radial-gradient(ellipse 80% 50% at 20% 10%, rgba(20,184,166,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 90%, rgba(245,158,11,0.06) 0%, transparent 60%);
        background-attachment: fixed;
    }

    /* ── Sidebar ──────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #090c14;
        border-right: 1px solid #1a2235;
    }
    section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] a { color: #14b8a6 !important; }
    section[data-testid="stSidebar"] h2 { color: #e2e8f0 !important; }

    /* ── Tipografia ───────────────────────────────────── */
    h1 {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.8px;
        color: #f1f5f9 !important;
        border-bottom: 2px solid #14b8a6;
        padding-bottom: 10px;
        display: inline-block;
    }
    h2 {
        color: #cbd5e1 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #14b8a6 !important;
    }
    h3 { color: #e2e8f0 !important; font-weight: 500 !important; }
    p, label, div { color: #94a3b8; }

    /* ── Metric Cards ─────────────────────────────────── */
    [data-testid="metric-container"] {
        background-color: #0d1117;
        border: 1px solid #1e2d3d;
        border-top: 2px solid #14b8a6;
        border-radius: 4px 4px 8px 8px;
        padding: 20px;
        transition: all 0.25s ease;
    }
    [data-testid="metric-container"]:hover {
        border-top-color: #f59e0b;
        background-color: #0f1520;
    }
    [data-testid="metric-container"] label {
        color: #475569 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600 !important;
        font-family: 'DM Mono', monospace !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 30px !important;
        font-weight: 700 !important;
        font-family: 'DM Mono', monospace !important;
        letter-spacing: -1px;
    }

    /* ── Botões ───────────────────────────────────────── */
    .stButton > button {
        background-color: transparent !important;
        color: #14b8a6 !important;
        border: 1px solid #14b8a6 !important;
        border-radius: 4px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 13px;
        width: 100%;
        letter-spacing: 0.5px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #14b8a6 !important;
        color: #06080f !important;
    }

    /* ── Sliders ──────────────────────────────────────── */
    .stSlider > div > div > div { background-color: #14b8a6; }

    /* ── Tabs ─────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 1px solid #1e2d3d;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #475569;
        border-radius: 0;
        padding: 10px 20px;
        font-weight: 500;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #14b8a6 !important;
        border-bottom: 2px solid #14b8a6 !important;
    }

    /* ── Expanders ────────────────────────────────────── */
    .streamlit-expanderHeader {
        background-color: #0d1117 !important;
        border: 1px solid #1e2d3d !important;
        border-left: 3px solid #f59e0b !important;
        border-radius: 0 4px 4px 0 !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background-color: #090c14 !important;
        border: 1px solid #1e2d3d !important;
        border-top: none !important;
    }

    /* ── Selectbox ────────────────────────────────────── */
    .stSelectbox > div > div {
        background-color: #0d1117 !important;
        border: 1px solid #1e2d3d !important;
        border-radius: 4px !important;
        color: #e2e8f0 !important;
        font-family: 'DM Mono', monospace !important;
    }

    /* ── Number Input ─────────────────────────────────── */
    .stNumberInput > div > div > input {
        background-color: #0d1117 !important;
        border: 1px solid #1e2d3d !important;
        color: #e2e8f0 !important;
        border-radius: 4px !important;
    }

    /* ── Divisores ────────────────────────────────────── */
    hr {
        border: none;
        height: 1px;
        background-color: #1e2d3d;
        margin: 1.8rem 0;
    }

    /* ── Badges ───────────────────────────────────────── */
    .badge-high {
        background-color: #1a0a0a;
        color: #f87171;
        padding: 6px 16px;
        border-radius: 3px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
        border: 1px solid #ef4444;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-med {
        background-color: #1a120a;
        color: #fbbf24;
        padding: 6px 16px;
        border-radius: 3px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
        border: 1px solid #f59e0b;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-low {
        background-color: #051a14;
        color: #34d399;
        padding: 6px 16px;
        border-radius: 3px;
        font-weight: 700;
        font-size: 13px;
        display: inline-block;
        border: 1px solid #14b8a6;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ── Cards customizados ───────────────────────────── */
    .info-box {
        background-color: #0d1117;
        border-left: 3px solid #14b8a6;
        border-radius: 0 4px 4px 0;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 14px;
    }
    .action-box {
        background-color: #080f0d;
        border-left: 3px solid #34d399;
        border-radius: 0 4px 4px 0;
        padding: 11px 16px;
        margin: 5px 0;
        font-size: 14px;
        color: #a7f3d0 !important;
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Dataframe ────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid #1e2d3d !important;
        border-radius: 4px;
    }

    /* ── Scrollbar ────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #06080f; }
    ::-webkit-scrollbar-thumb { background: #1e2d3d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #14b8a6; }
</style>
""", unsafe_allow_html=True)

PALETTE = ['#14b8a6','#f59e0b','#6366f1','#34d399','#f87171','#38bdf8']

# ── Dados e modelo ──────────────────────────────────────────
@st.cache_resource
def load_all():
    SEED = 42
    np.random.seed(SEED)
    N = 15_000
    profile_probs = [0.30,0.25,0.20,0.15,0.10]
    profiles = np.random.choice(['engajado','ocasional','inativo','novo','vip'], size=N, p=profile_probs)

    records = []
    for i, profile in enumerate(profiles):
        if profile == 'engajado':
            tenure=np.random.randint(12,60); monthly_txn=np.random.poisson(25)
            avg_balance=np.random.lognormal(8.5,0.5); products=np.random.randint(3,7)
            login_days=np.random.randint(18,30); support_calls=np.random.poisson(0.5); churn_base=0.04
        elif profile == 'ocasional':
            tenure=np.random.randint(6,36); monthly_txn=np.random.poisson(8)
            avg_balance=np.random.lognormal(7.5,0.7); products=np.random.randint(1,4)
            login_days=np.random.randint(5,18); support_calls=np.random.poisson(1.2); churn_base=0.18
        elif profile == 'inativo':
            tenure=np.random.randint(3,24); monthly_txn=np.random.poisson(2)
            avg_balance=np.random.lognormal(6.5,1.0); products=np.random.randint(1,3)
            login_days=np.random.randint(0,6); support_calls=np.random.poisson(2.5); churn_base=0.45
        elif profile == 'novo':
            tenure=np.random.randint(1,6); monthly_txn=np.random.poisson(12)
            avg_balance=np.random.lognormal(7.0,0.8); products=np.random.randint(1,3)
            login_days=np.random.randint(8,25); support_calls=np.random.poisson(1.8); churn_base=0.22
        else:
            tenure=np.random.randint(24,72); monthly_txn=np.random.poisson(45)
            avg_balance=np.random.lognormal(10.0,0.4); products=np.random.randint(5,9)
            login_days=np.random.randint(22,30); support_calls=np.random.poisson(0.3); churn_base=0.02

        age=np.random.randint(18,65)
        pix_txn=int(monthly_txn*np.random.uniform(0.3,0.7))
        credit_card_spend=round(abs(np.random.lognormal(6.5,1.2)),2)
        has_investment=int(np.random.random()<(0.7 if profile=='vip' else 0.3))
        has_insurance=int(np.random.random()<(0.5 if profile in ['vip','engajado'] else 0.15))
        has_loan=int(np.random.random()<0.25)
        nps_score=np.random.choice(range(0,11),
            p=[0.02,0.02,0.03,0.04,0.05,0.06,0.08,0.15,0.20,0.20,0.15]
            if profile in ['engajado','vip'] else
            [0.08,0.08,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.08,0.06])
        last_txn_days=max(0,int(np.random.exponential(
            5 if profile in ['engajado','vip'] else 20 if profile=='ocasional' else 45)))

        churn_prob = churn_base
        if support_calls>=3:       churn_prob+=0.20
        if login_days<=3:          churn_prob+=0.25
        if monthly_txn==0:         churn_prob+=0.30
        if nps_score<=4:           churn_prob+=0.15
        if last_txn_days>60:       churn_prob+=0.20
        if products==1:            churn_prob+=0.10
        if avg_balance<500:        churn_prob+=0.08
        if tenure<3:               churn_prob+=0.12
        if has_investment and has_insurance: churn_prob*=0.4
        if products>=5:            churn_prob*=0.5
        if nps_score>=9:           churn_prob*=0.3
        churn_prob=min(churn_prob+np.random.uniform(-0.03,0.03),0.95)
        churned=int(np.random.random()<churn_prob)

        records.append({
            'customer_id':f'CLI{i+1:07d}','profile':profile,'age':age,
            'tenure_months':tenure,'monthly_txn':max(0,monthly_txn),
            'pix_txn':max(0,pix_txn),'avg_balance':round(max(0,avg_balance),2),
            'credit_card_spend':credit_card_spend,'products':products,
            'login_days_month':login_days,'support_calls':max(0,support_calls),
            'last_txn_days_ago':last_txn_days,'has_investment':has_investment,
            'has_insurance':has_insurance,'has_loan':has_loan,
            'nps_score':nps_score,'churned':churned,
        })

    df = pd.DataFrame(records)

    # RFM
    df['R'] = df['last_txn_days_ago']
    df['F'] = df['monthly_txn']
    df['M'] = df['avg_balance'] + df['credit_card_spend']
    df['R_score'] = pd.qcut(df['R'].rank(method='first'),5,labels=[5,4,3,2,1]).astype(int)
    df['F_score'] = pd.qcut(df['F'].rank(method='first'),5,labels=[1,2,3,4,5]).astype(int)
    df['M_score'] = pd.qcut(df['M'].rank(method='first'),5,labels=[1,2,3,4,5]).astype(int)
    df['RFM_score'] = df['R_score']+df['F_score']+df['M_score']
    df['RFM_segment'] = df['RFM_score'].apply(lambda x:
        'Champions' if x>=13 else 'Loyal' if x>=10 else
        'Potential' if x>=7  else 'At Risk' if x>=5 else 'Lost')
    df['engagement_score'] = (
        df['login_days_month']/30*0.3 +
        (df['monthly_txn']/df['monthly_txn'].max())*0.3 +
        df['products']/df['products'].max()*0.2 +
        (1-df['last_txn_days_ago']/df['last_txn_days_ago'].max())*0.2).round(4)
    df['product_depth']  = df['products']/9
    df['balance_trend']  = np.where(df['avg_balance']>df['avg_balance'].median(),1,0)

    # KMeans
    feats_cluster = ['R_score','F_score','M_score','login_days_month',
                     'products','tenure_months','engagement_score','nps_score']
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(df[feats_cluster])
    km     = KMeans(n_clusters=5, random_state=SEED, n_init=10)
    df['cluster'] = km.fit_predict(X_sc)

    cluster_names = {}
    cp = df.groupby('cluster').agg(churn_rate=('churned','mean'),
                                    avg_products=('products','mean'),
                                    avg_engagement=('engagement_score','mean')).round(3)
    for c in range(5):
        r = cp.loc[c]
        if r['churn_rate']<0.08 and r['avg_products']>=4:   cluster_names[c]='VIP Fidelizado'
        elif r['churn_rate']<0.12 and r['avg_engagement']>0.6: cluster_names[c]='Engajado Ativo'
        elif r['churn_rate']>0.40:                           cluster_names[c]='Alto Risco'
        elif r['churn_rate']>0.20:                           cluster_names[c]='Risco Moderado'
        else:                                                cluster_names[c]='Ocasional'
    df['cluster_name'] = df['cluster'].map(cluster_names)

    # Modelo
    FEATURES = ['age','tenure_months','monthly_txn','pix_txn','avg_balance',
                'credit_card_spend','products','login_days_month','support_calls',
                'last_txn_days_ago','has_investment','has_insurance','has_loan',
                'nps_score','R_score','F_score','M_score','RFM_score',
                'engagement_score','product_depth','balance_trend']
    X = df[FEATURES]; y = df['churned']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=SEED)
    model = XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,
                          use_label_encoder=False,eval_metric='auc',random_state=SEED,n_jobs=-1)
    model.fit(X_train,y_train)

    df['churn_prob'] = model.predict_proba(X)[:,1]
    df['churn_risk']  = pd.cut(df['churn_prob'],bins=[0,0.2,0.5,0.8,1.0],
                                labels=['Baixo','Médio','Alto','Crítico'])

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer(X_test)
    shap_imp   = pd.DataFrame({'feature':FEATURES,
                                'mean_shap':np.abs(shap_vals.values).mean(axis=0)
                               }).sort_values('mean_shap',ascending=False)

    return df, model, FEATURES, auc, shap_imp, cluster_names

df, model, FEATURES, auc, shap_imp, cluster_names = load_all()

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📉 Churn Analytics")
    st.markdown("**Gabriel Alessi Naumann**")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/gabriel-alessi-naumann/) · [GitHub](https://github.com/GabrielAlessi)")
    st.markdown("---")
    page = st.selectbox("Página", [
        "🏠 Visão Geral",
        "🔵 Análise RFM e Clusters",
        "🔴 Score Individual",
        "🎯 Plano de Intervenção",
    ])
    st.markdown("---")
    st.markdown(f"**Modelo:** XGBoost  \n**ROC-AUC:** `{auc:.4f}`  \n**Base:** {len(df):,} clientes")

def dark_fig(figsize=(12,5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor='#06080f')
    ax.set_facecolor('#111827')
    ax.tick_params(colors='#9aa0a6')
    for spine in ax.spines.values(): spine.set_edgecolor('#1a2234')
    return fig, ax

def dark_figs(nrows, ncols, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='#06080f')
    for ax in np.array(axes).flatten():
        ax.set_facecolor('#0a0d14')
        ax.tick_params(colors='#475569')
        for spine in ax.spines.values(): spine.set_edgecolor('#1e2d3d')
    return fig, axes

# ── PAGE 1: Visão Geral ─────────────────────────────────────
if page == "🏠 Visão Geral":

    # ── Variáveis de suporte ─────────────────────────────────
    total    = len(df)
    churned  = int(df['churned'].sum())
    churn_rt = df['churned'].mean() * 100
    criticos = int((df['churn_risk'] == 'Crítico').sum())
    alto     = int((df['churn_risk'] == 'Alto').sum())

    # ── Header ──────────────────────────────────────────────
    st.title("Análise Preditiva de Churn")
    st.markdown("<p style='color:#475569;font-size:15px;margin-top:-10px;'>Banco Digital · 15.000 clientes · Modelo XGBoost · ROC-AUC 0.92</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Linha 1: 5 KPIs lado a lado ─────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.metric("Total Clientes", f"{total:,}")
    with k2: st.metric("Churn Rate", f"{churn_rt:.1f}%")
    with k3: st.metric("Churned", f"{churned:,}")
    with k4: st.metric("Risco Crítico", f"{criticos:,}")
    with k5: st.metric("Risco Alto", f"{alto:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Linha 2: Gráfico largo (7) + painel lateral (3) ─────
    col_main, col_side = st.columns([7, 3])

    with col_main:
        st.markdown("## Churn Rate por Perfil")
        fig, ax = dark_fig((10, 4))
        cp = df.groupby('profile')['churned'].mean().sort_values(ascending=True) * 100
        bars = ax.barh(cp.index, cp.values, color=[
            PALETTE[0] if v < churn_rt else PALETTE[4] for v in cp.values
        ], alpha=0.9, height=0.55)
        ax.axvline(churn_rt, color=PALETTE[1], linestyle='--', linewidth=1.5,
                   label=f'Média {churn_rt:.1f}%', alpha=0.8)
        for bar, v in zip(bars, cp.values):
            ax.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{v:.1f}%', va='center', color='#cbd5e1', fontsize=10)
        ax.set_xlabel('Churn Rate (%)', color='#475569')
        ax.legend(facecolor='#0a0d14', labelcolor='#cbd5e1', fontsize=9)
        ax.set_xlim(0, cp.max() * 1.2)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_side:
        st.markdown("## Distribuição de Risco")
        fig, ax = dark_fig((4, 4))
        rc = df['churn_risk'].value_counts().reindex(['Crítico','Alto','Médio','Baixo'])
        colors_r = [PALETTE[4], PALETTE[1], PALETTE[2], PALETTE[3]]
        wedges, texts, autotexts = ax.pie(
            rc.values, labels=rc.index, colors=colors_r,
            autopct='%1.0f%%', startangle=90,
            wedgeprops={'edgecolor':'#06080f','linewidth':2},
            pctdistance=0.75)
        for t in texts:     t.set_color('#94a3b8'); t.set_fontsize(9)
        for t in autotexts: t.set_color('#f1f5f9'); t.set_fontsize(9); t.set_fontweight('bold')
        ax.set_facecolor('#06080f')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Linha 3: 3 colunas iguais ───────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("## Churn × Produtos")
        fig, ax = dark_fig((5, 3.5))
        cp2 = df.groupby('products')['churned'].mean() * 100
        ax.fill_between(cp2.index, cp2.values, alpha=0.15, color=PALETTE[0])
        ax.plot(cp2.index, cp2.values, marker='o', color=PALETTE[0],
                linewidth=2, markersize=6)
        ax.set_xlabel('Nº de Produtos', color='#475569', fontsize=10)
        ax.set_ylabel('Churn %', color='#475569', fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.markdown("## Churn × NPS")
        fig, ax = dark_fig((5, 3.5))
        df['nps_group'] = pd.cut(df['nps_score'], bins=[-1,6,8,10],
                                  labels=['Detrator (0-6)','Neutro (7-8)','Promotor (9-10)'])
        cn = df.groupby('nps_group', observed=True)['churned'].mean() * 100
        ax.bar(cn.index, cn.values,
               color=[PALETTE[4], PALETTE[1], PALETTE[3]], alpha=0.9, width=0.5)
        for i, v in enumerate(cn.values):
            ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', color='#cbd5e1', fontsize=10)
        ax.set_ylabel('Churn %', color='#475569', fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c3:
        st.markdown("## Top Features SHAP")
        fig, ax = dark_fig((5, 3.5))
        top8 = shap_imp.head(8)
        ax.barh(top8['feature'][::-1], top8['mean_shap'][::-1],
                color=PALETTE[0], alpha=0.85, height=0.6)
        ax.set_xlabel('Mean |SHAP|', color='#475569', fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

elif page == "🔵 Análise RFM e Clusters":

    st.title("Segmentação RFM & Clusters")
    st.markdown("<p style='color:#475569;font-size:15px;margin-top:-10px;'>K-Means sobre features RFM + comportamentais · 5 segmentos identificados</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Linha 1: tabela larga + pizza ───────────────────────
    col_tbl, col_pie = st.columns([7, 3])

    with col_tbl:
        st.markdown("## Perfil dos Segmentos")
        cluster_stats = df.groupby('cluster_name').agg(
            Clientes    = ('customer_id','count'),
            Churn_Rate  = ('churned','mean'),
            RFM_Score   = ('RFM_score','mean'),
            Produtos    = ('products','mean'),
            Engagement  = ('engagement_score','mean'),
            NPS         = ('nps_score','mean'),
        ).round(2)
        cluster_stats['Churn_Rate'] = (cluster_stats['Churn_Rate'] * 100).round(1).astype(str) + '%'
        st.dataframe(cluster_stats.style.background_gradient(
            subset=['RFM_Score','Produtos','Engagement'], cmap='Blues'),
            use_container_width=True)

    with col_pie:
        st.markdown("## Distribuição")
        fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor='#06080f')
        ax.set_facecolor('#06080f')
        sz = df['cluster_name'].value_counts()
        wedges, texts, autotexts = ax.pie(
            sz.values, labels=None,
            colors=PALETTE[:len(sz)], autopct='%1.0f%%',
            startangle=90,
            wedgeprops={'edgecolor':'#06080f','linewidth':1.5},
            pctdistance=0.72)
        for t in autotexts:
            t.set_color('#f1f5f9'); t.set_fontweight('bold'); t.set_fontsize(9)
        ax.legend(sz.index, loc='lower center', bbox_to_anchor=(0.5, -0.28),
                  ncol=1, facecolor='#06080f', labelcolor='#94a3b8',
                  fontsize=8, framealpha=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Linha 2: scatter largo (8) + churn por cluster (4) ──
    col_sc, col_bar = st.columns([8, 4])

    with col_sc:
        st.markdown("## Mapa de Engajamento × RFM")
        fig, ax = dark_fig((10, 5))
        for i, cid in enumerate(sorted(df['cluster'].unique())):
            mask = df['cluster'] == cid
            name = cluster_names.get(cid, f'C{cid}')
            ax.scatter(df[mask]['engagement_score'], df[mask]['RFM_score'],
                      color=PALETTE[i % len(PALETTE)], alpha=0.25, s=12, label=name)
        ax.set_xlabel('Engagement Score', color='#475569')
        ax.set_ylabel('RFM Score', color='#475569')
        leg = ax.legend(facecolor='#0a0d14', labelcolor='#cbd5e1',
                        fontsize=9, markerscale=3, framealpha=0.8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_bar:
        st.markdown("## Churn por Cluster")
        fig, ax = dark_fig((5, 5))
        cc = df.groupby('cluster_name')['churned'].mean().sort_values() * 100
        colors_c = [PALETTE[4] if v > 25 else PALETTE[1] if v > 12 else PALETTE[3]
                    for v in cc.values]
        ax.barh(cc.index, cc.values, color=colors_c, alpha=0.9, height=0.55)
        ax.set_xlabel('%', color='#475569')
        for i, v in enumerate(cc.values):
            ax.text(v + 0.3, i, f'{v:.1f}%', va='center', color='#cbd5e1', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Linha 3: scores RFM em 3 colunas ────────────────────
    st.markdown("## Distribuição dos Scores RFM por Status")
    r1, r2, r3 = st.columns(3)
    for col_r, (score, label) in zip([r1,r2,r3],
        [('R_score','Recency'), ('F_score','Frequency'), ('M_score','Monetary')]):
        with col_r:
            fig, ax = dark_fig((5, 3))
            for cv, color, name in [(0,PALETTE[0],'Ativo'),(1,PALETTE[4],'Churn')]:
                counts = df[df['churned']==cv][score].value_counts().sort_index()
                ax.bar(counts.index + (0.2 if cv==1 else -0.2),
                       counts.values, 0.35, label=name, color=color, alpha=0.85)
            ax.set_title(label, color='#cbd5e1', fontsize=11)
            ax.set_xlabel('Score (1=pior, 5=melhor)', color='#475569', fontsize=9)
            ax.legend(facecolor='#0a0d14', labelcolor='#cbd5e1', fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

elif page == "🔴 Score Individual":

    st.title("Score de Churn Individual")
    st.markdown("<p style='color:#475569;font-size:15px;margin-top:-10px;'>Insira os dados do cliente para calcular o risco em tempo real</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Layout: inputs (4) + resultado (8) ──────────────────
    col_form, col_result = st.columns([4, 8])

    with col_form:
        st.markdown("## Dados do Cliente")

        with st.expander("📊 Comportamento", expanded=True):
            monthly_txn_v = st.slider("Transações/Mês", 0, 60, 10)
            login_days_v  = st.slider("Dias de Login/Mês", 0, 30, 10)
            last_txn_v    = st.slider("Dias desde última transação", 0, 120, 15)
            pix_v         = st.slider("Transações PIX/Mês", 0, 40, 5)

        with st.expander("👤 Perfil", expanded=True):
            age_v     = st.slider("Idade", 18, 65, 35)
            tenure_v  = st.slider("Tempo de Casa (meses)", 1, 72, 12)
            products_v= st.slider("Nº de Produtos", 1, 9, 2)
            support_v = st.slider("Chamadas de Suporte", 0, 10, 1)
            nps_v     = st.slider("NPS Score", 0, 10, 7)

        with st.expander("💰 Financeiro", expanded=True):
            balance_v  = st.number_input("Saldo Médio (R$)", 0, 200000, 5000, 500)
            cc_spend_v = st.number_input("Gasto Cartão/Mês (R$)", 0, 50000, 1500, 100)
            inv_v  = st.checkbox("Possui Investimento?")
            ins_v  = st.checkbox("Possui Seguro?")
            loan_v = st.checkbox("Possui Empréstimo?")

    with col_result:
        # Calcular scores
        def calc_r_score(v):
            if v <= 5:    return 5
            elif v <= 15: return 4
            elif v <= 30: return 3
            elif v <= 60: return 2
            else:         return 1

        r_score = calc_r_score(last_txn_v)
        f_score = min(5, max(1, int(monthly_txn_v / 12) + 1))
        m_val   = balance_v + cc_spend_v
        m_score = min(5, max(1, int(m_val / 15000) + 1))
        rfm_s   = r_score + f_score + m_score
        eng     = round(
            login_days_v/30*0.3 + min(monthly_txn_v/60,1)*0.3 +
            products_v/9*0.2 + max(0, 1-last_txn_v/120)*0.2, 4)
        prod_d  = products_v/9
        bal_tr  = 1 if balance_v > 5000 else 0

        instance = pd.DataFrame([{
            'age':age_v,'tenure_months':tenure_v,'monthly_txn':monthly_txn_v,
            'pix_txn':pix_v,'avg_balance':balance_v,'credit_card_spend':cc_spend_v,
            'products':products_v,'login_days_month':login_days_v,'support_calls':support_v,
            'last_txn_days_ago':last_txn_v,'has_investment':int(inv_v),
            'has_insurance':int(ins_v),'has_loan':int(loan_v),'nps_score':nps_v,
            'R_score':r_score,'F_score':f_score,'M_score':m_score,'RFM_score':rfm_s,
            'engagement_score':eng,'product_depth':prod_d,'balance_trend':bal_tr,
        }])

        prob    = float(model.predict_proba(instance)[0,1])
        score   = int((1-prob)*1000)
        risk    = 'Crítico' if prob>0.8 else 'Alto' if prob>0.5 else 'Médio' if prob>0.2 else 'Baixo'
        rfm_seg = 'Champions' if rfm_s>=13 else 'Loyal' if rfm_s>=10 else 'Potential' if rfm_s>=7 else 'At Risk' if rfm_s>=5 else 'Lost'

        # ── KPIs do resultado ───────────────────────────────
        st.markdown("## Resultado da Análise")
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Prob. Churn",     f"{prob*100:.1f}%")
        with m2: st.metric("Score Retenção",  f"{score}")
        with m3: st.metric("Nível de Risco",  risk)
        with m4: st.metric("Segmento RFM",    rfm_seg)

        badge_cls = 'badge-high' if risk in ['Crítico','Alto'] else 'badge-med' if risk=='Médio' else 'badge-low'
        label_b   = '⚠ ALTO RISCO DE CHURN' if risk in ['Crítico','Alto'] else '— RISCO MODERADO' if risk=='Médio' else '✓ BAIXO RISCO'
        st.markdown(f"<br><div class='{badge_cls}'>{label_b}</div><br>", unsafe_allow_html=True)

        st.markdown("---")

        # ── Linha: gauge (prob) + SHAP + RFM ───────────────
        g1, g2 = st.columns([5, 5])

        with g1:
            st.markdown("## Fatores de Risco (SHAP)")
            explainer_local = shap.TreeExplainer(model)
            sv_local  = explainer_local(instance)
            shap_v    = sv_local.values[0]
            sorted_i  = np.argsort(np.abs(shap_v))[::-1][:10]
            feats_s   = [FEATURES[i] for i in sorted_i]
            vals_s    = shap_v[sorted_i]
            fig, ax   = dark_fig((6, 5))
            colors_sh = [PALETTE[4] if v>0 else PALETTE[0] for v in vals_s[::-1]]
            ax.barh(feats_s[::-1], vals_s[::-1], color=colors_sh, alpha=0.9, height=0.6)
            ax.axvline(0, color='#2d3748', linewidth=1)
            ax.set_xlabel('SHAP value', color='#475569', fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with g2:
            st.markdown("## Perfil RFM")
            fig, ax = dark_fig((6, 3))
            scores_rfm  = [r_score, f_score, m_score]
            labels_rfm  = ['Recency', 'Frequency', 'Monetary']
            colors_rfm  = [PALETTE[3] if v>=4 else PALETTE[1] if v>=3 else PALETTE[4] for v in scores_rfm]
            bars = ax.bar(labels_rfm, scores_rfm, color=colors_rfm, alpha=0.9, width=0.45)
            ax.set_ylim(0, 6)
            ax.set_ylabel('Score (1-5)', color='#475569', fontsize=10)
            for bar, v in zip(bars, scores_rfm):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.1,
                        f'{v}/5', ha='center', color='#f1f5f9', fontsize=13, fontweight='bold')
            ax.axhline(3, color='#2d3748', linewidth=1, linestyle='--', alpha=0.6)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("---")
            rfm_data = pd.DataFrame({
                'Métrica': ['RFM Total','Engagement','Segmento','Produtos'],
                'Valor':   [f'{rfm_s}/15', f'{eng:.3f}', rfm_seg, f'{products_v}']
            })
            st.dataframe(rfm_data, use_container_width=True, hide_index=True)

elif page == "🎯 Plano de Intervenção":

    st.title("Plano de Retenção por Segmento")
    st.markdown("<p style='color:#475569;font-size:15px;margin-top:-10px;'>Ações priorizadas por cluster · Baseadas nos principais drivers de churn</p>", unsafe_allow_html=True)
    st.markdown("---")

    intervention_map = {
        'Alto Risco': {
            'prioridade':'🔴 Crítica','cor':PALETTE[4],
            'acoes':[
                '🚨 Contato proativo via gerente de conta em até 48h',
                '💰 Cashback personalizado de 3-5% por 60 dias',
                '🔓 Isenção de tarifas por 3 meses',
                '📞 Pesquisa de satisfação urgente + resolução de pendências',
                '🎁 Oferta exclusiva de upgrade de conta',
            ],'triggers':['login_days_month','support_calls','last_txn_days_ago']
        },
        'Risco Moderado': {
            'prioridade':'🟠 Alta','cor':PALETTE[1],
            'acoes':[
                '📱 Campanha de reengajamento no app',
                '🎁 Oferta de produto complementar com benefício exclusivo',
                '💡 Tutorial de funcionalidades não utilizadas',
                '⭐ Convite para programa de fidelidade premium',
                '📊 Relatório financeiro personalizado mensal',
            ],'triggers':['engagement_score','monthly_txn','nps_score']
        },
        'Ocasional': {
            'prioridade':'🟡 Média','cor':PALETTE[2],
            'acoes':[
                '📊 E-mail mensal com resumo financeiro personalizado',
                '🏆 Missões gamificadas para aumentar engajamento',
                '🤝 Programa de indicação com benefício mútuo',
                '📈 Oferta de produto de investimento alinhado ao perfil',
                '🎯 Desafio mensal com recompensa em cashback',
            ],'triggers':['F_score','engagement_score','products']
        },
        'Engajado Ativo': {
            'prioridade':'🟢 Baixa','cor':PALETTE[3],
            'acoes':[
                '📣 Convite para grupo exclusivo de feedback de produto',
                '💎 Acesso antecipado a novos produtos',
                '🎯 Upgrades de limite por lealdade comprovada',
                '🌟 Programa de embaixadores com benefícios exclusivos',
            ],'triggers':['RFM_score','products','nps_score']
        },
        'VIP Fidelizado': {
            'prioridade':'🔵 Monitoramento','cor':PALETTE[0],
            'acoes':[
                '💎 Gerente dedicado com atendimento prioritário',
                '🌟 Programa VIP com benefícios personalizados',
                '🎁 Gifts e experiências exclusivas em datas especiais',
                '📈 Consultoria financeira personalizada trimestral',
            ],'triggers':['M_score','products','tenure_months']
        },
    }

    cluster_summary = df.groupby('cluster_name').agg(
        total        = ('customer_id','count'),
        churn_rate   = ('churned','mean'),
        criticos     = ('churn_risk', lambda x: (x=='Crítico').sum()),
        alto_risco   = ('churn_risk', lambda x: (x=='Alto').sum()),
        avg_balance  = ('avg_balance','mean'),
        receita_risco= ('M', lambda x: x[df.loc[x.index,'churn_risk'].isin(['Alto','Crítico'])].sum()),
    ).round(2)

    total_risco    = (df['churn_risk'].isin(['Alto','Crítico'])).sum()
    receita_total  = df[df['churn_risk'].isin(['Alto','Crítico'])]['M'].sum()

    # ── Linha 1: 3 KPIs + matriz de priorização (lado a lado) ─
    col_kpi, col_matrix = st.columns([4, 8])

    with col_kpi:
        st.markdown("## Visão Geral")
        st.metric("Em Risco (Alto+Crítico)", f"{total_risco:,}")
        st.metric("Receita Exposta", f"R$ {receita_total/1e6:.1f}M")
        st.metric("Segmentos Ativos", f"{df['cluster_name'].nunique()}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## Filtrar Segmento")
        seg_selecionado = st.selectbox("", ['Todos'] + sorted(df['cluster_name'].unique().tolist()))

    with col_matrix:
        st.markdown("## Matriz de Priorização")
        priority_order = {'🔴 Crítica':4,'🟠 Alta':3,'🟡 Média':2,'🟢 Baixa':1,'🔵 Monitoramento':0}
        priority_color_map = {
            '🔴 Crítica':PALETTE[4],'🟠 Alta':PALETTE[1],'🟡 Média':PALETTE[2],
            '🟢 Baixa':PALETTE[3],'🔵 Monitoramento':PALETTE[0]}

        fig, ax = dark_fig((9, 5))
        for seg in df['cluster_name'].unique():
            if seg not in intervention_map or seg not in cluster_summary.index: continue
            plan  = intervention_map[seg]
            stats = cluster_summary.loc[seg]
            size  = max(200, int(stats['receita_risco']/5000))
            ax.scatter(stats['churn_rate']*100,
                       int(stats['criticos'])+int(stats['alto_risco']),
                       color=priority_color_map[plan['prioridade']],
                       s=size, alpha=0.85, edgecolors='#2d3748', linewidths=1.5, zorder=5)
            ax.annotate(seg, (stats['churn_rate']*100,
                              int(stats['criticos'])+int(stats['alto_risco'])),
                        xytext=(8,6), textcoords='offset points',
                        color='#cbd5e1', fontsize=9)
        ax.set_xlabel('Churn Rate (%)', color='#475569')
        ax.set_ylabel('Clientes em Risco', color='#475569')
        ax.text(0.98, 0.04, '● tamanho = receita exposta', transform=ax.transAxes,
                ha='right', color='#475569', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Linha 2: cards de intervenção ────────────────────────
    st.markdown("## Planos de Ação")
    segs_to_show = df['cluster_name'].unique() if seg_selecionado == 'Todos' else [seg_selecionado]

    for seg in segs_to_show:
        if seg not in intervention_map: continue
        plan  = intervention_map[seg]
        stats = cluster_summary.loc[seg] if seg in cluster_summary.index else None

        with st.expander(f"{plan['prioridade']} — {seg}", expanded=(seg_selecionado != 'Todos')):
            if stats is not None:
                e1,e2,e3,e4 = st.columns(4)
                with e1: st.metric("Clientes",      f"{int(stats['total']):,}")
                with e2: st.metric("Churn Rate",    f"{stats['churn_rate']*100:.1f}%")
                with e3: st.metric("Críticos",      f"{int(stats['criticos']):,}")
                with e4: st.metric("Receita Risco", f"R$ {stats['receita_risco']/1e3:.0f}k")

            left, right = st.columns([5, 5])
            with left:
                st.markdown("**Triggers principais:**")
                st.markdown(" · ".join([f"`{t}`" for t in plan['triggers']]))
                st.markdown("<br>**Ações recomendadas:**", unsafe_allow_html=True)
                for acao in plan['acoes']:
                    st.markdown(f"<div class='action-box'>{acao}</div>", unsafe_allow_html=True)

            with right:
                if stats is not None:
                    fig, ax = dark_fig((5, 3.5))
                    risco_seg = df[df['cluster_name']==seg]['churn_risk'].value_counts().reindex(
                        ['Baixo','Médio','Alto','Crítico'], fill_value=0)
                    colors_rs = [PALETTE[3],PALETTE[1],PALETTE[2],PALETTE[4]]
                    ax.barh(risco_seg.index, risco_seg.values, color=colors_rs, alpha=0.9, height=0.5)
                    ax.set_xlabel('Clientes', color='#475569', fontsize=9)
                    ax.set_title('Distribuição de Risco', color='#cbd5e1', fontsize=10)
                    for i, v in enumerate(risco_seg.values):
                        if v > 0: ax.text(v+1, i, str(v), va='center', color='#cbd5e1', fontsize=9)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

