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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Base ─────────────────────────────────────────── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #0a1628 50%, #0d0d1a 100%);
        background-attachment: fixed;
    }

    /* ── Sidebar ──────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
        border-right: 1px solid rgba(99,179,237,0.15);
    }

    /* ── Tipografia ───────────────────────────────────── */
    h1 {
        background: linear-gradient(90deg, #63b3ed, #9f7aea, #68d391);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    h2, h3 { color: #e2e8f0 !important; font-weight: 600 !important; }
    p, label, div { color: #a0aec0; }

    /* ── Metric Cards ─────────────────────────────────── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(99,179,237,0.08) 0%, rgba(159,122,234,0.08) 100%);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 24px rgba(99,179,237,0.06);
        transition: all 0.3s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(99,179,237,0.5);
        box-shadow: 0 8px 32px rgba(99,179,237,0.15);
        transform: translateY(-2px);
    }
    [data-testid="metric-container"] label {
        color: #718096 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        background: linear-gradient(90deg, #63b3ed, #9f7aea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* ── Botões ───────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0, #553c9a);
        color: white !important;
        border: 1px solid rgba(99,179,237,0.3) !important;
        border-radius: 10px;
        padding: 10px 28px;
        font-weight: 600;
        font-size: 14px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(43,108,176,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99,179,237,0.35);
    }

    /* ── Sliders ──────────────────────────────────────── */
    .stSlider > div > div > div { background: linear-gradient(90deg, #63b3ed, #9f7aea); }

    /* ── Tabs ─────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(13,17,23,0.8);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 12px;
        padding: 5px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #718096;
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99,179,237,0.2), rgba(159,122,234,0.2)) !important;
        color: #63b3ed !important;
        border: 1px solid rgba(99,179,237,0.3) !important;
    }

    /* ── Expanders ────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(99,179,237,0.06), rgba(159,122,234,0.06)) !important;
        border: 1px solid rgba(99,179,237,0.2) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background: rgba(13,17,23,0.6) !important;
        border: 1px solid rgba(99,179,237,0.15) !important;
        border-top: none !important;
    }

    /* ── Selectbox ────────────────────────────────────── */
    .stSelectbox > div > div {
        background: rgba(13,17,23,0.8) !important;
        border: 1px solid rgba(99,179,237,0.2) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }

    /* ── Divisores ────────────────────────────────────── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), transparent);
        margin: 1.5rem 0;
    }

    /* ── Badges ───────────────────────────────────────── */
    .badge-high {
        background: linear-gradient(135deg, #742a2a, #c53030);
        color: #fed7d7; padding: 7px 18px; border-radius: 20px;
        font-weight: 700; font-size: 14px; display: inline-block;
        border: 1px solid rgba(252,129,129,0.3);
        box-shadow: 0 4px 15px rgba(197,48,48,0.25);
    }
    .badge-med {
        background: linear-gradient(135deg, #7b341e, #c05621);
        color: #feebc8; padding: 7px 18px; border-radius: 20px;
        font-weight: 700; font-size: 14px; display: inline-block;
        border: 1px solid rgba(237,137,54,0.3);
        box-shadow: 0 4px 15px rgba(192,86,33,0.25);
    }
    .badge-low {
        background: linear-gradient(135deg, #1c4532, #276749);
        color: #c6f6d5; padding: 7px 18px; border-radius: 20px;
        font-weight: 700; font-size: 14px; display: inline-block;
        border: 1px solid rgba(72,187,120,0.3);
        box-shadow: 0 4px 15px rgba(39,103,73,0.25);
    }

    /* ── Cards customizados ───────────────────────────── */
    .info-box {
        background: linear-gradient(135deg, rgba(99,179,237,0.06), rgba(159,122,234,0.06));
        border-left: 3px solid #63b3ed;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px; margin: 8px 0;
    }
    .action-box {
        background: linear-gradient(135deg, rgba(104,211,145,0.06), rgba(99,179,237,0.04));
        border-left: 3px solid #68d391;
        border-radius: 0 8px 8px 0;
        padding: 11px 16px; margin: 6px 0; font-size: 14px;
        color: #c6f6d5 !important;
    }

    /* ── Dataframe ────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid rgba(99,179,237,0.2) !important;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

PALETTE = ['#63b3ed','#9f7aea','#68d391','#f6ad55','#fc8181','#76e4f7']

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
    fig, ax = plt.subplots(figsize=figsize, facecolor='#0a0c14')
    ax.set_facecolor('#111827')
    ax.tick_params(colors='#9aa0a6')
    for spine in ax.spines.values(): spine.set_edgecolor('#1a2234')
    return fig, ax

def dark_figs(nrows, ncols, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='#0a0c14')
    for ax in np.array(axes).flatten():
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#718096')
        for spine in ax.spines.values(): spine.set_edgecolor('rgba(99,179,237,0.1)')
    return fig, axes

# ── PAGE 1: Visão Geral ─────────────────────────────────────
if page == "🏠 Visão Geral":
    st.title("📉 Análise Preditiva de Churn — Fintech")
    st.markdown("Identificação de padrões não-lineares de churn e intervenções segmentadas para retenção de clientes.")
    st.markdown("---")

    total     = len(df)
    churned   = df['churned'].sum()
    churn_rt  = df['churned'].mean()*100
    criticos  = (df['churn_risk']=='Crítico').sum()
    alto      = (df['churn_risk']=='Alto').sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("Total Clientes", f"{total:,}")
    with c2: st.metric("Churn Rate", f"{churn_rt:.1f}%")
    with c3: st.metric("Churned", f"{churned:,}")
    with c4: st.metric("Risco Crítico", f"{criticos:,}")
    with c5: st.metric("Risco Alto", f"{alto:,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Churn Rate por Perfil")
        fig, ax = dark_fig((8,5))
        cp = df.groupby('profile')['churned'].mean().sort_values(ascending=False)*100
        colors_p = [PALETTE[5] if v>20 else PALETTE[0] for v in cp.values]
        ax.bar(cp.index, cp.values, color=colors_p, alpha=0.85)
        ax.axhline(churn_rt, color='rgba(255,255,255,0.4)', linestyle='--', alpha=0.5, label=f'Média {churn_rt:.1f}%')
        ax.set_ylabel('%', color='#718096')
        ax.legend(facecolor='#0d1117', labelcolor='#e2e8f0')
        for i,v in enumerate(cp.values):
            ax.text(i, v+0.3, f'{v:.1f}%', ha='center', color='#e2e8f0', fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("### Distribuição de Risco")
        fig, ax = dark_fig((8,5))
        rc = df['churn_risk'].value_counts().reindex(['Baixo','Médio','Alto','Crítico'])
        colors_r = [PALETTE[3],PALETTE[4],PALETTE[1],PALETTE[5]]
        ax.bar(rc.index, rc.values, color=colors_r, alpha=0.85)
        ax.set_ylabel('Clientes', color='#718096')
        for i,v in enumerate(rc.values):
            ax.text(i, v+30, f'{v:,}', ha='center', color='#e2e8f0', fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Churn Rate por Nº de Produtos")
        fig, ax = dark_fig((8,4))
        cp2 = df.groupby('products')['churned'].mean()*100
        ax.plot(cp2.index, cp2.values, marker='o', color=PALETTE[0], linewidth=2.5, markersize=8)
        ax.fill_between(cp2.index, cp2.values, alpha=0.15, color=PALETTE[0])
        ax.set_xlabel('Produtos', color='#718096')
        ax.set_ylabel('%', color='#718096')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown("### Top Features de Churn (SHAP)")
        fig, ax = dark_fig((8,4))
        top10 = shap_imp.head(10)
        ax.barh(top10['feature'][::-1], top10['mean_shap'][::-1],
                color=PALETTE[0], alpha=0.85)
        ax.set_xlabel('Mean |SHAP value|', color='#718096')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ── PAGE 2: RFM e Clusters ──────────────────────────────────
elif page == "🔵 Análise RFM e Clusters":
    st.title("🔵 Análise RFM e Segmentação de Clusters")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Segmentos RFM")
        fig, ax = dark_fig((8,5))
        seg_counts = df['RFM_segment'].value_counts()
        seg_churn  = df.groupby('RFM_segment')['churned'].mean()*100
        colors_s   = [PALETTE[5] if seg_churn.get(s,0)>20 else PALETTE[0] for s in seg_counts.index]
        ax.bar(seg_counts.index, seg_counts.values, color=colors_s, alpha=0.85)
        ax.set_ylabel('Clientes', color='#718096')
        ax2 = ax.twinx()
        ax2.plot(seg_counts.index, [seg_churn.get(s,0) for s in seg_counts.index],
                 marker='D', color=PALETTE[5], linewidth=2, markersize=8, label='Churn%')
        ax2.set_ylabel('Churn Rate (%)', color=PALETTE[5])
        ax2.tick_params(colors=PALETTE[5])
        ax2.legend(facecolor='#0d1117', labelcolor='#e2e8f0')
        for spine in ax2.spines.values(): spine.set_edgecolor('#1a2234')
        ax2.set_facecolor('#111827')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("### Distribuição RFM Score")
        fig, ax = dark_fig((8,5))
        for churn_val, color, name in [(0,PALETTE[0],'Ativo'),(1,PALETTE[5],'Churn')]:
            vals = df[df['churned']==churn_val]['RFM_score']
            ax.hist(vals, bins=20, alpha=0.6, color=color, label=name, density=True)
        ax.set_xlabel('RFM Score', color='#718096')
        ax.legend(facecolor='#0d1117', labelcolor='#e2e8f0')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("### Perfil dos Clusters")

    cluster_stats = df.groupby('cluster_name').agg(
        Clientes       = ('customer_id','count'),
        Churn_Rate     = ('churned','mean'),
        Avg_RFM        = ('RFM_score','mean'),
        Avg_Produtos   = ('products','mean'),
        Avg_Saldo      = ('avg_balance','mean'),
        Avg_Engagement = ('engagement_score','mean'),
        Avg_NPS        = ('nps_score','mean'),
    ).round(2)
    cluster_stats['Churn_Rate'] = (cluster_stats['Churn_Rate']*100).round(1).astype(str)+'%'
    st.dataframe(cluster_stats.style.background_gradient(
        subset=['Avg_RFM','Avg_Produtos','Avg_Engagement'], cmap='Blues'),
        use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Churn Rate por Cluster")
        fig, ax = dark_fig((8,5))
        cc = df.groupby('cluster_name')['churned'].mean().sort_values(ascending=False)*100
        colors_c = [PALETTE[5] if v>25 else PALETTE[4] if v>12 else PALETTE[0] for v in cc.values]
        ax.barh(cc.index, cc.values, color=colors_c, alpha=0.85)
        ax.set_xlabel('%', color='#718096')
        for i,v in enumerate(cc.values):
            ax.text(v+0.2, i, f'{v:.1f}%', va='center', color='#e2e8f0', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown("### Scatter: Engagement vs RFM")
        fig, ax = dark_fig((8,5))
        cluster_ids = df['cluster'].unique()
        for i, cid in enumerate(sorted(cluster_ids)):
            mask = df['cluster']==cid
            ax.scatter(df[mask]['engagement_score'], df[mask]['RFM_score'],
                      color=PALETTE[i%len(PALETTE)], alpha=0.3, s=8,
                      label=cluster_names.get(cid, f'C{cid}'))
        ax.set_xlabel('Engagement Score', color='#718096')
        ax.set_ylabel('RFM Score', color='#718096')
        ax.legend(facecolor='#0d1117', labelcolor='#e2e8f0', fontsize=8, markerscale=3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ── PAGE 3: Score Individual ─────────────────────────────────
elif page == "🔴 Score Individual":
    st.title("🔴 Score de Churn — Cliente Individual")
    st.markdown("Preencha os dados do cliente para calcular o risco de churn em tempo real.")
    st.markdown("---")

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.markdown("### 📝 Dados do Cliente")
        age_v        = st.slider("Idade", 18, 65, 35)
        tenure_v     = st.slider("Tempo de Casa (meses)", 1, 72, 12)
        products_v   = st.slider("Nº de Produtos", 1, 9, 2)
        monthly_txn_v= st.slider("Transações/Mês", 0, 60, 10)
        login_days_v = st.slider("Dias de Login/Mês", 0, 30, 10)
        last_txn_v   = st.slider("Dias desde última transação", 0, 120, 15)
        support_v    = st.slider("Chamadas de Suporte (último mês)", 0, 10, 1)
        nps_v        = st.slider("NPS Score", 0, 10, 7)
        balance_v    = st.number_input("Saldo Médio (R$)", 0, 200000, 5000, 500)
        cc_spend_v   = st.number_input("Gasto Cartão/Mês (R$)", 0, 50000, 1500, 100)
        pix_v        = st.slider("Transações PIX/Mês", 0, 40, 5)
        inv_v        = st.checkbox("Possui Investimento?")
        ins_v        = st.checkbox("Possui Seguro?")
        loan_v       = st.checkbox("Possui Empréstimo?")

    with col_out:
        # Calcular features derivadas
        def calc_r_score(v):
            if v <= 5:   return 5
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
            login_days_v/30*0.3 +
            min(monthly_txn_v/60,1)*0.3 +
            products_v/9*0.2 +
            max(0, 1-last_txn_v/120)*0.2, 4)
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

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Prob. Churn", f"{prob*100:.1f}%")
        with c2: st.metric("Score Retenção", f"{score}")
        with c3: st.metric("Risco", risk)
        with c4: st.metric("Segmento RFM", rfm_seg)

        badge_cls = 'badge-high' if risk in ['Crítico','Alto'] else 'badge-med' if risk=='Médio' else 'badge-low'
        st.markdown(f"<div class='{badge_cls}'>{'🔴 ALTO RISCO DE CHURN' if risk in ['Crítico','Alto'] else '🟡 RISCO MODERADO' if risk=='Médio' else '🟢 BAIXO RISCO'}</div>", unsafe_allow_html=True)

        st.markdown("---")
        tab1, tab2 = st.tabs(["📊 SHAP — Fatores de Risco", "📋 Perfil RFM"])

        with tab1:
            explainer_local = shap.TreeExplainer(model)
            sv_local = explainer_local(instance)
            shap_v   = sv_local.values[0]
            sorted_i = np.argsort(np.abs(shap_v))[::-1][:10]
            feats_s  = [FEATURES[i] for i in sorted_i]
            vals_s   = shap_v[sorted_i]
            fig, ax  = dark_fig((9,5))
            ax.barh(feats_s[::-1], vals_s[::-1],
                    color=['#f87171' if v>0 else '#60a5fa' for v in vals_s[::-1]], alpha=0.85)
            ax.axvline(0, color='rgba(255,255,255,0.4)', linewidth=0.8, alpha=0.5)
            ax.set_title('Fatores que contribuem para o risco', color='#e2e8f0')
            ax.set_xlabel('SHAP value (+ aumenta risco)', color='#718096')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with tab2:
            rfm_data = {
                'Métrica': ['Recency Score','Frequency Score','Monetary Score','RFM Total','Engagement','Segmento'],
                'Valor':   [f'{r_score}/5', f'{f_score}/5', f'{m_score}/5', f'{rfm_s}/15', f'{eng:.3f}', rfm_seg]
            }
            st.dataframe(pd.DataFrame(rfm_data), use_container_width=True)

            fig, ax = dark_fig((7,4))
            scores  = [r_score, f_score, m_score]
            labels  = ['Recency\n(5=recente)', 'Frequency\n(5=frequente)', 'Monetary\n(5=alto valor)']
            colors_rfm = [PALETTE[3] if v>=4 else PALETTE[4] if v>=3 else PALETTE[5] for v in scores]
            ax.bar(labels, scores, color=colors_rfm, alpha=0.85)
            ax.set_ylim(0, 5.5)
            ax.set_ylabel('Score', color='#718096')
            ax.set_title('Scores RFM do Cliente', color='#e2e8f0')
            for i,v in enumerate(scores):
                ax.text(i, v+0.1, f'{v}/5', ha='center', color='#e2e8f0', fontsize=12, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ── PAGE 4: Plano de Intervenção ────────────────────────────
elif page == "🎯 Plano de Intervenção":
    st.title("🎯 Plano de Intervenção por Segmento")
    st.markdown("Ações de retenção priorizadas por cluster, baseadas nos principais drivers de churn.")
    st.markdown("---")

    intervention_map = {
        'Alto Risco': {
            'prioridade':'🔴 Crítica','cor':PALETTE[5],
            'acoes':[
                '🚨 Contato proativo via gerente de conta em até 48h',
                '💰 Cashback personalizado de 3-5% por 60 dias',
                '🔓 Isenção de tarifas por 3 meses',
                '📞 Pesquisa de satisfação urgente + resolução de pendências',
                '🎁 Oferta exclusiva de upgrade de conta',
            ],'triggers':['login_days_month','support_calls','last_txn_days_ago']
        },
        'Risco Moderado': {
            'prioridade':'🟠 Alta','cor':PALETTE[4],
            'acoes':[
                '📱 Campanha de reengajamento no app',
                '🎁 Oferta de produto complementar com benefício exclusivo',
                '💡 Tutorial de funcionalidades não utilizadas',
                '⭐ Convite para programa de fidelidade premium',
                '📊 Relatório financeiro personalizado mensal',
            ],'triggers':['engagement_score','monthly_txn','nps_score']
        },
        'Ocasional': {
            'prioridade':'🟡 Média','cor':PALETTE[1],
            'acoes':[
                '📊 E-mail mensal com resumo financeiro personalizado',
                '🏆 Missões gamificadas para aumentar engajamento',
                '🤝 Programa de indicação: benefício para cliente e indicado',
                '📈 Oferta de produto de investimento alinhado ao perfil',
                '🎯 Desafio mensal com recompensa em cashback',
            ],'triggers':['F_score','engagement_score','products']
        },
        'Engajado Ativo': {
            'prioridade':'🟢 Baixa','cor':PALETTE[3],
            'acoes':[
                '📣 Convite para grupo exclusivo de feedback de produto',
                '💎 Acesso antecipado a novos produtos e funcionalidades',
                '🎯 Upgrades de limite por lealdade comprovada',
                '🌟 Programa de embaixadores com benefícios exclusivos',
            ],'triggers':['RFM_score','products','nps_score']
        },
        'VIP Fidelizado': {
            'prioridade':'🔵 Monitoramento','cor':PALETTE[0],
            'acoes':[
                '💎 Gerente dedicado com atendimento prioritário',
                '🌟 Programa VIP com benefícios exclusivos e personalizados',
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

    # Métricas gerais
    total_risco = (df['churn_risk'].isin(['Alto','Crítico'])).sum()
    receita_total = df[df['churn_risk'].isin(['Alto','Crítico'])]['M'].sum()

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Clientes em Risco (Alto+Crítico)", f"{total_risco:,}")
    with c2: st.metric("Receita em Risco (R$)", f"{receita_total/1e6:.1f}M")
    with c3: st.metric("Segmentos Monitorados", f"{df['cluster_name'].nunique()}")

    st.markdown("---")

    # Filtro de segmento
    seg_selecionado = st.selectbox("Filtrar por segmento:", ['Todos'] + sorted(df['cluster_name'].unique().tolist()))

    segs_to_show = df['cluster_name'].unique() if seg_selecionado == 'Todos' else [seg_selecionado]

    for seg in segs_to_show:
        if seg not in intervention_map:
            continue
        plan  = intervention_map[seg]
        stats = cluster_summary.loc[seg] if seg in cluster_summary.index else None

        with st.expander(f"{plan['prioridade']} — {seg}", expanded=(seg_selecionado != 'Todos')):
            if stats is not None:
                c1,c2,c3,c4 = st.columns(4)
                with c1: st.metric("Total Clientes", f"{int(stats['total']):,}")
                with c2: st.metric("Churn Rate", f"{stats['churn_rate']*100:.1f}%")
                with c3: st.metric("Críticos", f"{int(stats['criticos']):,}")
                with c4: st.metric("Receita em Risco", f"R$ {stats['receita_risco']/1e3:.0f}k")

            st.markdown("**🔍 Principais triggers de churn:**")
            st.markdown(" · ".join([f"`{t}`" for t in plan['triggers']]))

            st.markdown("**✅ Ações recomendadas:**")
            for acao in plan['acoes']:
                st.markdown(f"<div class='action-box'>{acao}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Matriz de Priorização")
    fig, ax = dark_fig((12,6))
    priority_order = {'🔴 Crítica':4,'🟠 Alta':3,'🟡 Média':2,'🟢 Baixa':1,'🔵 Monitoramento':0}
    priority_color_map = {'🔴 Crítica':PALETTE[5],'🟠 Alta':PALETTE[4],'🟡 Média':PALETTE[1],
                          '🟢 Baixa':PALETTE[3],'🔵 Monitoramento':PALETTE[0]}

    for seg in segs_to_show:
        if seg not in intervention_map or seg not in cluster_summary.index: continue
        plan  = intervention_map[seg]
        stats = cluster_summary.loc[seg]
        ax.scatter(stats['churn_rate']*100,
                   int(stats['criticos'])+int(stats['alto_risco']),
                   color=priority_color_map[plan['prioridade']],
                   s=400, alpha=0.85, edgecolors='white', linewidths=1.5,
                   zorder=5)
        ax.annotate(seg, (stats['churn_rate']*100,
                          int(stats['criticos'])+int(stats['alto_risco'])),
                    xytext=(8,4), textcoords='offset points',
                    color='#e2e8f0', fontsize=9)

    ax.set_xlabel('Churn Rate (%)', color='#718096')
    ax.set_ylabel('Clientes em Risco (Alto + Crítico)', color='#718096')
    ax.set_title('Priorização: Churn Rate vs Volume em Risco', color='#e2e8f0')
    plt.tight_layout(); st.pyplot(fig); plt.close()
