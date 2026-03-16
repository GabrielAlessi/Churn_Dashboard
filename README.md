[README_CHURN.md](https://github.com/user-attachments/files/26032350/README_CHURN.md)
# 📉 Análise Preditiva de Churn — Fintech / Banco Digital

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-fbbf24?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explicabilidade-818cf8?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-34d399?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-60a5fa?style=for-the-badge)

> **Predictive churn analytics for digital banking | Identifies non-linear churn patterns via RFM + ML | Segmented retention interventions per cluster | ROC-AUC 0.92**

---

## 📌 Visão Geral

Reter clientes em um banco digital é mais barato do que adquirir novos — mas exige identificar quem está prestes a sair e por quê. Este projeto constrói um pipeline completo de análise preditiva de churn: desde a segmentação RFM e clusterização até a modelagem preditiva com XGBoost e a geração automática de planos de intervenção segmentados, tudo disponível em uma aplicação interativa em produção.

### Resultados

| Métrica | Valor |
|:--------|------:|
| **ROC-AUC (XGBoost)** | **0.92** |
| **Clientes analisados** | **15.000** |
| **Segmentos identificados** | **5 clusters** |
| **Features engineered** | **21 (RFM + comportamentais)** |
| **Intervenções geradas** | **4-5 por segmento** |

---

## 🎯 Problema de Negócio

**Contexto:** Um banco digital com 15.000 clientes ativos precisa identificar proativamente quem está em risco de cancelar a conta — e agir de forma personalizada antes que o churn aconteça.

**Desafios endereçados:**
- Padrões de churn são **não-lineares**: ter 3+ produtos reduz o risco de forma desproporcional
- Clientes diferentes exigem **intervenções diferentes** — oferta de cashback não funciona para um cliente VIP
- Time de CRM precisa de **priorização clara** — quem contatar primeiro e com qual ação
- Custo de retenção deve ser proporcional ao **valor do cliente em risco**

---

## 🖥️ Aplicação em Produção

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churndashboard-jlene4ghqevkrsxshcjpa9.streamlit.app/)

🔗 **[Acessar a aplicação](https://churndashboard-jlene4ghqevkrsxshcjpa9.streamlit.app/)**

As análises deste projeto estão disponíveis em uma aplicação interativa com 4 páginas:

| Página | Conteúdo |
|:-------|:---------|
| 🏠 **Visão Geral** | Churn rate, distribuição de risco e top features SHAP |
| 🔵 **Análise RFM e Clusters** | Segmentos RFM, perfil dos clusters e scatter de engajamento |
| 🔴 **Score Individual** | Sliders interativos → score de churn em tempo real + SHAP |
| 🎯 **Plano de Intervenção** | Ações de retenção priorizadas por segmento + matriz de priorização |

---

## 🗂️ Estrutura do Projeto

```
churn-analysis/
│
├── 📓 notebooks/
│   └── churn_analysis.ipynb          # Notebook principal completo
│
├── churn_dashboard.py                # Aplicação interativa em produção
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Pipeline Completo

### Etapa 1 — Engenharia de Features RFM
- **Recency (R):** dias desde a última transação → score 1-5
- **Frequency (F):** transações mensais → score 1-5
- **Monetary (M):** saldo médio + gasto no cartão → score 1-5
- **RFM Score:** combinação dos três (3-15) → segmentos: Champions, Loyal, Potential, At Risk, Lost
- **Features derivadas:** `engagement_score`, `product_depth`, `balance_trend`

### Etapa 2 — Segmentação com K-Means
- Método do cotovelo + Silhouette Score para k ótimo
- 5 clusters com nomeação automática por comportamento
- Visualizações: scatter engagement vs RFM, heatmap de risco, perfil comparativo

### Etapa 3 — Modelagem Preditiva
- 4 modelos comparados: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- XGBoost vencedor com ROC-AUC 0.92 e PR-AUC 0.78
- 21 features incluindo scores RFM + variáveis comportamentais

### Etapa 4 — Explicabilidade com SHAP
- Importância global (Mean |SHAP value|)
- Waterfall individual para alto risco vs baixo risco
- Identificação dos principais triggers por segmento

### Etapa 5 — Intervenções Segmentadas
- Plano de ação automático por cluster com 4-5 ações específicas
- Priorização por churn rate + volume de clientes em risco + receita exposta
- Matriz de priorização visual: churn rate vs clientes em risco

---

## 🔍 Padrões Não-Lineares Identificados

| Padrão | Insight |
|:-------|:--------|
| **Efeito multiplicador de produtos** | 3+ produtos reduz churn de forma desproporcional, não linear |
| **Combinação investimento + seguro** | Clientes com ambos têm churn 60% menor — "produto âncora" |
| **NPS + suporte** | Combinação de NPS baixo E múltiplas chamadas eleva risco de forma super-aditiva (+35%) |
| **Tenure curto ≠ alto risco** | Novos clientes engajados têm churn menor que antigos inativos |
| **Login abaixo de 5 dias/mês** | Preditor mais forte de churn — 3x mais risco que a média |

---

## 🎯 Plano de Intervenção por Segmento

| Segmento | Prioridade | Ação Principal |
|:---------|:----------:|:---------------|
| Alto Risco | 🔴 Crítica | Contato via gerente em 48h + cashback de 3-5% |
| Risco Moderado | 🟠 Alta | Reengajamento no app + produto complementar |
| Ocasional | 🟡 Média | Gamificação + e-mail financeiro personalizado |
| Engajado Ativo | 🟢 Baixa | Programa de indicação + produto de investimento |
| VIP Fidelizado | 🔵 Monitoramento | Consultoria financeira + benefícios exclusivos |

---

## ⚙️ Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/GabrielAlessi/churn-analysis.git
cd churn-analysis
```

### 2. Crie o ambiente virtual
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute o notebook
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

### 5. Rode a aplicação localmente
```bash
streamlit run churn_dashboard.py
```

---

## 📦 Dependências

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
shap>=0.42.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.28.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

---

## 🚀 Próximos Passos

- [ ] **Score em tempo real** — pipeline diário de atualização do churn score via dados transacionais
- [ ] **A/B Test das intervenções** — medir eficácia real de cada ação por segmento
- [ ] **Survival Analysis** — modelar *quando* o cliente vai fazer churn, não só *se*
- [ ] **CLV (Customer Lifetime Value)** — priorizar retenção por valor, não só por risco
- [ ] **Integração com CRM** — API REST `/churn-score` para uso direto pelo time comercial

---

## 👨‍💻 Autor

**Gabriel Alessi Naumann**  
Cientista de Dados

[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabriel--alessi--naumann-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/gabriel-alessi-naumann/)
[![GitHub](https://img.shields.io/badge/GitHub-GabrielAlessi-181717?style=flat&logo=github)](https://github.com/GabrielAlessi)
[![Kaggle](https://img.shields.io/badge/Kaggle-gabrielalessinaumann-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/gabrielalessinaumann)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

*⭐ Se este projeto foi útil para você, considere deixar uma estrela no repositório!*
