
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Étude Malnutrition lié au poids chez les adultes au Cameroun",
    page_icon="",
    layout="wide"
)

# Variables globales
TARGET_VARS = [
    'CASEID', 'V012', 'V013', 'V025', 'V024', 'V106', 'V107', 'V133',
    'V130', 'V131', 'V190', 'V191', 'V501', 'V502', 'V437', 'V438',
    'V445', 'V463A', 'V463AA', 'V201', 'V213', 'V714', 'V745A',
    'V157', 'V158', 'V159', 'V136', 'V137', 'V155', 'V717',
    'V113', 'V116'
]

# --- Fonctions de chargement ---
@st.cache_data
def load_table(filename):
    paths = [os.path.join('tables', filename), filename]
    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except:
                continue
    return None

@st.cache_data
def load_json(filename):
    paths = [os.path.join('tables', filename), filename]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                continue
    return None

@st.cache_data
def load_text(filename):
    paths = [os.path.join('tables', filename), filename]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                continue
    return None

@st.cache_resource
def load_model(filename):
    paths = [os.path.join('models', filename), filename]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except:
                continue
    return None

# --- Chargement des données ---
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_parquet("CMIR71FL.parquet")
        df = df[TARGET_VARS]
    except:
        try:
            df = pd.read_csv("CMIR71FL.csv", usecols=TARGET_VARS, low_memory=False)
        except FileNotFoundError:
            return None
    
    RENAME = {
        'V012': 'age', 'V013': 'age_groupe', 'V024': 'region',
        'V025': 'milieu', 'V106': 'education', 'V107': 'annees_etudes',
        'V133': 'annees_education_unique', 'V130': 'religion',
        'V131': 'ethnie', 'V190': 'richesse', 'V191': 'richesse_score',
        'V501': 'statut_marital', 'V502': 'statut_union',
        'V437': 'poids_kg10', 'V438': 'taille_cm10', 'V445': 'imc_dhs',
        'V463A': 'fume_cigarette', 'V463AA': 'fume_actuel',
        'V201': 'enfants_total', 'V213': 'enceinte', 'V714': 'travaille',
        'V745A': 'possede_maison', 'V157': 'lit_journal',
        'V158': 'ecoute_radio', 'V159': 'regarde_tv',
        'V136': 'taille_menage', 'V137': 'enfants_moins_5ans',
        'V155': 'alphabetisation', 'V717': 'occupation_groupe'
    }
    
    existing_rename = {k: v for k, v in RENAME.items() if k in df.columns}
    df = df.rename(columns=existing_rename)
    
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and col not in ['CASEID', 'richesse_score', 'imc_dhs', 'poids_kg10', 'taille_cm10']:
            df[col] = df[col].replace([99, 999, 9998, 9999], np.nan)
    
    if 'imc_dhs' in df.columns:
        df['imc'] = df['imc_dhs'] / 100.0
    else:
        df['imc'] = np.nan
    
    if 'poids_kg10' in df.columns and 'taille_cm10' in df.columns:
        df['poids_kg'] = df['poids_kg10'] / 10.0
        df['taille_m'] = df['taille_cm10'] / 1000.0
        mask = df['imc'].isna() & df['poids_kg'].between(25, 200) & df['taille_m'].between(1.2, 2.2)
        df.loc[mask, 'imc'] = df.loc[mask, 'poids_kg'] / df.loc[mask, 'taille_m']**2
    
    df.loc[~df['imc'].between(12, 60), 'imc'] = np.nan
    
    if 'enceinte' in df.columns:
        df = df[df['imc'].between(12, 60) & (df['enceinte'] != 1)].copy()
    else:
        df = df[df['imc'].between(12, 60)].copy()
    
    def classe(imc):
        if imc < 18.5:
            return 0
        elif imc < 25:
            return 1
        elif imc < 30:
            return 2
        else:
            return 3
    
    df['imc_classe'] = df['imc'].apply(classe)
    df['imc_classe_label'] = df['imc_classe'].map({
        0: 'Maigreur', 1: 'Normal', 2: 'Surpoids', 3: 'Obésité'
    })
    
    FEATURES = [
        'age', 'milieu', 'education', 'annees_education_unique',
        'richesse', 'richesse_score', 'fume_actuel', 'enfants_total',
        'travaille', 'taille_menage', 'lit_journal', 'ecoute_radio',
        'regarde_tv', 'alphabetisation', 'religion', 'possede_maison',
        'occupation_groupe', 'statut_union'
    ]
    
    for col in FEATURES:
        if col in df.columns:
            if df[col].dtype == 'float64' and df[col].nunique() > 6:
                df[col] = df[col].fillna(df[col].median())
            else:
                if len(df[col].mode()) > 0:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
    
    return df

# --- Fonctions d'affichage simples ---
def show_insight(text):
    """Affiche une interprétation sans HTML complexe"""
    st.info(f" **Interprétation :** {text}")

def show_table(df, title, insight=None):
    """Affiche une table avec interprétation"""
    if df is not None and not df.empty:
        st.subheader(title)
        st.dataframe(df, use_container_width=True)
        if insight:
            show_insight(insight)
        return True
    return False

def show_image(filename, caption=""):
    """Affiche une image"""
    try:
        st.image(f'figures/{filename}', use_container_width=True)
        if caption:
            st.caption(caption)
        return True
    except:
        return False

# --- Interface principale ---
def main():
    st.title(" Étude de la Malnutrition au Cameroun")
    st.markdown("*Données DHS Cameroun 2018*")
    
    # Chargement
    with st.spinner('Chargement des données...'):
        df = load_and_preprocess_data()
    
    if df is None:
        st.error(" Fichier de données non trouvé.")
        return
    
    st.success(f" {len(df):,} femmes incluses dans l'analyse")
    
    # Sidebar
    st.sidebar.title("Navigation")
    sections = [
        "Accueil",
        "Descriptive",
        "Bivariée",
        "Tests",
        "Multivariée",
        "ML",
        "Prédiction"
    ]
    section = st.sidebar.radio("Aller à :", sections)
    
    # --- ACCUEIL ---
    if section == "Accueil":
        st.header(" Présentation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Contexte**")
            st.write("""
            Analyse de la malnutrition pondérale chez les femmes adultes (15-49 ans) 
            au Cameroun à partir des données DHS 2018.
            """)
        with col2:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Effectif", f"{len(df):,}")
                st.metric("IMC moyen", f"{df['imc'].mean():.1f}")
            with c2:
                urb = len(df[df['milieu']==1]) if 'milieu' in df.columns else 0
                rur = len(df[df['milieu']==2]) if 'milieu' in df.columns else 0
                st.metric("Urbaines", f"{urb:,}")
                st.metric("Rurales", f"{rur:,}")
        
        st.subheader("Distribution des classes IMC")
        if 'imc_classe_label' in df.columns:
            counts = df['imc_classe_label'].value_counts()
            pcts = (counts / len(df) * 100).round(1)
            cols = st.columns(4)
            for i, (cls, cnt) in enumerate(counts.items()):
                if i < 4:
                    with cols[i]:
                        st.metric(cls, f"{cnt:,}", f"{pcts[cls]:.1f}%")
        
        st.subheader("Aperçu des données")
        cols = ['age', 'milieu', 'education', 'richesse', 'imc', 'imc_classe_label']
        available = [c for c in cols if c in df.columns]
        st.dataframe(df[available].head(10))
    
    # --- DESCRIPTIVE ---
    elif section == "Descriptive":
        st.header(" Analyse Descriptive")
        
        tab1, tab2 = st.tabs(["Statistiques", "Distribution"])
        
        with tab1:
            desc = load_table('01_desc_imc.csv')
            show_table(desc, "Statistiques descriptives de l'IMC",
                      "IMC moyen: 24.6 kg/m², écart-type: 5.0")
            
            milieu = load_table('02_imc_par_milieu.csv')
            show_table(milieu, "IMC par milieu",
                      "Urbain: 25.6 vs Rural: 23.4 kg/m²")
            
            edu = load_table('03_imc_par_education.csv')
            show_table(edu, "IMC par éducation",
                      "Augmentation avec le niveau d'éducation")
            
            rich = load_table('04_imc_par_richesses.csv')
            show_table(rich, "IMC par richesse",
                      "Gradient socio-économique très net")
        
        with tab2:
            st.subheader("Distribution de l'IMC")
            if not show_image('fig1_hist_imc.png'):
                st.warning("Image non trouvée")
            
            st.subheader("Classes IMC")
            if not show_image('fig5_dist_classes.png'):
                st.warning("Image non trouvée")
    
    # --- BIVARIÉE ---
    elif section == "Bivariée":
        st.header(" Analyse Bivariée")
        
        tab1, tab2 = st.tabs(["Visualisations", "Corrélations"])
        
        with tab1:
            for img in ['fig2_boxplots.png', 'fig3_scatter_age_imc.png', 'fig4_heatmap_corr.png']:
                if not show_image(img):
                    st.warning(f"Image {img} non trouvée")
        
        with tab2:
            st.subheader("Corrélations avec l'IMC")
            corr_data = [
                {'Variable': 'Âge', 'Corrélation': '+0.30'},
                {'Variable': 'Richesse Score', 'Corrélation': '+0.33'},
                {'Variable': 'Quintile de richesse', 'Corrélation': '+0.27'},
                {'Variable': 'Regarde TV', 'Corrélation': '+0.22'},
                {'Variable': 'Éducation', 'Corrélation': '+0.20'}
            ]
            st.dataframe(pd.DataFrame(corr_data), use_container_width=True)
    
    # --- TESTS ---
    elif section == "Tests":
        st.header(" Tests d'Hypothèses")
        
        tests_data = load_json('05_tests_hypotheses.json')
        
        if tests_data:
            def fmt(v):
                if v is None:
                    return "—"
                if isinstance(v, (int, float)):
                    return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
                return str(v)
            
            test_df = pd.DataFrame([
                {
                    'Test': k.replace('_', ' ').title(),
                    'Statistique': fmt(v.get('statistic', v.get('correlation', '—'))),
                    'p-value': fmt(v.get('p_value', '—')),
                    'Significatif': v.get('p_value', 1) < 0.05 if isinstance(v.get('p_value'), (int, float)) else False
                }
                for k, v in tests_data.items() if 'p_value' in v
            ])
            
            st.dataframe(test_df, use_container_width=True)
            
            show_insight(
                "Tous les tests sont significatifs (p < 0.001). "
                "L'IMC n'est pas normal. Les différences urbain/rural, "
                "richesses et éducation sont très significatives."
            )
        
        with st.expander("Détail des tests"):
            if tests_data:
                for k, v in tests_data.items():
                    st.write(f"**{k.replace('_', ' ').title()}**")
                    if 'statistic' in v:
                        st.write(f"- Statistique : {v['statistic']}")
                    if 'correlation' in v:
                        st.write(f"- Corrélation : {v['correlation']}")
                    if 'p_value' in v:
                        st.write(f"- p-value : {v['p_value']:.2e}" if isinstance(v['p_value'], (int, float)) else f"- p-value : {v['p_value']}")
                    st.divider()
    
    # --- MULTIVARIÉE ---
    elif section == "Multivariée":
        st.header("🎯 Analyse Multivariée")
        
        tab1, tab2, tab3 = st.tabs(["ACP & Clusters", "Régressions", "Tables"])
        
        with tab1:
            pca_var = load_table('07_pca_variance.csv')
            show_table(pca_var, "Variance expliquée par l'ACP",
                      "86% de variance expliquée par 8 composantes")
            
            kmeans_profils = load_table('11_kmeans_profils.csv')
            show_table(kmeans_profils, "Profils des clusters (k=4)",
                      "4 profils distincts identifiés")
            
            for img in ['fig6_acp.png', 'fig7_kmeans_choix.png', 'fig8_clusters_acp.png']:
                if not show_image(img):
                    st.warning(f"Image {img} non trouvée")
        
        with tab2:
            pls = load_table('08_pls_coefs.csv')
            show_table(pls, "Coefficients PLS")
            
            ols = load_table('09b_ols_coefs.csv')
            show_table(ols, "Coefficients OLS")
            
            logit = load_table('10_logit_odds_ratios.csv')
            show_table(logit, "Odds Ratios")
        
        with tab3:
            st.subheader("Tables multivariées")
            for f in ['06_vif.csv', '07_pca_variance.csv', '08_pls_coefs.csv', 
                      '09b_ols_coefs.csv', '10_logit_odds_ratios.csv', '11_kmeans_profils.csv']:
                data = load_table(f)
                if data is not None:
                    st.write(f"**{f.replace('.csv', '').replace('_', ' ').title()}**")
                    st.dataframe(data)
                    st.divider()
    
    # --- ML ---
    elif section == "ML":
        st.header(" Modèles ML")
        
        tab1, tab2 = st.tabs(["Performance", "Tables"])
        
        with tab1:
            results = load_table('14_classification_results.csv')
            show_table(results, "Comparaison des modèles",
                      "XGBoost et Random Forest sont les meilleurs")
            
            conf = load_table('15_confusion_matrix.csv')
            show_table(conf, "Matrice de confusion")
            
            report = load_text('16_classification_report.txt')
            if report:
                st.code(report, language='text')
            
            for img in ['fig9_confusion.png', 'fig11_models_comparison.png']:
                if not show_image(img):
                    st.warning(f"Image {img} non trouvée")
        
        with tab2:
            st.subheader("Tables ML")
            for f in ['12_regressions_continues.csv', '14_classification_results.csv', 
                      '15_confusion_matrix.csv', '17_feature_importance.csv']:
                data = load_table(f)
                if data is not None:
                    st.write(f"**{f.replace('.csv', '').replace('_', ' ').title()}**")
                    st.dataframe(data)
                    st.divider()
    
    # --- PRÉDICTION ---
    elif section == "Prédiction":
        st.header(" Prédiction de la Classe IMC")
        
        st.info(" Modèle : Random Forest optimisé (F1: 0.534)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Âge", 15, 49, 30)
            milieu = st.selectbox("Milieu", ["Urbain", "Rural"])
            education = st.selectbox("Éducation", ["Aucun", "Primaire", "Secondaire", "Supérieur"])
            annees_etudes = st.slider("Années d'études", 0, 20, 10)
        
        with col2:
            richesse = st.slider("Quintile de richesse", 1, 5, 3)
            enfants = st.slider("Nombre d'enfants", 0, 10, 2)
            taille_menage = st.slider("Taille du ménage", 1, 20, 5)
        
        milieu_enc = 1 if milieu == "Urbain" else 2
        edu_map = {"Aucun": 0, "Primaire": 1, "Secondaire": 2, "Supérieur": 3}
        education_enc = edu_map[education]
        
        model = load_model('best_model_compact.pkl')
        scaler = load_model('scaler.pkl')
        
        if st.button("Prédire", type="primary"):
            if model is not None:
                with st.spinner("Prédiction..."):
                    features = ['age', 'milieu', 'education', 'richesse_score', 'annees_education_unique', 'taille_menage']
                    X_pred = pd.DataFrame([[age, milieu_enc, education_enc, 3, annees_etudes, taille_menage]], 
                                          columns=features)
                    
                    if scaler is not None:
                        X_pred_scaled = scaler.transform(X_pred)
                    else:
                        X_pred_scaled = X_pred.values
                    
                    pred = model.predict(X_pred_scaled)
                    classes = ['Maigreur', 'Normal', 'Surpoids', 'Obésité']
                    class_name = classes[pred[0]]
                    probas = model.predict_proba(X_pred_scaled)[0]
                    
                    st.success(f" **Classe IMC prédite : {class_name}**")
                    st.write(f"Confiance : {max(probas)*100:.1f}%")
                    
                    # Graphique des probabilités
                    fig, ax = plt.subplots(figsize=(6, 3))
                    colors = ['#e74c3c', '#2ecc71', '#f1c40f', '#e67e22']
                    ax.bar(classes, probas, color=colors)
                    ax.set_ylabel('Probabilité')
                    ax.set_ylim(0, 1)
                    for i, v in enumerate(probas):
                        ax.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontsize=9)
                    st.pyplot(fig)
                    plt.close()
            else:
                st.error(" Modèle non disponible")
    
    # Pied de page
    st.markdown("---")
    st.caption(" Étude Malnutrition lié au poids Cameroun — Données DHS 2018. s donnees (taille, imc et poids) des hommes adultes etant indisponibles, notre etude est centrée sur les femmes adultes")

if __name__ == "__main__":
    main()