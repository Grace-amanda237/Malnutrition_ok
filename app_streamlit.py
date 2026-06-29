# app_streamlit.py
# Application Streamlit pour l'étude de la malnutrition au Cameroun
# Support Parquet

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
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Étude Malnutrition Cameroun",
    page_icon="📊",
    layout="wide"
)

# --- Style CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a5276;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1a5276;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a5276;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem 1rem;
        background-color: #eaf2f8;
        border-radius: 8px;
    }
    .insight-box {
        background-color: #fef9e7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f1c40f;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #d5f5e3;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #27ae60;
        margin: 1rem 0;
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #7f8c8d;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }
    .model-badge {
        display: inline-block;
        background-color: #1a5276;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Variables globales ---
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
    """Charge une table CSV"""
    # Chercher dans le dossier tables/
    paths = [
        os.path.join('tables', filename),
        filename
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except:
                continue
    return None

@st.cache_data
def load_json(filename):
    """Charge un fichier JSON"""
    paths = [
        os.path.join('tables', filename),
        filename
    ]
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
    """Charge un fichier texte"""
    paths = [
        os.path.join('tables', filename),
        filename
    ]
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
    """Charge un modèle pickle"""
    paths = [
        os.path.join('models', filename),
        filename
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except:
                continue
    return None

# --- Fonction de chargement des données ---
@st.cache_data
def load_and_preprocess_data():
    """Charge et prétraite les données DHS"""
    try:
        # Essayer de lire le fichier Parquet
        df = pd.read_parquet("CMIR71FL.parquet")
        df = df[TARGET_VARS]
        st.info("📂 Fichier Parquet chargé avec succès")
    except:
        try:
            # Essayer de lire le fichier CSV
            df = pd.read_csv("CMIR71FL.csv", usecols=TARGET_VARS, low_memory=False)
            st.info("📂 Fichier CSV chargé avec succès")
        except FileNotFoundError:
            st.warning("⚠️ Aucun fichier de données trouvé. Veuillez uploader le fichier.")
            uploaded_file = st.file_uploader("Uploader le fichier CMIR71FL.csv", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, usecols=TARGET_VARS, low_memory=False)
            else:
                return None
    
    # Renommage
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
    
    # Appliquer le renommage
    existing_rename = {k: v for k, v in RENAME.items() if k in df.columns}
    df = df.rename(columns=existing_rename)
    
    # Nettoyage des codes spéciaux
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and col not in ['CASEID', 'richesse_score', 'imc_dhs', 'poids_kg10', 'taille_cm10']:
            df[col] = df[col].replace([99, 999, 9998, 9999], np.nan)
    
    # Calcul IMC
    if 'imc_dhs' in df.columns:
        df['imc'] = df['imc_dhs'] / 100.0
    elif 'V445' in df.columns:
        df['imc'] = df['V445'] / 100.0
    else:
        df['imc'] = np.nan
    
    # Si l'IMC est manquant, le calculer à partir du poids et de la taille
    if 'poids_kg10' in df.columns and 'taille_cm10' in df.columns:
        df['poids_kg'] = df['poids_kg10'] / 10.0
        df['taille_m'] = df['taille_cm10'] / 1000.0
        mask = df['imc'].isna() & df['poids_kg'].between(25, 200) & df['taille_m'].between(1.2, 2.2)
        df.loc[mask, 'imc'] = df.loc[mask, 'poids_kg'] / df.loc[mask, 'taille_m']**2
    
    # Nettoyage IMC
    df.loc[~df['imc'].between(12, 60), 'imc'] = np.nan
    
    # Exclusion des femmes enceintes
    if 'enceinte' in df.columns:
        df = df[df['imc'].between(12, 60) & (df['enceinte'] != 1)].copy()
    else:
        df = df[df['imc'].between(12, 60)].copy()
    
    # Classification IMC
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
    
    # Imputation des variables manquantes
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

# --- Fonctions d'affichage ---
def display_table_with_insight(df, title, insight=None):
    """Affiche une table avec une interprétation"""
    st.markdown(f"### {title}")
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True)
        if insight:
            st.markdown(f"""
            <div class="insight-box">
                <strong>💡 Interprétation :</strong> {insight}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Table non disponible")

def display_model_info():
    """Affiche les informations sur les modèles disponibles"""
    st.markdown("""
    <div class="insight-box">
        <strong>🤖 Modèles disponibles :</strong><br>
        • <strong>best_model_compact.pkl</strong> — Random Forest optimisé (F1 macro: 0.534)<br>
        • <strong>ordinal_logit.pkl</strong> — Régression Logistique Ordinale (AIC: 38.2)<br>
        • <strong>scaler.pkl</strong> — StandardScaler pour normalisation des données
    </div>
    """, unsafe_allow_html=True)

# --- Interface principale ---
def main():
    # En-tête
    st.markdown('<div class="main-header">📊 Facteurs déterminants de la malnutrition liée au poids chez les adultes</div>', unsafe_allow_html=True)
    st.markdown('*Données DHS Cameroun 2018 — Analyse anthropométrique et socio-démographique*')
    
    # Chargement des données
    with st.spinner('Chargement des données...'):
        df = load_and_preprocess_data()
    
    if df is None:
        st.error("❌ Impossible de charger les données. Veuillez vérifier le fichier.")
        return
    
    st.success(f"✅ Données chargées : {len(df):,} femmes incluses dans l'analyse")
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    sections = [
        "🏠 Accueil",
        "📊 Analyse Descriptive",
        "📈 Analyse Bivariée",
        "🔬 Tests d'Hypothèses",
        "🎯 Analyse Multivariée",
        "🤖 Modèles ML",
        "📋 Prédiction"
    ]
    selected_section = st.sidebar.radio("Aller à :", sections)
    
    # --- SECTION: ACCUEIL ---
    if selected_section == "🏠 Accueil":
        st.markdown('<div class="section-header">🏠 Présentation de l\'Étude</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Contexte
            Cette étude analyse la malnutrition pondérale chez les femmes adultes (15-49 ans) 
            au Cameroun à partir des données de l'Enquête Démographique et de Santé (DHS) de 2018.
            
            **Objectifs principaux :**
            - Décrire la distribution de l'IMC
            - Identifier les facteurs socio-démographiques associés
            - Développer un modèle prédictif de la classe IMC
            """)
        with col2:
            st.markdown("### Indicateurs clés")
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("👩 Effectif", f"{len(df):,}")
                st.metric("📊 IMC moyen", f"{df['imc'].mean():.1f} kg/m²")
            with col2b:
                urbain = len(df[df['milieu']==1]) if 'milieu' in df.columns else 0
                rural = len(df[df['milieu']==2]) if 'milieu' in df.columns else 0
                st.metric("🏙️ Urbaines", f"{urbain:,}")
                st.metric("🌾 Rurales", f"{rural:,}")
        
        # Distribution des classes
        st.markdown("### Distribution des classes IMC")
        if 'imc_classe_label' in df.columns:
            class_counts = df['imc_classe_label'].value_counts()
            class_pcts = (class_counts / len(df) * 100).round(1)
            cols = st.columns(4)
            for i, (cls, count) in enumerate(class_counts.items()):
                if i < 4:
                    with cols[i]:
                        st.metric(label=cls, value=f"{count:,}", delta=f"{class_pcts[cls]:.1f}%")
        
        display_model_info()
        st.markdown("### Aperçu des données")
        cols_to_show = ['age', 'milieu', 'education', 'richesse', 'imc', 'imc_classe_label']
        available_cols = [c for c in cols_to_show if c in df.columns]
        st.dataframe(df[available_cols].head(10))
    
    # --- SECTION: ANALYSE DESCRIPTIVE ---
    elif selected_section == "📊 Analyse Descriptive":
        st.markdown('<div class="section-header">📊 Analyse Descriptive</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Statistiques IMC", "📈 Distribution", "📋 Tables"])
        
        with tab1:
            # Table 01: Descriptives IMC
            desc = load_table('01_desc_imc.csv')
            if desc is None or desc.empty:
                # Créer les données à partir du DataFrame
                if 'imc' in df.columns:
                    desc_data = df['imc'].describe()
                    desc = pd.DataFrame({
                        'Statistique': desc_data.index,
                        'Valeur': desc_data.values
                    })
            display_table_with_insight(
                desc,
                "Statistiques descriptives de l'IMC",
                "L'IMC moyen est de 24.6 kg/m² (écart-type 5.0). La médiane (23.6) est proche de la moyenne, "
                "mais le maximum (56.4) et le minimum (12.7) montrent une variabilité importante."
            )
            
            # Table 02: IMC par milieu
            milieu = load_table('02_imc_par_milieu.csv')
            display_table_with_insight(
                milieu,
                "IMC par milieu de résidence",
                "Les femmes urbaines ont un IMC moyen (25.6) significativement plus élevé que les rurales (23.4)."
            )
            
            # Table 03: IMC par éducation
            edu = load_table('03_imc_par_education.csv')
            display_table_with_insight(
                edu,
                "IMC par niveau d'éducation",
                "L'IMC augmente avec le niveau d'éducation, passant de 23.3 (Aucun) à 26.0 (Supérieur)."
            )
            
            # Table 04: IMC par richesse
            rich = load_table('04_imc_par_richesses.csv')
            display_table_with_insight(
                rich,
                "IMC par quintile de richesse",
                "Gradient socio-économique très net : l'IMC passe de 22.6 (plus pauvre) à 26.3 (plus riche)."
            )
        
        with tab2:
            st.markdown("### Distribution de l'IMC")
            try:
                st.image('figures/fig1_hist_imc.png', use_container_width=True)
                st.caption("Distribution de l'IMC avec seuils OMS")
            except:
                st.warning("Image fig1_hist_imc.png non trouvée")
            
            try:
                st.image('figures/fig5_dist_classes.png', use_container_width=True)
                st.caption("Répartition des classes IMC")
            except:
                st.warning("Image fig5_dist_classes.png non trouvée")
        
        with tab3:
            st.markdown("### Toutes les tables descriptives")
            for f in ['01_desc_imc.csv', '02_imc_par_milieu.csv', '03_imc_par_education.csv', '04_imc_par_richesses.csv']:
                data = load_table(f)
                if data is not None:
                    st.markdown(f"#### {f.replace('.csv', '').replace('_', ' ').title()}")
                    st.dataframe(data)
                    st.divider()
    
    # --- SECTION: ANALYSE BIVARIÉE ---
    elif selected_section == "📈 Analyse Bivariée":
        st.markdown('<div class="section-header">📈 Analyse Bivariée</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📊 Visualisations", "📋 Tables"])
        
        with tab1:
            st.markdown("### Boxplots par variables clés")
            try:
                st.image('figures/fig2_boxplots.png', use_container_width=True)
                st.caption("Boxplots de l'IMC selon le milieu, l'éducation, la richesse et le tabagisme")
            except:
                st.warning("Image fig2_boxplots.png non trouvée")
            
            st.markdown("### Relation entre l'âge et l'IMC")
            try:
                st.image('figures/fig3_scatter_age_imc.png', use_container_width=True)
                st.caption("Nuage de points avec tendance linéaire")
            except:
                st.warning("Image fig3_scatter_age_imc.png non trouvée")
            
            st.markdown("### Matrice de corrélation (Spearman)")
            try:
                st.image('figures/fig4_heatmap_corr.png', use_container_width=True)
                st.caption("Matrice de corrélation de Spearman entre variables numériques")
            except:
                st.warning("Image fig4_heatmap_corr.png non trouvée")
        
        with tab2:
            st.markdown("### Tables bivariées")
            st.markdown("""
            **Principales corrélations avec l'IMC :**
            - **Âge** : +0.30 (corrélation positive modérée)
            - **Richesse** : +0.27 (gradient socio-économique)
            - **Richesse Score** : +0.33
            - **Regarde TV** : +0.22 (sédentarité)
            - **Éducation** : +0.20
            """)
    
    # --- SECTION: TESTS D'HYPOTHÈSES ---
    elif selected_section == "🔬 Tests d'Hypothèses":
        st.markdown('<div class="section-header">🔬 Tests d\'Hypothèses</div>', unsafe_allow_html=True)
        
        tests_data = load_json('05_tests_hypotheses.json')
        
        if tests_data:
            st.markdown("### Synthèse des tests statistiques")
            
            test_df = pd.DataFrame([
                {
                    'Test': k.replace('_', ' ').title(),
                    'Statistique': f"{v.get('statistic', v.get('correlation', '—')):.4f}",
                    'p-value': f"{v.get('p_value', 0):.2e}",
                    'Significatif': v.get('p_value', 1) < 0.05
                }
                for k, v in tests_data.items() if 'p_value' in v
            ])
            
            st.dataframe(test_df, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
                <strong>💡 Interprétation :</strong><br>
                • <strong>Shapiro-Wilk</strong> : Rejet de la normalité (p < 0.001)<br>
                • <strong>t-test Urbain/Rural</strong> : Différence très significative (p < 0.001)<br>
                • <strong>ANOVA Richesse</strong> : Effet très significatif (p < 0.001)<br>
                • <strong>ANOVA Éducation</strong> : Effet très significatif (p < 0.001)<br>
                • <strong>Corrélation Âge-IMC</strong> : Positive et significative (r ≈ 0.30)<br>
                • <strong>Chi² IMC×Milieu</strong> : Association très significative
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("📊 Détail des tests"):
            if tests_data:
                for key, value in tests_data.items():
                    st.markdown(f"**{key.replace('_', ' ').title()}**")
                    if 'statistic' in value:
                        st.write(f"- Statistique : {value['statistic']:.4f}")
                    if 'correlation' in value:
                        st.write(f"- Corrélation : {value['correlation']:.4f}")
                    if 'p_value' in value:
                        st.write(f"- p-value : {value['p_value']:.2e}")
                    st.divider()
    
    # --- SECTION: ANALYSE MULTIVARIÉE ---
    elif selected_section == "🎯 Analyse Multivariée":
        st.markdown('<div class="section-header">🎯 Analyse Multivariée</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["🔍 ACP & Clusters", "📊 Régressions", "📋 Tables"])
        
        with tab1:
            # Table 07: PCA variance
            pca_var = load_table('07_pca_variance.csv')
            display_table_with_insight(
                pca_var,
                "Variance expliquée par l'ACP",
                "Les 8 premières composantes expliquent 86% de la variance totale. "
                "Les deux premiers axes (CP1: 35.6%, CP2: 13.4%) résument les dimensions socio-économique et démographique."
            )
            
            # Table 11: K-means profils
            kmeans_profils = load_table('11_kmeans_profils.csv')
            display_table_with_insight(
                kmeans_profils,
                "Profils des clusters K-Means (k=4)",
                "Quatre profils distincts : jeunes urbaines éduquées (Cluster 0), adultes mixtes (Cluster 1), "
                "rurales peu éduquées (Cluster 2), urbaines aisées (Cluster 3)."
            )
            
            st.markdown("### Visualisations ACP et Clusters")
            for fig in ['fig6_acp.png', 'fig7_kmeans_choix.png', 'fig8_clusters_acp.png']:
                try:
                    st.image(f'figures/{fig}', use_container_width=True)
                except:
                    st.warning(f"Image {fig} non trouvée")
        
        with tab2:
            # Table 08: PLS coefficients
            pls_coefs = load_table('08_pls_coefs.csv')
            display_table_with_insight(
                pls_coefs,
                "Coefficients PLS",
                "La PLS confirme l'importance de l'âge, de la richesse et des médias dans la prédiction de l'IMC."
            )
            
            # Table 09b: OLS coefficients
            ols_coefs = load_table('09b_ols_coefs.csv')
            display_table_with_insight(
                ols_coefs,
                "Coefficients de régression linéaire multiple",
                "Âge (+0.13), richesse (+0.78) et regarde TV (+0.51) augmentent l'IMC. "
                "Lit journal (−0.53) et milieu rural (−0.31) le diminuent."
            )
            
            # Table 10: Logit odds ratios
            logit_or = load_table('10_logit_odds_ratios.csv')
            display_table_with_insight(
                logit_or,
                "Odds Ratios - Régression Logistique",
                "L'âge, la richesse et l'éducation augmentent les chances d'être dans une catégorie d'IMC plus élevée. "
                "Le milieu rural réduit ces chances."
            )
        
        with tab3:
            st.markdown("### Tables de l'analyse multivariée")
            table_files = ['06_vif.csv', '07_pca_variance.csv', '08_pls_coefs.csv', 
                          '09b_ols_coefs.csv', '10_logit_odds_ratios.csv', '11_kmeans_profils.csv']
            for f in table_files:
                data = load_table(f)
                if data is not None:
                    st.markdown(f"#### {f.replace('.csv', '').replace('_', ' ').title()}")
                    st.dataframe(data)
                    st.divider()
    
    # --- SECTION: MODÈLES ML ---
    elif selected_section == "🤖 Modèles ML":
        st.markdown('<div class="section-header">🤖 Modèles de Machine Learning</div>', unsafe_allow_html=True)
        
        display_model_info()
        
        tab1, tab2, tab3 = st.tabs(["📊 Performance", "📈 Importance", "📋 Tables"])
        
        with tab1:
            # Table 14: Classification results
            class_results = load_table('14_classification_results.csv')
            display_table_with_insight(
                class_results,
                "Comparaison des modèles de classification",
                "XGBoost (F1=0.519) et Random Forest (F1=0.515) sont les meilleurs modèles. "
                "Le Random Forest optimisé atteint F1=0.534."
            )
            
            # Table 15: Confusion matrix
            conf_matrix = load_table('15_confusion_matrix.csv')
            display_table_with_insight(
                conf_matrix,
                "Matrice de confusion - Random Forest optimisé",
                "La classe 'Normal' est la mieux prédite (74% de recall). "
                "La classe 'Maigreur' reste difficile à prédire en raison de sa rareté."
            )
            
            # Table 16: Classification report
            class_report = load_text('16_classification_report.txt')
            if class_report:
                st.markdown("#### Rapport de classification")
                st.code(class_report, language='text')
            
            st.markdown("### Visualisations")
            for fig in ['fig9_confusion.png', 'fig11_models_comparison.png']:
                try:
                    st.image(f'figures/{fig}', use_container_width=True)
                except:
                    st.warning(f"Image {fig} non trouvée")
        
        with tab2:
            # Table 17: Feature importance
            feat_importance = load_table('17_feature_importance.csv')
            display_table_with_insight(
                feat_importance,
                "Importance des variables - Random Forest",
                "L'âge est le prédicteur le plus important (13.6%), suivi du score de richesse (10.5%) "
                "et des années d'études (9.4%)."
            )
            
            try:
                st.image('figures/fig10_feature_importance.png', use_container_width=True)
                st.caption("Importance des variables - Random Forest vs XGBoost")
            except:
                st.warning("Image fig10_feature_importance.png non trouvée")
        
        with tab3:
            st.markdown("### Tables des modèles ML")
            table_files = ['12_regressions_continues.csv', '14_classification_results.csv', 
                          '15_confusion_matrix.csv', '17_feature_importance.csv']
            for f in table_files:
                data = load_table(f)
                if data is not None:
                    st.markdown(f"#### {f.replace('.csv', '').replace('_', ' ').title()}")
                    st.dataframe(data)
                    st.divider()
    
    # --- SECTION: PRÉDICTION ---
    elif selected_section == "📋 Prédiction":
        st.markdown('<div class="section-header">📋 Prédiction de la Classe IMC</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
            <strong>🤖 Modèle utilisé :</strong> Random Forest optimisé 
            <span class="model-badge">F1: 0.534</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Entrez les caractéristiques de la patiente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Âge (années)", 15, 49, 30)
            milieu = st.selectbox("Milieu de résidence", ["Urbain", "Rural"])
            education = st.selectbox("Niveau d'éducation", ["Aucun", "Primaire", "Secondaire", "Supérieur"])
            annees_etudes = st.slider("Années d'études", 0, 20, 10)
        
        with col2:
            richesse = st.slider("Quintile de richesse", 1, 5, 3)
            enfants = st.slider("Nombre d'enfants", 0, 10, 2)
            taille_menage = st.slider("Taille du ménage", 1, 20, 5)
            travail = st.selectbox("Travaille", ["Oui", "Non"])
        
        milieu_enc = 1 if milieu == "Urbain" else 2
        edu_map = {"Aucun": 0, "Primaire": 1, "Secondaire": 2, "Supérieur": 3}
        education_enc = edu_map[education]
        travail_enc = 1 if travail == "Oui" else 0
        
        model = load_model('best_model_compact.pkl')
        scaler = load_model('scaler.pkl')
        
        if st.button("🔮 Prédire la classe IMC", type="primary"):
            if model is not None:
                with st.spinner("Prédiction en cours..."):
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
                    proba = max(probas) * 100
                    
                    colors = {'Maigreur': '#e74c3c', 'Normal': '#2ecc71', 'Surpoids': '#f1c40f', 'Obésité': '#e67e22'}
                    
                    st.markdown(f"""
                    <div class="result-box" style="border-left-color: {colors.get(class_name, '#3498db')};">
                        <h2 style="color: {colors.get(class_name, '#3498db')};">
                            🏷️ Classe IMC prédite : <strong>{class_name}</strong>
                        </h2>
                        <p style="font-size: 1.2rem; color: #2c3e50;">Confiance : {proba:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(classes, probas, color=['#e74c3c', '#2ecc71', '#f1c40f', '#e67e22'])
                    ax.set_ylabel('Probabilité')
                    ax.set_title('Probabilités par classe')
                    ax.set_ylim(0, 1)
                    for i, v in enumerate(probas):
                        ax.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontsize=10)
                    st.pyplot(fig)
                    plt.close()
            else:
                st.error("❌ Modèle non disponible. Veuillez vérifier le fichier 'best_model_compact.pkl'.")
    
    # --- Pied de page ---
    st.markdown("""
    <div class="footer">
        <p>📊 Étude de la Malnutrition au Cameroun — Données DHS 2018</p>
        <p style="font-size: 0.8rem;">Développé avec Streamlit • Analyse statistique et Machine Learning</p>
        <p style="font-size: 0.7rem; color: #95a5a6;">
            Modèles : Random Forest (F1: 0.534) • Régression Logistique Ordinale (AIC: 38.2)
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()