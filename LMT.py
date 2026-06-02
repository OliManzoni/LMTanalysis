import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
import io
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page / Page configuration
st.set_page_config(
    page_title="LMT Phenotype Analytics",
    page_icon="🧠",
    layout="wide"
)

# Style CSS / CSS Styling
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .main .block-container { padding-top: 2rem; }
    h1 { color: #1e3d59; }
    h2 { color: #17b978; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# DICTIONNAIRE DE TRADUCTION / TRANSLATION DICTIONARY
# ==============================================================================
lang = st.sidebar.radio("🌐 Language / Langue", ["Français", "English"])

text = {
    "Français": {
        "title": "🧠 LMT Phenotype Analytics : Pipeline de Plasticité Comportementale",
        "subtitle": "Interface Graphique d'Analyse des Trajectoires du Live Mouse Tracker.",
        "rationale_exp": "📖 Afficher le Rationnel Mathématique & l'Architecture des Domaines",
        "rationale_md": """
### 1. Logique Mathématique et Statistique
* **Résidualisation Hiérarchique Globale (LMM) :** Calcule la variance maternelle sur *l'ensemble* de la cohorte expérimentale pour maximiser la puissance statistique avant isolation des groupes.
* **Z-Score Ancré sur la Nouveauté :** Normalisation calée sur le groupe témoin ($Z=0$).
* **Tests de Welch Point-par-Point :** Gère l'hétéroscédasticité sans assumer l'égalité des variances.

### 2. Architecture des 5 Domaines Composites
* **Locomotor Drive :** `MoveisolatedTotalLen + totalDistance`
* **Exploratory Spatial Strategy :** `Center_Periphery_Ratio + RearisolatedTotalLen`
* **Vigilance / Risk Assessment :** `SAPNb + StopisolatedNb`
* **Social Tolerance :** `Group2TotalLen + Group3TotalLen + SidebysideContactTotalLen`
* **Active Social Engagement :** `Oral-oral + Oral-genital + Socialapproach + Train2 + FollowZone`
        """,
        "import_h": "📁 Importation des Données",
        "upload": "Glissez un fichier (.txt ou .csv) ici",
        "load_ok": "Données chargées avec succès !",
        "wait_file": "En attente d'un fichier .txt ou .csv.",
        "param_h": "⚙️ Paramètres du Génotype",
        "sep": "Séparateur de la colonne 'genotype'",
        "err_parse": "Erreur de parsing de la colonne 'genotype'",
        "group_h": "🔬 Sélection des Groupes pour Visualisation",
        "ctrl": "Groupe CONTRÔLE (Référence Z=0)",
        "target": "Groupe TRAITÉ / MUTANT",
        "phase": "Chronologie des phases (Axe X)",
        "plot_h": "🎨 Personnalisation des Figures",
        "w": "Largeur de la figure",
        "h": "Hauteur de la figure",
        "star": "Taille des étoiles / p-values",
        "color": "Couleur du groupe traité",
        "zoom": "Activer le zoom manuel de l'axe Y",
        "run": "🚀 Lancer l'Analyse Computationnelle Globale",
        "running": "Exécution du pipeline statistique LMM global en cours...",
        "res_h": "📊 Trajectoires Phénotypiques Longitudinales",
        "ylab": "Adjusted Z-Score\n(Litter-Corrected)",
        "exp_h": "💾 Exportation des Résultats",
        "dl_fig": "🖼️ Télécharger la Figure (Qualité Publication)",
        "dl_csv": "📊 Télécharger les Données Calculées (CSV)",
        "exp_ok": "Analyses prêtes pour exportation !"
    },
    "English": {
        "title": "🧠 LMT Phenotype Analytics: Behavioral Plasticity Pipeline",
        "subtitle": "Graphical Interface for Live Mouse Tracker Trajectory Analysis.",
        "rationale_exp": "📖 Show Mathematical Rationale & Domain Architecture",
        "rationale_md": """
### 1. Mathematical and Statistical Logic
* **Global Hierarchical Residualization (LMM):** Computes maternal variance across the *entire* experimental cohort to maximize statistical power prior to group isolation.
* **Novelty-Anchored Z-Score:** Normalization anchored to the control group ($Z=0$).
* **Point-by-Point Welch's Tests:** Handles heteroscedasticity without assuming equal variances.

### 2. 5-Domain Composite Architecture
* **Locomotor Drive:** `MoveisolatedTotalLen + totalDistance`
* **Exploratory Spatial Strategy:** `Center_Periphery_Ratio + RearisolatedTotalLen`
* **Vigilance / Risk Assessment:** `SAPNb + StopisolatedNb`
* **Social Tolerance:** `Group2TotalLen + Group3TotalLen + SidebysideContactTotalLen`
* **Active Social Engagement:** `Oral-oral + Oral-genital + Socialapproach + Train2 + FollowZone`
        """,
        "import_h": "📁 Data Import",
        "upload": "Drop a data file (.txt or .csv) here",
        "load_ok": "Data successfully loaded!",
        "wait_file": "Waiting for a .txt or .csv file.",
        "param_h": "⚙️ Genotype Parameters",
        "sep": "Separator for 'genotype' column",
        "err_parse": "Error parsing 'genotype' column",
        "group_h": "🔬 Group Selection for Visualization",
        "ctrl": "CONTROL Group (Reference Z=0)",
        "target": "TREATED / MUTANT Group",
        "phase": "Chronology of phases (X-Axis)",
        "plot_h": "🎨 Figure Customization",
        "w": "Figure width",
        "h": "Figure height",
        "star": "Star / p-values size",
        "color": "Treated group color",
        "zoom": "Enable manual Y-axis zoom",
        "run": "🚀 Run Global Computational Analysis",
        "running": "Executing global LMM statistical pipeline...",
        "res_h": "📊 Longitudinal Phenotypic Trajectories",
        "ylab": "Adjusted Z-Score\n(Litter-Corrected)",
        "exp_h": "💾 Export Results",
        "dl_fig": "🖼️ Download Figure (Publication Quality)",
        "dl_csv": "📊 Download Calculated Data (CSV)",
        "exp_ok": "Analyses ready for export!"
    }
}

t = text[lang]

st.title(t["title"])
st.write(t["subtitle"])

with st.expander(t["rationale_exp"], expanded=False):
    st.markdown(t["rationale_md"])

# ==============================================================================
# BARRE LATÉRALE - CONFIGURATION & IMPORTEUR DE FICHIERS
# ==============================================================================
st.sidebar.header(t["import_h"])
fichier_uploade = st.sidebar.file_uploader(t["upload"], type=["txt", "csv"])

@st.cache_data
def charger_donnees(fichier):
    content = fichier.read()
    try:
        df_temp = pd.read_csv(io.BytesIO(content), sep='\t')
        if 'genotype' in df_temp.columns and len(df_temp.columns) > 5:
            return df_temp
        df_temp = pd.read_csv(io.BytesIO(content), sep=',')
        if 'genotype' in df_temp.columns and len(df_temp.columns) > 5:
            return df_temp
        df_temp = pd.read_csv(io.BytesIO(content), sep=';')
        if 'genotype' in df_temp.columns and len(df_temp.columns) > 5:
            return df_temp
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        return None

df_raw = None
if fichier_uploade is not None:
    df_raw = charger_donnees(fichier_uploade)
    if df_raw is not None:
        st.sidebar.success(t["load_ok"])
else:
    st.sidebar.warning(t["wait_file"])

if df_raw is not None:
    df = df_raw.copy()
    
    st.sidebar.header(t["param_h"])
    separateur = st.sidebar.text_input(t["sep"], "_")
    
    try:
        if 'genotype' in df.columns:
            df['Sex'] = df['genotype'].str.split(separateur).str[0]
            df['Treatment'] = df['genotype'].str.split(separateur).str[1]
        else:
            st.sidebar.error("La colonne 'genotype' est introuvable.")
            st.stop()
    except Exception as e:
        st.sidebar.error(f"{t['err_parse']}: {e}")
        st.stop()

    groupes_detectes = sorted(df['Treatment'].dropna().unique().tolist())
    sexes_detectes = sorted(df['Sex'].dropna().unique().tolist())
    phases_detectees = sorted(df['day'].dropna().unique().tolist())

    st.sidebar.subheader(t["group_h"])
    idx_sham = groupes_detectes.index('SHAM') if 'SHAM' in groupes_detectes else 0
    groupe_controle = st.sidebar.selectbox(t["ctrl"], groupes_detectes, index=idx_sham)
    groupe_cible = st.sidebar.selectbox(t["target"], [g for g in groupes_detectes if g != groupe_controle], index=0)
    
    phases_par_defaut = [p for p in ['Exploration', 'Night 1', 'Night 2', 'Night 3'] if p in phases_detectees]
    phases_ordre = st.sidebar.multiselect(t["phase"], phases_detectees, default=phases_par_defaut)

    st.sidebar.header(t["plot_h"])
    largeur_fig = st.sidebar.slider(t["w"], 10, 24, 16)
    hauteur_fig = st.sidebar.slider(t["h"], 15, 45, 28)
    taille_etoiles = st.sidebar.slider(t["star"], 12, 30, 18)
    couleur_cible = st.sidebar.color_picker(t["color"], "#d62728")
    
    activer_zoom_y = st.sidebar.checkbox(t["zoom"])
    y_min, y_max = -3.0, 3.0
    if activer_zoom_y:
        col_z1, col_z2 = st.sidebar.columns(2)
        y_min = col_z1.number_input("Y Min", value=-3.0, step=0.5)
        y_max = col_z2.number_input("Y Max", value=3.0, step=0.5)

    # ==============================================================================
    # PIPELINE DE TRAITEMENT (MODÈLE GLOBAL)
    # ==============================================================================
    df['Center_Periphery_Ratio'] = df['CenterZoneTotalLen'] / (df['PeripheryZoneTotalLen'] + 1e-5)
    
    metriques_domaines = {
        'Locomotor_Drive': ['MoveisolatedTotalLen', 'totalDistance'],
        'Exploratory_Spatial_Strategy': ['Center_Periphery_Ratio', 'RearisolatedTotalLen'],
        'Vigilance_Risk': ['SAPNb', 'StopisolatedNb'],
        'Social_Tolerance': ['Group2TotalLen', 'Group3TotalLen', 'SidebysideContactTotalLen', 'SidebysideContact,oppositewayTotalLen'],
        'Active_Social_Engagement': ['Oral-oralContactTotalLen', 'Oral-genitalContactTotalLen', 'SocialapproachNb', 'Train2TotalLen', 'FollowZoneTotalLen']
    }

    for domaine in metriques_domaines:
        metriques_domaines[domaine] = [m for m in metriques_domaines[domaine] if m in df.columns]

    toutes_metriques = [m for sous_liste in metriques_domaines.values() for m in sous_liste]
    
    # Nettoyage global avant LMM
    df_calcul = df.dropna(subset=['day']).copy()
    for m in toutes_metriques:
        df_calcul[m] = df_calcul[m].fillna(0)
    df_calcul['day'] = pd.Categorical(df_calcul['day'], categories=phases_ordre, ordered=True)

    if st.button(t["run"], type="primary"):
        with st.spinner(t["running"]):
            
            # 1. Modèle Mixte Linéaire (LMM) sur la totalité des données
            for m in toutes_metriques:
                df_calcul[f'{m}_adj'] = df_calcul[m]
                try:
                    modele = smf.mixedlm(f"Q('{m}') ~ Treatment", df_calcul, groups=df_calcul["group"])
                    resultat = modele.fit(method='cg')
                    effets_aleatoires = resultat.random_effects
                    df_calcul[f'{m}_adj'] = df_calcul.apply(
                        lambda ligne: ligne[m] - (effets_aleatoires[ligne['group']]['Group'] if ligne['group'] in effets_aleatoires else 0), 
                        axis=1
                    )
                except:
                    pass

            # 2. Z-Scores basés sur la ligne de base globale
            df_z = df_calcul.copy()
            for m in toutes_metriques:
                z_col = f"{m}_z"
                df_z[z_col] = np.nan
                for s in sexes_detectes:
                    masque_base = (df_z['day'] == phases_ordre[0]) & (df_z['Sex'] == s) & (df_z['Treatment'] == groupe_controle)
                    moyenne_base = df_z.loc[masque_base, f'{m}_adj'].mean()
                    std_base = df_z.loc[masque_base, f'{m}_adj'].std()
                    
                    if pd.isna(std_base) or std_base == 0: 
                        std_base = 1.0
                        
                    masque_sexe = (df_z['Sex'] == s)
                    df_z.loc[masque_sexe, z_col] = (df_z.loc[masque_sexe, f'{m}_adj'] - moyenne_base) / std_base

            # 3. Agrégation Globale
            for domaine, metriques in metriques_domaines.items():
                colonnes_z = [f"{m}_z" for m in metriques]
                df_z[f"{domaine}_Index"] = df_z[colonnes_z].mean(axis=1)

            indices_finaux = [f"{d}_Index" for d in metriques_domaines.keys()]
            df_modele = df_z[['group', 'RFID', 'Sex', 'Treatment', 'day'] + indices_finaux].dropna().copy()

            # ==============================================================================
            # FILTRAGE FINAL POUR VISUALISATION (SHAM vs CIBLE)
            # ==============================================================================
            st.header(t["res_h"])
            
            def obtenir_etoiles(p):
                if p < 0.001: return '***'
                if p < 0.01: return '**'
                if p < 0.05: return '*'
                return ''

            sns.set_theme(style="whitegrid", context="talk")
            fig, axes = plt.subplots(5, 2, figsize=(largeur_fig, hauteur_fig), sharex=True)
            
            palette = {groupe_controle: '#7f7f7f', groupe_cible: couleur_cible}
            etiquettes_sexe = {'M': 'Males', 'F': 'Females'}
            
            # Isolement strict pour la représentation graphique et les tests de Welch
            df_plot = df_modele[df_modele['Treatment'].isin([groupe_controle, groupe_cible])].copy()
            df_plot['Treatment'] = pd.Categorical(df_plot['Treatment'], categories=[groupe_controle, groupe_cible], ordered=True)
            
            for idx_domaine, nom_indice in enumerate(indices_finaux):
                domaine_cle = list(metriques_domaines.keys())[idx_domaine]
                for idx_sexe, s in enumerate(sexes_detectes):
                    ax = axes[idx_domaine, idx_sexe]
                    sous_groupe = df_plot[df_plot['Sex'] == s]
                    
                    if not sous_groupe.empty:
                        sns.pointplot(data=sous_groupe, x='day', y=nom_indice, hue='Treatment', palette=palette, 
                                      markers=['o', 's'], capsize=.1, ax=ax, dodge=True, errwidth=1.5)
                    
                    if activer_zoom_y:
                        ax.set_ylim(y_min, y_max)
                    
                    liste_m = metriques_domaines[domaine_cle]
                    metriques_formatees = " +\n".join([", ".join(liste_m[:2]), ", ".join(liste_m[2:])]) if len(liste_m) > 3 else " +\n".join(liste_m)
                    
                    ax.set_title(f"{domaine_cle.replace('_', ' ')} - {etiquettes_sexe.get(s, s)}\n({metriques_formatees})", fontsize=12, pad=12)
                    ax.axhline(0, ls='--', color='black', alpha=0.3)
                    ax.set_ylabel(t["ylab"] if idx_sexe == 0 else '')
                    ax.set_xlabel('')
                    
                    if ax.get_legend() is not None: 
                        ax.get_legend().remove()
                    
                    for idx_jour, jour in enumerate(phases_ordre):
                        valeurs_ctrl = sous_groupe[(sous_groupe['day'] == jour) & (sous_groupe['Treatment'] == groupe_controle)][nom_indice].dropna()
                        valeurs_cible = sous_groupe[(sous_groupe['day'] == jour) & (sous_groupe['Treatment'] == groupe_cible)][nom_indice].dropna()
                        
                        if len(valeurs_ctrl) > 2 and len(valeurs_cible) > 2:
                            _, p_val = ttest_ind(valeurs_ctrl, valeurs_cible, equal_var=False)
                            etoiles = obtenir_etoiles(p_val)
                            if etoiles:
                                moyenne_cible = valeurs_cible.mean()
                                texte_annotation = f"{etoiles}\n(p<0.001)" if p_val < 0.001 else f"{etoiles}\n(p={p_val:.3f})"
                                ax.text(idx_jour, moyenne_cible + 0.45, texte_annotation, color=couleur_cible, 
                                        ha='center', va='bottom', fontweight='bold', fontsize=taille_etoiles)
            
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, title="Treatment", loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=2)
            plt.tight_layout()
            fig.subplots_adjust(top=0.94, hspace=0.45)
            
            st.pyplot(fig)
            
            # ==============================================================================
            # EXPORTATION
            # ==============================================================================
            st.header(t["exp_h"])
            col1, col2 = st.columns(2)
            
            buffer_img = io.BytesIO()
            fig.savefig(buffer_img, format='png', bbox_inches='tight', dpi=300)
            buffer_img.seek(0)
            
            col1.download_button(
                label=t["dl_fig"],
                data=buffer_img,
                file_name=f"LMT_Figure_{groupe_controle}_vs_{groupe_cible}.png",
                mime="image/png"
            )
            
            buffer_csv = io.StringIO()
            df_plot.to_csv(buffer_csv, index=False)
            
            col2.download_button(
                label=t["dl_csv"],
                data=buffer_csv.getvalue(),
                file_name=f"LMT_Donnees_{groupe_controle}_vs_{groupe_cible}.csv",
                mime="text/csv"
            )
            st.success(t["exp_ok"])
