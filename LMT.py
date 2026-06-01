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

# Configuration de la page de la GUI Streamlit
st.set_page_config(
    page_title="LMT Phenotype Analytics - Manzoni Lab",
    page_icon="🧠",
    layout="wide"
)

# Style CSS personnalisé pour l'interface scientifique
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .main .block-container { padding-top: 2rem; }
    h1 { color: #1e3d59; }
    h2 { color: #17b978; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 LMT Phenotype Analytics : Pipeline de Plasticité Comportementale")
st.write("Conçu pour le **Manzoni Lab** — Interface Graphique d'Analyse des Trajectoires du Live Mouse Tracker.")

# ==============================================================================
# RATIONNELS TEXTUELS (Intégrés de manière ergonomique dans la GUI)
# ==============================================================================
with st.expander("📖 Afficher le Rationnel Mathématique & l'Architecture des Domaines", expanded=False):
    st.markdown("""
    ### 1. Logique Mathématique et Statistique
    * **Résidualisation Hiérarchique (LMM) :** Corrige le biais de pseudo-réplication en prenant la portée (`group`) comme intercept aléatoire, isolant l'effet pharmacologique ou génétique pur.
    * **Z-Score Ancré sur la Nouveauté :** Normalisation calée exclusivement sur la première heure d'exploration du groupe témoin ($Z=0$), transformant la cinétique en une trajectoire d'habituation lisible.
    * **Tests de Welch Point-par-Point :** Gère l'hétéroscédasticité induite par les traitements sans assumer l'égalité des variances.

    ### 2. Architecture des 5 Domaines Composites
    * **Locomotor Drive :** `MoveisolatedTotalLen + totalDistance` (Activité cinétique pure)
    * **Exploratory Spatial Strategy :** `Center_Periphery_Ratio + RearisolatedTotalLen` (Cartographie cognitive 3D)
    * **Vigilance / Risk Assessment :** `SAPNb + StopisolatedNb` (Marqueurs d'anxiété aiguë)
    * **Social Tolerance :** `Group2TotalLen + Group3TotalLen + SidebysideContactTotalLen` (Proximité passive)
    * **Active Social Engagement :** `Oral-oral + Oral-genital + Socialapproach + Train2 + FollowZone` (Interaction dirigée)
    """)

# ==============================================================================
# BARRE LATÉRALE - CONFIGURATION & IMPORTEUR DE FICHIERS
# ==============================================================================
st.sidebar.header("📁 Importation des Données")
fichier_uploade = st.sidebar.file_uploader("Glissez un fichier de données LMT (.txt) ici", type=["txt"])

# Chargement intelligent (mise en cache pour optimiser les performances de la GUI)
@st.cache_data
def charger_donnees(source):
    return pd.read_csv(source, sep='\t')

df_raw = None
if fichier_uploade is not None:
    df_raw = charger_donnees(fichier_uploade)
    st.sidebar.success("Données chargées !")
else:
    st.sidebar.warning("En attente d'un fichier .txt. Téléversez une manipulation pour démarrer.")

# Exécution de l'interface dès que les données sont disponibles
if df_raw is not None:
    df = df_raw.copy()
    
    # Extraction dynamique du Sexe et du Traitement (Ajustable selon la manip)
    st.sidebar.header("⚙️ Paramètres du Génotype")
    separateur = st.sidebar.text_input("Séparateur de la colonne 'genotype'", "_")
    
    try:
        df['Sex'] = df['genotype'].str.split(separateur).str[0]
        df['Treatment'] = df['genotype'].str.split(separateur).str[1]
    except Exception as e:
        st.sidebar.error(f"Erreur de parsing de la colonne 'genotype': {e}")

    # Détection automatique des facteurs expérimentaux
    groupes_detectes = sorted(df['Treatment'].dropna().unique().tolist())
    sexes_detectes = sorted(df['Sex'].dropna().unique().tolist())
    phases_detectees = sorted(df['day'].dropna().unique().tolist())

    st.sidebar.subheader("🔬 Sélection des Groupes")
    groupe_controle = st.sidebar.selectbox("Groupe CONTRÔLE (Référence Z=0)", groupes_detectes, index=groupes_detectes.index('SHAM') if 'SHAM' in groupes_detectes else 0)
    groupe_cible = st.sidebar.selectbox("Groupe TRAITÉ / MUTANT", [g for g in groupes_detectes if g != groupe_controle], index=0)
    
    phases_par_defaut = [p for p in ['Exploration', 'Night 1', 'Night 2', 'Night 3'] if p in phases_detectees]
    phases_ordre = st.sidebar.multiselect("Chronologie des phases (Axe X)", phases_detectees, default=phases_par_defaut)

    # Paramètres de rendu graphique et Zoom interactif
    st.sidebar.header("🎨 Personnalisation des Figures")
    largeur_fig = st.sidebar.slider("Largeur de la figure", 10, 24, 16)
    hauteur_fig = st.sidebar.slider("Hauteur de la figure", 15, 45, 28)
    taille_etoiles = st.sidebar.slider("Taille des étoiles / p-values", 12, 30, 18)
    couleur_cible = st.sidebar.color_picker("Couleur du groupe traité", "#d62728")
    
    activer_zoom_y = st.sidebar.checkbox("Activer le zoom manuel de l'axe Y")
    y_min, y_max = -3.0, 3.0
    if activer_zoom_y:
        col_z1, col_z2 = st.sidebar.columns(2)
        y_min = col_z1.number_input("Y Min", value=-3.0, step=0.5)
        y_max = col_z2.number_input("Y Max", value=3.0, step=0.5)

    # ==============================================================================
    # PIPELINE DE TRAITEMENT DES DONNÉES
    # ==============================================================================
    # Reconstruction et vérification des métriques
    df['Center_Periphery_Ratio'] = df['CenterZoneTotalLen'] / (df['PeripheryZoneTotalLen'] + 1e-5)
    
    metriques_domaines = {
        'Locomotor_Drive': ['MoveisolatedTotalLen', 'totalDistance'],
        'Exploratory_Spatial_Strategy': ['Center_Periphery_Ratio', 'RearisolatedTotalLen'],
        'Vigilance_Risk': ['SAPNb', 'StopisolatedNb'],
        'Social_Tolerance': ['Group2TotalLen', 'Group3TotalLen', 'SidebysideContactTotalLen', 'SidebysideContact,oppositewayTotalLen'],
        'Active_Social_Engagement': ['Oral-oralContactTotalLen', 'Oral-genitalContactTotalLen', 'SocialapproachNb', 'Train2TotalLen', 'FollowZoneTotalLen']
    }

    # Nettoyage des métriques absentes du fichier importé
    for domaine in metriques_domaines:
        metriques_domaines[domaine] = [m for m in metriques_domaines[domaine] if m in df.columns]

    toutes_metriques = [m for sous_liste in metriques_domaines.values() for m in sous_liste]
    for m in toutes_metriques:
        df[m] = df[m].fillna(0)

    # Filtrage des lignes
    df_filtered = df[df['Treatment'].isin([groupe_controle, groupe_cible])].copy()
    df_filtered['day'] = pd.Categorical(df_filtered['day'], categories=phases_ordre, ordered=True)
    df_filtered = df_filtered.dropna(subset=['day']).copy()

    # Déclenchement de l'analyse
    if st.button("🚀 Lancer l'Analyse Computationnelle", type="primary"):
        with st.spinner("Exécution du pipeline statistique en cours..."):
            
            # 1. Résidualisation LMM
            for m in toutes_metriques:
                df_filtered[f'{m}_adj'] = df_filtered[m]
                try:
                    modele = smf.mixedlm(f"Q('{m}') ~ Treatment", df_filtered, groups=df_filtered["group"])
                    resultat = modele.fit(method='cg')
                    effets_aleatoires = resultat.random_effects
                    df_filtered[f'{m}_adj'] = df_filtered.apply(
                        lambda ligne: ligne[m] - (effets_aleatoires[ligne['group']]['Group'] if ligne['group'] in effets_aleatoires else 0), 
                        axis=1
                    )
                except:
                    pass

            # 2. Calcul du Z-Score ancré
            df_z = df_filtered.copy()
            for m in toutes_metriques:
                z_col = f"{m}_z"
                df_z[z_col] = np.nan
                for s in sexes_detectes:
                    masque_base = (df_z['day'] == phases_ordre[0]) & (df_z['Sex'] == s) & (df_z['Treatment'] == groupe_controle)
                    moyenne_base = df_z.loc[masque_base, f'{m}_adj'].mean()
                    std_base = df_z.loc[masque_base, f'{m}_adj'].std()
                    
                    # Correction appliquée ici (or au lieu de ||)
                    if pd.isna(std_base) or std_base == 0: 
                        std_base = 1.0
                        
                    masque_sexe = (df_z['Sex'] == s)
                    df_z.loc[masque_sexe, z_col] = (df_z.loc[masque_sexe, f'{m}_adj'] - moyenne_base) / std_base

            # 3. Agrégation des indices composites
            for domaine, metriques in metriques_domaines.items():
                colonnes_z = [f"{m}_z" for m in metriques]
                df_z[f"{domaine}_Index"] = df_z[colonnes_z].mean(axis=1)

            indices_finaux = [f"{d}_Index" for d in metriques_domaines.keys()]
            df_modele = df_z[['group', 'RFID', 'Sex', 'Treatment', 'day'] + indices_finaux].dropna().copy()

            # ==============================================================================
            # CONSTRUCTIONS ET RENDU DES GRAPHIQUES
            # ==============================================================================
            st.header("📊 Trajectoires Phénotypiques Longitudes")
            
            def obtenir_etoiles(p):
                if p < 0.001: return '***'
                if p < 0.01: return '**'
                if p < 0.05: return '*'
                return ''

            sns.set_theme(style="whitegrid", context="talk")
            fig, axes = plt.subplots(5, 2, figsize=(largeur_fig, hauteur_fig), sharex=True)
            
            palette = {groupe_controle: '#7f7f7f', groupe_cible: couleur_cible}
            etiquettes_sexe = {'M': 'Males', 'F': 'Females'}
            
            df_plot = df_modele.copy()
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
                    ax.set_ylabel('Adjusted Z-Score\n(Litter-Corrected)' if idx_sexe == 0 else '')
                    ax.set_xlabel('')
                    
                    if ax.get_legend() is not None: 
                        ax.get_legend().remove()
                    
                    # Extraction des p-values Welch t-test point-par-point
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
            
            # Rendu visuel immédiat dans la GUI
            st.pyplot(fig)
            
            # ==============================================================================
            # ZONE D'EXPORTATION ET TÉLÉCHARGEMENT
            # ==============================================================================
            st.header("💾 Exportation des Résultats")
            col1, col2 = st.columns(2)
            
            # 1. Sauvegarde de la figure haute définition en mémoire pour téléchargement
            buffer_img = io.BytesIO()
            fig.savefig(buffer_img, format='png', bbox_inches='tight', dpi=300)
            buffer_img.seek(0)
            
            col1.download_button(
                label="🖼️ Télécharger la Figure (Qualité Publication - DPI 300)",
                data=buffer_img,
                file_name=f"LMT_Figure_{groupe_controle}_vs_{groupe_cible}.png",
                mime="image/png"
            )
            
            # 2. Sauvegarde du fichier de données calculées au format CSV
            buffer_csv = io.StringIO()
            df_modele.to_csv(buffer_csv, index=False)
            
            col2.download_button(
                label="📊 Télécharger les Données Calculées (Z-Scores & Indices CSV)",
                data=buffer_csv.getvalue(),
                file_name=f"LMT_Donnees_{groupe_controle}_vs_{groupe_cible}.csv",
                mime="text/csv"
            )
            st.success("Analyses prêtes pour exportation !")
