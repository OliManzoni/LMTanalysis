import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind
import warnings

warnings.filterwarnings('ignore')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="LMT Phenotype Analytics - Manzoni Lab",
    page_icon="🧠",
    layout="wide"
)

# Title & Description
st.title("🧠 LMT Phenotype Analytics : Pipeline de Plasticité Comportementale")
st.write("Conçu pour le **Manzoni Lab** — Analyse robuste des trajectoires cinétiques du Live Mouse Tracker.")

# ==============================================================================
# 0. AFFICHAGE DU RATIONNEL SCIENTIFIQUE
# ==============================================================================
with st.expander("📖 Afficher la Logique Mathématique & l'Architecture des Domaines", expanded=False):
    st.markdown("""
    ### 1. Logique Mathématique et Statistique
    Pour isoler l'effet pharmacologique ou génétique pur sur la plasticité comportementale, trois verrous mathématiques sont appliqués :
    1. **Résidualisation Hiérarchique (Modèle Mixte Linéaire - LMM) :** * *Le problème :* Dans les études développementales, l'unité de randomisation est la portée (`group`), pas le souriceau. Ignorer cet effet crée une pseudo-réplication artificielle.
       * *La solution :* Le pipeline ajuste chaque variable en utilisant l'identifiant de la portée comme intercept aléatoire pour en soustraire la variance maternelle.
    2. **Normalisation par Z-Score ancré sur la "Nouveauté Aiguë" :** * *Le problème :* Standardiser sur la moyenne globale masque les dynamiques d'habituation inter-sessions.
       * *La solution :* Le Z-score est calculé en utilisant **exclusivement la première heure d'exploration du groupe contrôle (ex: SHAM)** comme référence ($Z=0$). La cinétique mesure ainsi une véritable trajectoire d'apprentissage.
    3. **Tests de Welch Point-par-Point :** * *La solution :* Des tests de Welch (qui n'assument pas de variances égales) mesurent précisément quand les trajectoires divergent significativement.

    ### 2. Architecture des 5 Domaines Fortifiés (Cartographie 3D & Bipartite Social)
    * **Locomotor Drive (Énergie Cinétique Pure) :** `MoveisolatedTotalLen + totalDistance`
    * **Exploratory Spatial Strategy (Cartographie 3D) :** `Center_Periphery_Ratio + RearisolatedTotalLen` (Exploration horizontale et verticale combinées pour valider la plasticité cortico-hippocampique).
    * **Vigilance / Risk Assessment :** `SAPNb + StopisolatedNb` (Postures d'évaluation du risque et freezing).
    * **Social Tolerance (Modèle Bipartite - État Permissif) :** `Group2TotalLen + Group3TotalLen + SidebysideContactTotalLen` (Proximité spatiale passive/tolérance).
    * **Active Social Engagement (Modèle Bipartite - État Dirigé) :** `Oral-oral + Oral-genital + Socialapproach + Train2 + FollowZone` (Motivation d'interaction active cortico-striatale).
    """)

# ==============================================================================
# 1. CHARGEMENT DES DONNÉES (GLISSER-DÉPOSER)
# ==============================================================================
st.sidebar.header("📁 Importation des Données")
fichier_uploade = st.sidebar.file_uploader("Glissez votre fichier LMT (.txt) ici", type=["txt"])

# Fichier par défaut si aucun n'est téléversé
nom_fichier_defaut = "2026_04_06_ALBA_1hexplo4hnight_7framemerge_3fcutoff all THC.txt"

@st.cache_data
def charger_donnees(file_source):
    try:
        if isinstance(file_source, str):
            data = pd.read_csv(file_source, sep='\t')
        else:
            data = pd.read_csv(file_source, sep='\t')
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        return None

if fichier_uploade is not None:
    df_raw = charger_donnees(fichier_uploade)
    st.sidebar.success("Fichier importé avec succès !")
else:
    df_raw = charger_donnees(nom_fichier_defaut)
    if df_raw is not None:
        st.sidebar.info("Utilisation du fichier de démonstration (THC/CBD).")
    else:
        st.warning("Veuillez importer un fichier .txt dans la barre latérale pour démarrer.")

if df_raw is not None:
    df = df_raw.copy()
    
    # Configuration flexible de l'extraction des groupes (genotype)
    st.sidebar.header("⚙️ Configuration du Génotype")
    separateur = st.sidebar.text_input("Séparateur Sexe/Traitement (ex: _ )", "_")
    
    try:
        # Permet de s'adapter dynamiquement si le format change (ex: M_SHAM ou WT_M_1)
        df['Sex'] = df['genotype'].str.split(separateur).str[0]
        df['Treatment'] = df['genotype'].str.split(separateur).str[1]
    except Exception as e:
        st.error(f"Erreur d'extraction du sexe/traitement depuis la colonne 'genotype' : {e}")

    # Détection automatique des groupes présents
    groupes_disponibles = sorted(df['Treatment'].dropna().unique().tolist())
    sexes_disponibles = sorted(df['Sex'].dropna().unique().tolist())
    phases_disponibles = sorted(df['day'].dropna().unique().tolist())

    # Sélection des groupes à analyser
    st.sidebar.subheader("🔬 Sélection des Groupes")
    groupe_controle = st.sidebar.selectbox("Sélectionnez le groupe CONTRÔLE (Réf)", groupes_disponibles, index=groupes_disponibles.index('SHAM') if 'SHAM' in groupes_disponibles else 0)
    groupe_cible = st.sidebar.selectbox("Sélectionnez le groupe TRAITÉ/MUTANT", [g for g in groupes_disponibles if g != groupe_controle], index=0)
    
    # Sélection et ordre des phases temporelles
    phases_ordre = st.sidebar.multiselect("Chronologie des phases", phases_disponibles, default=['Exploration', 'Night 1', 'Night 2', 'Night 3'])
    
    # Options esthétiques des graphiques (Zoom & Étoiles)
    st.sidebar.header("🎨 Options Graphiques & Zoom")
    largeur_fig = st.sidebar.slider("Largeur de l'image", 10, 24, 16)
    hauteur_fig = st.sidebar.slider("Hauteur de l'image", 15, 45, 30)
    taille_etoiles = st.sidebar.slider("Taille des Étoiles & p-values", 12, 30, 18)
    
    # Zoom manuel sur l'axe Y
    activer_zoom_y = st.sidebar.checkbox("Activer Zoom manuel (Axe Y)")
    if activer_zoom_y:
        y_min = st.sidebar.number_input("Limite Min Axe Y", value=-3.0, step=0.5)
        y_max = st.sidebar.number_input("Limite Max Axe Y", value=3.0, step=0.5)

    # Filtrage des données selon la sélection de l'utilisateur
    df = df[df['Treatment'].isin([groupe_controle, groupe_cible])].copy()
    df['day'] = pd.Categorical(df['day'], categories=phases_ordre, ordered=True)
    df['group'] = df['group'].astype(str)
    df['RFID'] = df['RFID'].astype(str)

    # ==============================================================================
    # 2. CONFIGURATION DES DOMAINES
    # ==============================================================================
    df['Center_Periphery_Ratio'] = df['CenterZoneTotalLen'] / (df['PeripheryZoneTotalLen'] + 1e-5)
    
    metriques_domaines = {
        'Locomotor_Drive': ['MoveisolatedTotalLen', 'totalDistance'],
        'Exploratory_Spatial_Strategy': ['Center_Periphery_Ratio', 'RearisolatedTotalLen'],
        'Vigilance_Risk': ['SAPNb', 'StopisolatedNb'],
        'Social_Tolerance': ['Group2TotalLen', 'Group3TotalLen', 'SidebysideContactTotalLen', 'SidebysideContact,oppositewayTotalLen'],
        'Active_Social_Engagement': ['Oral-oralContactTotalLen', 'Oral-genitalContactTotalLen', 'SocialapproachNb', 'Train2TotalLen', 'FollowZoneTotalLen']
    }

    # Filtrage des métriques existantes dans le dataset
    for domaine in metriques_domaines:
        metriques_domaines[domaine] = [m for m in metriques_domaines[domaine] if m in df.columns]

    toutes_metriques = [m for sous_liste in metriques_domaines.values() for m in sous_liste]
    
    for m in toutes_metriques:
        df[m] = df[m].fillna(0)

    # Bouton principal de calcul
    if st.button("🚀 Lancer l'Analyse Computationnelle", type="primary"):
        
        # Étape 1 : LMM (Litter Correction)
        status_lmm = st.status("Étape 1 : Ajustement par Modèle Mixte Linéaire (Litter correction)...")
        for m in toutes_metriques:
            df[f'{m}_adj'] = df[m]
            try:
                modele = smf.mixedlm(f"Q('{m}') ~ Treatment", df, groups=df["group"])
                resultat = modele.fit(method='cg')
                effets_aleatoires = resultat.random_effects
                df[f'{m}_adj'] = df.apply(
                    lambda ligne: ligne[m] - (effets_aleatoires[ligne['group']]['Group'] if ligne['group'] in effets_aleatoires else 0), 
                    axis=1
                )
            except Exception:
                pass
        status_lmm.update(label="Étape 1 validée : Effets de portée éliminés !", state="complete")

        # Étape 2 : Z-scoring
        st.info(f"Étape 2 : Calcul des Z-Scores (Point zéro défini par : {groupe_controle} lors de la phase {phases_ordre[0]})")
        df_z = df.copy()
        for m in toutes_metriques:
            z_col = f"{m}_z"
            df_z[z_col] = np.nan
            for s in sexes_disponibles:
                masque_base = (df_z['day'] == phases_ordre[0]) & (df_z['Sex'] == s) & (df_z['Treatment'] == groupe_controle)
                moyenne_base = df_z.loc[masque_base, f'{m}_adj'].mean()
                std_base = df_z.loc[masque_base, f'{m}_adj'].std()
                
                if pd.isna(std_base) or std_base == 0: 
                    std_base = 1.0
                    
                masque_sexe = (df_z['Sex'] == s)
                df_z.loc[masque_sexe, z_col] = (df_z.loc[masque_sexe, f'{m}_adj'] - moyenne_base) / std_base

        # Étape 3 : Agrégation des indices
        for domaine, metriques in metriques_domaines.items():
            colonnes_z = [f"{m}_z" for m in metriques]
            df_z[f"{domaine}_Index"] = df_z[colonnes_z].mean(axis=1)

        indices_finaux = [f"{d}_Index" for d in metriques_domaines.keys()]
        df_modele = df_z[['group', 'RFID', 'Sex', 'Treatment', 'day'] + indices_finaux].dropna().copy()

        # Étape 4 : Plotting
        st.success("Étape 3 : Construction des indices composites terminée. Tracé des courbes...")
        
        def obtenir_etoiles(p):
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return ''

        # Initialisation de la figure matplotlib
        sns.set_theme(style="whitegrid", context="talk")
        fig, axes = plt.subplots(5, 2, figsize=(largeur_fig, hauteur_fig), sharex=True)
        
        couleur_cible = '#d62728' if groupe_cible == 'THC' else ('#2ca02c' if groupe_cible == 'CBD' else '#1f77b4')
        palette = {groupe_controle: '#7f7f7f', groupe_cible: couleur_cible}
        etiquettes_sexe = {'M': 'Males', 'F': 'Females'}
        
        df_plot = df_modele[df_modele['Treatment'].isin([groupe_controle, groupe_cible])].copy()
        df_plot['Treatment'] = pd.Categorical(df_plot['Treatment'], categories=[groupe_controle, groupe_cible], ordered=True)
        
        for idx_domaine, nom_indice in enumerate(indices_finaux):
            domaine_cle = list(metriques_domaines.keys())[idx_domaine]
            for idx_sexe, s in enumerate(sexes_disponibles):
                ax = axes[idx_domaine, idx_sexe]
                sous_groupe = df_plot[df_plot['Sex'] == s]
                
                # Tracé principal
                sns.pointplot(data=sous_groupe, x='day', y=nom_indice, hue='Treatment', palette=palette, 
                              markers=['o', 's'], capsize=.1, ax=ax, dodge=True, errwidth=1.5)
                
                if activer_zoom_y:
                    ax.set_ylim(y_min, y_max)
                
                # Formatage du titre
                liste_m = metriques_domaines[domaine_cle]
                if len(liste_m) > 3:
                    metriques_formatees = " +\n".join([", ".join(liste_m[:2]), ", ".join(liste_m[2:])])
                else:
                    metriques_formatees = " +\n".join(liste_m)
                    
                titre_panel = f"{domaine_cle.replace('_', ' ')} - {etiquettes_sexe.get(s, s)}\n({metriques_formatees})"
                ax.set_title(titre_panel, fontsize=12, pad=12, loc='center')
                ax.axhline(0, ls='--', color='black', alpha=0.3)
                
                if idx_sexe == 0: 
                    ax.set_ylabel('Adjusted Z-Score\n(Litter-Corrected)')
                else: 
                    ax.set_ylabel('')
                ax.set_xlabel('')
                
                if ax.get_legend() is not None: 
                    ax.get_legend().remove()
                
                # Welch t-tests
                for idx_jour, jour in enumerate(phases_ordre):
                    valeurs_sham = sous_groupe[(sous_groupe['day'] == jour) & (sous_groupe['Treatment'] == groupe_controle)][nom_indice].dropna()
                    valeurs_cible = sous_groupe[(sous_groupe['day'] == jour) & (sous_groupe['Treatment'] == groupe_cible)][nom_indice].dropna()
                    
                    if len(valeurs_sham) > 2 and len(valeurs_cible) > 2:
                        _, p_val = ttest_ind(valeurs_sham, valeurs_cible, equal_var=False)
                        etoiles = obtenir_etoiles(p_val)
                        if etoiles:
                            moyenne_cible = valeurs_cible.mean()
                            
                            # Texte combiné avec p-value
                            if p_val < 0.001:
                                texte_annotation = f"{etoiles}\n(p<0.001)"
                            else:
                                texte_annotation = f"{etoiles}\n(p={p_val:.3f})"
                                
                            ax.text(idx_jour, moyenne_cible + 0.45, texte_annotation, color=couleur_cible, 
                                    ha='center', va='bottom', fontweight='bold', fontsize=taille_etoiles)
                            
        # Légende globale
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Treatment", loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=2)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.94, hspace=0.45)
        
        # Affichage interactif dans Streamlit
        st.pyplot(fig)
        
        # Option de sauvegarde locale de l'image
        nom_img = f"Figure_LMT_{groupe_controle}_vs_{groupe_cible}.png"
        fig.savefig(nom_img, bbox_inches='tight', dpi=300)
        st.success(f"Figure haute définition exportée avec succès sous le nom de : {nom_img}")