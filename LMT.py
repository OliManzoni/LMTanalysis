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

# Configuration de la page
st.set_page_config(
    page_title="LMT Phenotype Analytics",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .main .block-container { padding-top: 2rem; }
    h1 { color: #1e3d59; }
    h2 { color: #17b978; }
    </style>
    """, unsafe_allow_html=True)

lang = st.sidebar.radio("🌐 Language / Langue", ["Français", "English"])

text = {
    "Français": {
        "title": "🧠 LMT Phenotype Analytics : Pipeline de Plasticité Comportementale",
        "subtitle": "Interface Graphique d'Analyse des Trajectoires du Live Mouse Tracker (Version Fortifiée).",
        "rationale_exp": "📖 Afficher le Rationnel Mathématique",
        "import_h": "📁 Importation des Données",
        "upload": "Glissez un fichier (.txt ou .csv) ici",
        "load_ok": "Données chargées avec succès !",
        "wait_file": "En attente d'un fichier .txt ou .csv.",
        "param_h": "⚙️ Paramètres Expérimentaux",
        "col_select": "Sélectionnez la colonne contenant le Génotype :",
        "sep": "Séparateur de la colonne (ex: '_' pour M_THC)",
        "err_parse": "Erreur de séparation. Vérifiez le séparateur.",
        "cohort_h": "1️⃣ Cohorte Globale (LMM)",
        "cohort_help": "Sélectionnez tous les groupes à inclure dans le calcul de la variance maternelle.",
        "domain_config_h": "🧩 Configuration des Domaines",
        "group_h": "2️⃣ Comparaison (Graphique)",
        "ctrl": "Groupe CONTRÔLE (Référence Z=0)",
        "target": "Groupe TRAITÉ / MUTANT",
        "phase": "Chronologie des phases (Axe X)",
        "plot_h": "🎨 Personnalisation",
        "w": "Largeur",
        "h": "Hauteur",
        "star": "Taille étoiles",
        "color": "Couleur cible",
        "zoom": "Zoom Axe Y",
        "run": "🚀 Lancer l'Analyse Computationnelle Rigoureuse",
        "running": "Calcul des résidus LMM et Z-scores en cours...",
        "res_h": "📊 Trajectoires Phénotypiques (Litter-Corrected)",
        "ylab": "Adjusted Z-Score\n(Litter-Corrected)",
        "exp_h": "💾 Exportation",
        "dl_fig": "🖼️ Télécharger la Figure",
        "dl_csv": "📊 Télécharger les Données",
        "exp_ok": "Prêt !"
    },
    "English": {
        "title": "🧠 LMT Phenotype Analytics: Behavioral Plasticity Pipeline",
        "subtitle": "Graphical Interface for Live Mouse Tracker Trajectory Analysis (Fortified Version).",
        "rationale_exp": "📖 Show Mathematical Rationale",
        "import_h": "📁 Data Import",
        "upload": "Drop a data file (.txt or .csv) here",
        "load_ok": "Data successfully loaded!",
        "wait_file": "Waiting for a .txt or .csv file.",
        "param_h": "⚙️ Experimental Parameters",
        "col_select": "Select the column containing the Genotype:",
        "sep": "Column separator (e.g., '_' for M_THC)",
        "err_parse": "Parsing error. Check the separator.",
        "cohort_h": "1️⃣ Global Cohort (LMM)",
        "cohort_help": "Select all groups to include in maternal variance calculation.",
        "domain_config_h": "🧩 Domain Configuration",
        "group_h": "2️⃣ Comparison (Plot)",
        "ctrl": "CONTROL Group (Reference Z=0)",
        "target": "TREATED / MUTANT Group",
        "phase": "Chronology of phases",
        "plot_h": "🎨 Customization",
        "w": "Width",
        "h": "Height",
        "star": "Star size",
        "color": "Target color",
        "zoom": "Y-Axis Zoom",
        "run": "🚀 Run Rigorous Computational Analysis",
        "running": "Computing LMM residuals and Z-scores...",
        "res_h": "📊 Phenotypic Trajectories (Litter-Corrected)",
        "ylab": "Adjusted Z-Score\n(Litter-Corrected)",
        "exp_h": "💾 Export",
        "dl_fig": "🖼️ Download Figure",
        "dl_csv": "📊 Download Data",
        "exp_ok": "Ready!"
    }
}

t = text[lang]

st.title(t["title"])
st.write(t["subtitle"])

st.sidebar.header(t["import_h"])
fichier_uploade = st.sidebar.file_uploader(t["upload"], type=["txt", "csv"])

@st.cache_data
def charger_donnees(fichier):
    content = fichier.read()
    separateurs_a_tester = ['\t', ',', ';']
    for sep in separateurs_a_tester:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep)
            if len(df.columns) > 3:
                df.columns = df.columns.str.strip()
                return df
        except:
            continue
    try:
        df = pd.read_csv(io.BytesIO(content))
        df.columns = df.columns.str.strip()
        return df
    except:
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
    
    colonnes_dispos = list(df.columns)
    colonne_defaut_idx = 0
    for i, col in enumerate(colonnes_dispos):
        if col.lower() == 'genotype':
            colonne_defaut_idx = i
            break
        elif i == 1: 
            colonne_defaut_idx = 1
            
    colonne_choisie = st.sidebar.selectbox(t["col_select"], colonnes_dispos, index=colonne_defaut_idx)
    df.rename(columns={colonne_choisie: 'genotype'}, inplace=True)
    
    separateur = st.sidebar.text_input(t["sep"], "_")
    
    try:
        df['Sex'] = df['genotype'].str.split(separateur).str[0]
        df['Treatment'] = df['genotype'].str.split(separateur).str[1]
    except Exception as e:
        st.sidebar.error(t["err_parse"])
        st.stop()

    groupes_detectes = sorted(df['Treatment'].dropna().unique().tolist())
    sexes_detectes = sorted(df['Sex'].dropna().unique().tolist())
    phases_detectees = sorted(df['day'].dropna().unique().tolist())

    st.sidebar.subheader(t["cohort_h"])
    st.sidebar.caption(t["cohort_help"])
    defaut_cohort = ['SHAM', 'THC', 'CBD'] if all(g in groupes_detectes for g in ['SHAM', 'THC', 'CBD']) else groupes_detectes
    groupes_inclusion = st.sidebar.multiselect("Groupes LMM:", groupes_detectes, default=defaut_cohort)

    if not groupes_inclusion:
        st.sidebar.warning("Veuillez sélectionner au moins un groupe.")
        st.stop()
        
    if 'CenterZoneTotalLen' in df.columns and 'PeripheryZoneTotalLen' in df.columns:
        df['Center_Periphery_Ratio'] = df['CenterZoneTotalLen'] / (df['PeripheryZoneTotalLen'] + 1e-5)
    
    colonnes_exclues = ['RFID', 'group', 'day', 'Sex', 'Treatment', 'genotype']
    colonnes_possibles = [c for c in df.columns if c not in colonnes_exclues and pd.api.types.is_numeric_dtype(df[c])]
    
    defauts_domaines_architectures = {
        'Locomotor_Drive': ['MoveisolatedTotalLen', 'totalDistance'],
        'Exploratory_Spatial_Strategy': ['Center_Periphery_Ratio', 'RearisolatedTotalLen'],
        'Vigilance_Risk': ['SAPNb', 'StopisolatedNb'],
        'Social_Tolerance': ['Group2TotalLen', 'Group3TotalLen', 'SidebysideContactTotalLen', 'SidebysideContact,oppositewayTotalLen'],
        'Active_Social_Engagement': ['Oral-oralContactTotalLen', 'Oral-genitalContactTotalLen', 'SocialapproachNb', 'Train2TotalLen', 'FollowZoneTotalLen']
    }
    
    metriques_domaines = {}
    with st.sidebar.expander(t["domain_config_h"], expanded=False):
        for domaine, defauts in defauts_domaines_architectures.items():
            defauts_presents = [m for m in defauts if m in colonnes_possibles]
            selection = st.multiselect(f"{domaine.replace('_', ' ')}", options=colonnes_possibles, default=defauts_presents)
            if selection:
                metriques_domaines[domaine] = selection

    st.sidebar.subheader(t["group_h"])
    idx_sham = groupes_inclusion.index('SHAM') if 'SHAM' in groupes_inclusion else 0
    groupe_controle = st.sidebar.selectbox(t["ctrl"], groupes_inclusion, index=idx_sham)
    groupe_cible = st.sidebar.selectbox(t["target"], [g for g in groupes_inclusion if g != groupe_controle], index=0)
    
    phases_par_defaut = [p for p in ['Exploration', 'Night 1', 'Night 2', 'Night 3'] if p in phases_detectees]
    phases_ordre = st.sidebar.multiselect(t["phase"], phases_detectees, default=phases_par_defaut)

    st.sidebar.header(t["plot_h"])
    largeur_fig = st.sidebar.slider(t["w"], 10, 24, 16)
    hauteur_fig = st.sidebar.slider(t["h"], 15, 45, 28)
    taille_etoiles = st.sidebar.slider(t["star"], 12, 30, 18)
    couleur_cible = st.sidebar.color_picker(t["color"], "#2ca02c" if groupe_cible == 'CBD' else "#d62728")

    toutes_metriques = [m for sous_liste in metriques_domaines.values() for m in sous_liste]

    if st.button(t["run"], type="primary"):
        if not metriques_domaines:
            st.error("Configurez au moins un domaine.")
            st.stop()
            
        with st.spinner(t["running"]):
            df_calcul = df[df['Treatment'].isin(groupes_inclusion)].copy()
            for m in toutes_metriques:
                df_calcul[m] = df_calcul[m].fillna(0)
            
            df_calcul['day'] = pd.Categorical(df_calcul['day'], categories=phases_ordre, ordered=True)
            df_calcul['group'] = df_calcul['group'].astype(str)
            df_calcul['RFID'] = df_calcul['RFID'].astype(str)

            # Exécution rigoureuse du LMM avec C(Treatment) pour forcer le modèle catégoriel
            for m in toutes_metriques:
                df_calcul[f'{m}_adj'] = df_calcul[m]
                try:
                    modele = smf.mixedlm(f"Q('{m}') ~ C(Treatment)", df_calcul, groups=df_calcul["group"])
                    resultat = modele.fit(method='cg')
                    effets_aleatoires = resultat.random_effects
                    df_calcul[f'{m}_adj'] = df_calcul.apply(
                        lambda ligne: ligne[m] - (effets_aleatoires[ligne['group']]['Group'] if ligne['group'] in effets_aleatoires else 0), 
                        axis=1
                    )
                except:
                    pass

            df_z = df_calcul.copy()
            for m in toutes_metriques:
                z_col = f"{m}_z"
                df_z[z_col] = np.nan
                for s in sexes_detectes:
                    masque_base = (df_z['day'] == phases_ordre[0]) & (df_z['Sex'] == s) & (df_z['Treatment'] == groupe_controle)
                    moyenne_base = df_z.loc[masque_base, f'{m}_adj'].mean()
                    std_base = df_z.loc[masque_base, f'{m}_adj'].std()
                    if pd.isna(std_base) or std_base == 0: std_base = 1.0
                    masque_sexe = (df_z['Sex'] == s)
                    df_z.loc[masque_sexe, z_col] = (df_z.loc[masque_sexe, f'{m}_adj'] - moyenne_base) / std_base

            for domaine, metriques in metriques_domaines.items():
                colonnes_z = [f"{m}_z" for m in metriques]
                df_z[f"{domaine}_Index"] = df_z[colonnes_z].mean(axis=1)

            indices_finaux = [f"{d}_Index" for d in metriques_domaines.keys()]
            df_modele = df_z[['group', 'RFID', 'Sex', 'Treatment', 'day'] + indices_finaux].dropna().copy()

            st.header(t["res_h"])
            
            def obtenir_etoiles(p):
                if p < 0.001: return '***'
                if p < 0.01: return '**'
                if p < 0.05: return '*'
                return ''

            sns.set_theme(style="whitegrid", context="talk")
            fig, axes = plt.subplots(len(metriques_domaines), 2, figsize=(largeur_fig, 5 * len(metriques_domaines)), sharex=True)
            if len(metriques_domaines) == 1: axes = np.array([axes])
                
            palette = {groupe_controle: '#7f7f7f', groupe_cible: couleur_cible}
            etiquettes_sexe = {'M': 'Males', 'F': 'Females'}
            
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
                    
                    liste_m = metriques_domaines[domaine_cle]
                    metriques_formatees = " +\n".join([", ".join(liste_m[:2]), ", ".join(liste_m[2:])]) if len(liste_m) > 3 else " +\n".join(liste_m)
                    
                    ax.set_title(f"{domaine_cle.replace('_', ' ')} - {etiquettes_sexe.get(s, s)}\n({metriques_formatees})", fontsize=12, pad=12)
                    ax.axhline(0, ls='--', color='black', alpha=0.3)
                    ax.set_ylabel(t["ylab"] if idx_sexe == 0 else '')
                    ax.set_xlabel('')
                    if ax.get_legend() is not None: ax.get_legend().remove()
                    
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
            
            st.header(t["exp_h"])
            col1, col2 = st.columns(2)
            buffer_img = io.BytesIO()
            fig.savefig(buffer_img, format='png', bbox_inches='tight', dpi=300)
            buffer_img.seek(0)
            
            col1.download_button(label=t["dl_fig"], data=buffer_img, file_name=f"LMT_Figure_Rigorous.png", mime="image/png")
            buffer_csv = io.StringIO()
            df_plot.to_csv(buffer_csv, index=False)
            col2.download_button(label=t["dl_csv"], data=buffer_csv.getvalue(), file_name=f"LMT_Data_Rigorous.csv", mime="text/csv")
