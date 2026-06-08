import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind, false_discovery_control
import io
import warnings

warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
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

# ── Translations ──────────────────────────────────────────────────────────────
text = {
    "Français": {
        "title": "🧠 LMT Phenotype Analytics : Pipeline de Plasticité Comportementale",
        "subtitle": "Interface Graphique d'Analyse des Trajectoires du Live Mouse Tracker (Version Corrigée).",
        "rationale_exp": "📖 Mode d'Emploi & Rationnel Mathématique",
        # Keep original markdown rationale here...
        "rationale_md": """(Documentation omitted for brevity - use original here)""",
        "import_h": "📁 Importation des Données",
        "upload": "Glissez un fichier (.txt ou .csv) ici",
        "load_ok": "Données chargées avec succès !",
        "wait_file": "En attente d'un fichier .txt ou .csv.",
        "param_h": "⚙️ Paramètres Expérimentaux",
        "col_select": "Sélectionnez la colonne contenant le Génotype :",
        "sep": "Séparateur (ex: '_' pour M_THC)",
        "err_parse": "Erreur de séparation. Vérifiez le séparateur.",
        "rfid_reuse_warn": "ℹ️ {} puces RFID réutilisées entre plusieurs fichiers sources. Traité comme animaux distincts via animal_id (fichier+RFID).",
        "true_dup_warn": "⚠️ {} lignes dupliquées exactes (même fichier, RFID et jour) supprimées. Vérifiez vos exports.",
        "cohort_h": "1️⃣ Cohorte Globale (LMM)",
        "cohort_help": "Sélectionnez tous les groupes à inclure dans le calcul de la variance de cage.",
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
        "run": "🚀 Lancer l'Analyse",
        "running": "Calcul des résidus LMM et Z-scores en cours...",
        "lmm_warn": "⚠️ LMM a échoué pour '{}' ({}). Données brutes utilisées.",
        "baseline_warn": "⚠️ Écart-type nul pour '{}' (Sexe={}, Phase de référence={}). SD=1 utilisé — vérifiez cette métrique.",
        "lmm_stats_h": "📈 Statistiques Globales LMM (Fixes & Aléatoires)",
        "res_h": "📊 Trajectoires Phénotypiques (Cage-Corrected)",
        "ylab": "Adjusted Z-Score\n(Cage-Corrected)",
        "exp_h": "💾 Exportation",
        "dl_fig": "🖼️ Télécharger la Figure",
        "dl_csv": "📊 Télécharger les Données",
        "fdr_label": "BH-FDR",
        "n_label": "n=",
    },
    "English": {
        "title": "🧠 LMT Phenotype Analytics: Behavioral Plasticity Pipeline",
        "subtitle": "Graphical Interface for Live Mouse Tracker Trajectory Analysis (Corrected Version).",
        "rationale_exp": "📖 Manual & Mathematical Rationale",
        # Keep original markdown rationale here...
        "rationale_md": """(Documentation omitted for brevity - use original here)""",
        "import_h": "📁 Data Import",
        "upload": "Drop a data file (.txt or .csv) here",
        "load_ok": "Data successfully loaded!",
        "wait_file": "Waiting for a .txt or .csv file.",
        "param_h": "⚙️ Experimental Parameters",
        "col_select": "Select the column containing the Genotype:",
        "sep": "Column separator (e.g., '_' for M_THC)",
        "err_parse": "Parsing error. Check the separator.",
        "rfid_reuse_warn": "ℹ️ {} RFID chips reused across multiple source files. Treated as distinct animals via animal_id (file+RFID).",
        "true_dup_warn": "⚠️ {} exact duplicate rows (same file, RFID and day) removed. Check your exports.",
        "cohort_h": "1️⃣ Global Cohort (LMM)",
        "cohort_help": "Select all groups to include in cage variance calculation.",
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
        "run": "🚀 Run Analysis",
        "running": "Computing LMM residuals and Z-scores...",
        "lmm_warn": "⚠️ LMM failed for '{}' ({}). Raw values used.",
        "baseline_warn": "⚠️ Zero SD for '{}' (Sex={}, baseline phase={}). SD=1 used — check this metric.",
        "lmm_stats_h": "📈 LMM Global Statistics (Fixed & Random Effects)",
        "res_h": "📊 Phenotypic Trajectories (Cage-Corrected)",
        "ylab": "Adjusted Z-Score\n(Cage-Corrected)",
        "exp_h": "💾 Export",
        "dl_fig": "🖼️ Download Figure",
        "dl_csv": "📊 Download Data",
        "fdr_label": "BH-FDR",
        "n_label": "n=",
    }
}

t = text[lang]

st.title(t["title"])
st.write(t["subtitle"])

with st.expander(t["rationale_exp"], expanded=False):
    st.markdown(t["rationale_md"])

# ── Data loading ──────────────────────────────────────────────────────────────
st.sidebar.header(t["import_h"])
fichier_uploade = st.sidebar.file_uploader(t["upload"], type=["txt", "csv"])

@st.cache_data
def charger_donnees(fichier):
    content = fichier.read()
    for sep in ['\t', ',', ';']:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep)
            if len(df.columns) > 3:
                df.columns = df.columns.str.strip()
                return df
        except Exception:
            continue
    try:
        df = pd.read_csv(io.BytesIO(content))
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return None

df_raw = None
if fichier_uploade is not None:
    df_raw = charger_donnees(fichier_uploade)
    if df_raw is not None:
        st.sidebar.success(t["load_ok"])
else:
    st.sidebar.warning(t["wait_file"])

# ── Main analysis ─────────────────────────────────────────────────────────────
if df_raw is not None:
    df = df_raw.copy()
    st.sidebar.header(t["param_h"])

    colonnes_dispos = list(df.columns)
    colonne_defaut_idx = next(
        (i for i, c in enumerate(colonnes_dispos) if c.lower() == 'genotype'),
        min(1, len(colonnes_dispos) - 1)
    )
    colonne_choisie = st.sidebar.selectbox(
        t["col_select"], colonnes_dispos, index=colonne_defaut_idx
    )
    df.rename(columns={colonne_choisie: 'genotype'}, inplace=True)

    separateur = st.sidebar.text_input(t["sep"], "_")

    try:
        df['Sex']       = df['genotype'].str.split(separateur).str[0]
        df['Treatment'] = df['genotype'].str.split(separateur).str[1]
    except Exception:
        st.sidebar.error(t["err_parse"])
        st.stop()
        
    # NEW: Toggle for Temporal Normalization (Rate computation)
    normalize_time = st.sidebar.checkbox(
        "⏱️ Normalize by Time (Convert to per-hour rate)", value=False
    )

    # ── FIX 1: Experiment-scoped unique identifiers ───────────────────────────
    df['source_file'] = (
        df['file']
        .str.replace('\\\\', '/', regex=False)
        .str.split('/')
        .str[-1]
    )
    df['animal_id'] = df['source_file'] + '__' + df['RFID'].astype(str)
    df['cage_id']   = df['source_file'] + '__' + df['group'].astype(str)

    n_before = len(df)
    df = df.drop_duplicates(subset=['source_file', 'RFID', 'day'], keep='first').reset_index(drop=True)
    n_true_dups = n_before - len(df)
    if n_true_dups > 0:
        st.sidebar.warning(t["true_dup_warn"].format(n_true_dups))

    n_reused_rfids = (df.groupby('RFID')['source_file'].nunique() > 1).sum()
    if n_reused_rfids > 0:
        st.sidebar.info(t["rfid_reuse_warn"].format(n_reused_rfids))

    groupes_detectes = sorted(df['Treatment'].dropna().unique().tolist())
    sexes_detectes   = sorted(df['Sex'].dropna().unique().tolist())
    phases_detectees = sorted(df['day'].dropna().unique().tolist())

    # ── Cohort selection ──────────────────────────────────────────────────────
    st.sidebar.subheader(t["cohort_h"])
    st.sidebar.caption(t["cohort_help"])
    defaut_cohort = (
        ['SHAM', 'THC', 'CBD']
        if all(g in groupes_detectes for g in ['SHAM', 'THC', 'CBD'])
        else groupes_detectes
    )
    groupes_inclusion = st.sidebar.multiselect(
        "Groupes LMM:", groupes_detectes, default=defaut_cohort
    )
    if not groupes_inclusion:
        st.sidebar.warning("Veuillez sélectionner au moins un groupe.")
        st.stop()

    # ── Derived metric ────────────────────────────────────────────────────────
    if 'CenterZoneTotalLen' in df.columns and 'PeripheryZoneTotalLen' in df.columns:
        df['Center_Periphery_Ratio'] = (
            df['CenterZoneTotalLen'] / (df['PeripheryZoneTotalLen'] + 1e-5)
        )

    colonnes_exclues = [
        'file', 'source_file', 'RFID', 'animal_id', 'group', 'cage_id',
        'day', 'Sex', 'Treatment', 'genotype',
        'strain', 'age', 'sex', 'MinTemp', 'MaxTemp', 'AvgTemp',
        'MinHumi', 'MaxHumi', 'AvgHumi'
    ]
    colonnes_possibles = [
        c for c in df.columns
        if c not in colonnes_exclues and pd.api.types.is_numeric_dtype(df[c])
    ]

    defauts_domaines = {
        'Locomotor_Drive':              ['MoveisolatedTotalLen', 'totalDistance'],
        'Exploratory_Spatial_Strategy': ['Center_Periphery_Ratio', 'RearisolatedTotalLen'],
        'Vigilance_Risk':               ['SAPNb', 'StopisolatedNb'],
        'Social_Tolerance':             ['Group2TotalLen', 'Group3TotalLen',
                                         'SidebysideContactTotalLen',
                                         'SidebysideContact,oppositewayTotalLen'],
        'Active_Social_Engagement':     ['Oral-oralContactTotalLen',
                                         'Oral-genitalContactTotalLen',
                                         'SocialapproachNb', 'Train2TotalLen',
                                         'FollowZoneTotalLen']
    }

    metriques_domaines = {}
    with st.sidebar.expander(t["domain_config_h"], expanded=False):
        for domaine, defauts in defauts_domaines.items():
            defauts_presents = [m for m in defauts if m in colonnes_possibles]
            selection = st.multiselect(
                f"{domaine.replace('_', ' ')}", options=colonnes_possibles,
                default=defauts_presents
            )
            if selection:
                metriques_domaines[domaine] = selection

    # ── Group comparison selectors ────────────────────────────────────────────
    st.sidebar.subheader(t["group_h"])
    idx_sham = groupes_inclusion.index('SHAM') if 'SHAM' in groupes_inclusion else 0
    groupe_controle = st.sidebar.selectbox(t["ctrl"], groupes_inclusion, index=idx_sham)
    autres_groupes  = [g for g in groupes_inclusion if g != groupe_controle]
    groupe_cible    = st.sidebar.selectbox(t["target"], autres_groupes, index=0)

    phases_par_defaut = [
        p for p in ['Exploration', 'Night 1', 'Night 2', 'Night 3']
        if p in phases_detectees
    ]
    phases_ordre = st.sidebar.multiselect(
        t["phase"], phases_detectees, default=phases_par_defaut
    )

    # ── Plot customisation ────────────────────────────────────────────────────
    st.sidebar.header(t["plot_h"])
    largeur_fig    = st.sidebar.slider(t["w"], 10, 24, 16)
    hauteur_fig    = st.sidebar.slider(t["h"], 15, 45, 28)
    taille_etoiles = st.sidebar.slider(t["star"], 8, 24, 12)
    couleur_defaut = "#2ca02c" if groupe_cible == 'CBD' else "#d62728"
    couleur_cible  = st.sidebar.color_picker(t["color"], couleur_defaut)

    toutes_metriques = [m for sous_liste in metriques_domaines.values() for m in sous_liste]

    # ── Run button ────────────────────────────────────────────────────────────
    if st.button(t["run"], type="primary"):
        if not metriques_domaines:
            st.error("Configurez au moins un domaine.")
            st.stop()
        if not phases_ordre:
            st.error("Sélectionnez au moins une phase.")
            st.stop()

        lmm_warnings      = []
        baseline_warnings = []
        lmm_stats_list    = [] # Store fixed/random effects for table

        with st.spinner(t["running"]):

            # ── Subset to selected cohort ─────────────────────────────────────
            df_calcul = df[df['Treatment'].isin(groupes_inclusion)].copy()
            df_calcul['day']     = pd.Categorical(
                df_calcul['day'], categories=phases_ordre, ordered=True
            )
            df_calcul['cage_id']   = df_calcul['cage_id'].astype(str)
            df_calcul['animal_id'] = df_calcul['animal_id'].astype(str)

            # ── NEW: Temporal Normalization ───────────────────────────────────
            if normalize_time:
                def get_duration(phase):
                    phase_str = str(phase).lower()
                    if 'exploration' in phase_str: return 1.0
                    if 'night' in phase_str: return 4.0
                    return 1.0 # Default fallback
                
                durations = df_calcul['day'].apply(get_duration)
                
                for m in toutes_metriques:
                    if any(key in m for key in ['Nb', 'TotalLen', 'totalDistance']):
                        df_calcul[m] = df_calcul[m] / durations

            # ── FIX 2: Drop all-NaN rows instead of filling NaN with 0 ───────
            df_calcul = df_calcul.dropna(subset=toutes_metriques, how='all')

            # ── Step 1: LMM cage correction (Now with Sex Interaction) ────────
            for m in toutes_metriques:
                df_calcul[f'{m}_adj'] = df_calcul[m].copy()
                rows_m = df_calcul.dropna(subset=[m, 'Sex', 'Treatment'])
                if rows_m.empty:
                    continue
                try:
                    # Formulate model: Use Treatment*Sex interaction if >1 sex is present
                    if len(sexes_detectes) > 1:
                        formula = f"Q('{m}') ~ C(Treatment) * C(Sex)"
                    else:
                        formula = f"Q('{m}') ~ C(Treatment)"
                        
                    modele   = smf.mixedlm(formula, rows_m, groups=rows_m['cage_id'])
                    resultat = modele.fit(method='cg', disp=False)
                    re       = resultat.random_effects
                    
                    # Store statistics for the results table
                    for term, pval in resultat.pvalues.items():
                        if term != 'Group Var':
                            lmm_stats_list.append({
                                'Metric': m,
                                'Term': term,
                                'Coefficient': resultat.params[term],
                                'P-Value (Raw)': pval,
                                'Cage Variance (Random Effect)': resultat.cov_re.iloc[0, 0]
                            })

                    df_calcul[f'{m}_adj'] = df_calcul.apply(
                        lambda row: row[m] - re.get(
                            row['cage_id'], pd.Series({'Group': 0})
                        ).get('Group', 0),
                        axis=1
                    )
                except Exception as e:
                    lmm_warnings.append(t["lmm_warn"].format(m, str(e)[:80]))

            # ── Step 2: Sex-stratified, baseline-anchored Z-score ─────────────
            df_z      = df_calcul.copy()
            phase_ref = phases_ordre[0]

            for m in toutes_metriques:
                z_col = f"{m}_z"
                df_z[z_col] = np.nan
                for s in sexes_detectes:
                    masque_base = (
                        (df_z['day']       == phase_ref) &
                        (df_z['Sex']       == s) &
                        (df_z['Treatment'] == groupe_controle)
                    )
                    vals_base = df_z.loc[masque_base, f'{m}_adj'].dropna()
                    if vals_base.empty:
                        continue
                    mu  = vals_base.mean()
                    sig = vals_base.std()
                    if pd.isna(sig) or sig == 0:
                        baseline_warnings.append(
                            t["baseline_warn"].format(m, s, phase_ref)
                        )
                        sig = 1.0
                    mask_sex = df_z['Sex'] == s
                    df_z.loc[mask_sex, z_col] = (
                        (df_z.loc[mask_sex, f'{m}_adj'] - mu) / sig
                    )

            # ── Step 3: Composite domain indices ─────────────────────────────
            for domaine, metriques in metriques_domaines.items():
                cols_z = [f"{m}_z" for m in metriques if f"{m}_z" in df_z.columns]
                if cols_z:
                    df_z[f"{domaine}_Index"] = df_z[cols_z].mean(axis=1)

            indices_finaux = [f"{d}_Index" for d in metriques_domaines.keys()]
            keep_cols = (
                ['cage_id', 'animal_id', 'Sex', 'Treatment', 'day'] + indices_finaux
            )
            df_modele = (
                df_z[[c for c in keep_cols if c in df_z.columns]]
                .dropna()
                .copy()
            )

        # ── Display computation warnings ──────────────────────────────────────
        for w in lmm_warnings:
            st.warning(w)
        for w in baseline_warnings:
            st.warning(w)

        # ── Step 4: LMM Stats Table Display ───────────────────────────────────
        if lmm_stats_list:
            st.header(t["lmm_stats_h"])
            df_lmm_stats = pd.DataFrame(lmm_stats_list)
            # Apply BH-FDR correction to the LMM fixed effect p-values globally
            df_lmm_stats['P-Value (FDR)'] = false_discovery_control(
                df_lmm_stats['P-Value (Raw)'], method='bh'
            )
            st.dataframe(df_lmm_stats.style.format({
                'Coefficient': "{:.3f}",
                'P-Value (Raw)': "{:.4f}",
                'P-Value (FDR)': "{:.4f}",
                'Cage Variance (Random Effect)': "{:.3f}"
            }))
            
        # ── Step 5: Welch t-tests + global BH-FDR correction ─────────────────
        def get_stars(p_adj):
            if p_adj < 0.001: return '***'
            if p_adj < 0.01:  return '**'
            if p_adj < 0.05:  return '*'
            return ''

        df_plot = df_modele[
            df_modele['Treatment'].isin([groupe_controle, groupe_cible])
        ].copy()
        df_plot['Treatment'] = pd.Categorical(
            df_plot['Treatment'],
            categories=[groupe_controle, groupe_cible], ordered=True
        )

        test_results = []
        for nom_indice in indices_finaux:
            for s in sexes_detectes:
                sg = df_plot[df_plot['Sex'] == s]
                for jour in phases_ordre:
                    v_ctrl  = sg[(sg['day'] == jour) & (sg['Treatment'] == groupe_controle)][nom_indice].dropna()
                    v_cible = sg[(sg['day'] == jour) & (sg['Treatment'] == groupe_cible)][nom_indice].dropna()
                    if len(v_ctrl) > 2 and len(v_cible) > 2:
                        _, p_raw = ttest_ind(v_ctrl, v_cible, equal_var=False)
                        test_results.append({
                            'indice':    nom_indice,
                            'sexe':      s,
                            'jour':      jour,
                            'p_raw':     p_raw,
                            'mean_cible': v_cible.mean(),
                            'std_cible':  v_cible.std(),
                            'n_ctrl':    len(v_ctrl),
                            'n_cible':   len(v_cible),
                        })

        if test_results:
            p_raw_arr = np.array([r['p_raw'] for r in test_results])
            p_adj_arr = false_discovery_control(p_raw_arr, method='bh')
            for r, p_adj in zip(test_results, p_adj_arr):
                r['p_adj'] = p_adj
                r['stars'] = get_stars(p_adj)

        annot_lookup = {
            (r['indice'], r['sexe'], r['jour']): r
            for r in test_results
        }

        # ── Plotting ──────────────────────────────────────────────────────────
        st.header(t["res_h"])

        sns.set_theme(style="whitegrid", context="talk")
        n_domains = len(metriques_domaines)
        n_sexes   = len(sexes_detectes)

        fig, axes = plt.subplots(
            n_domains, n_sexes,
            figsize=(largeur_fig, 5 * n_domains),
            sharex=True,
            squeeze=False   
        )

        palette         = {groupe_controle: '#7f7f7f', groupe_cible: couleur_cible}
        etiquettes_sexe = {'M': 'Males', 'F': 'Females'}

        for idx_dom, (domaine_cle, nom_indice) in enumerate(
            zip(metriques_domaines.keys(), indices_finaux)
        ):
            for idx_sx, s in enumerate(sexes_detectes):
                ax = axes[idx_dom, idx_sx]
                sg = df_plot[df_plot['Sex'] == s]

                if not sg.empty:
                    sns.pointplot(
                        data=sg, x='day', y=nom_indice,
                        hue='Treatment', palette=palette,
                        markers=['o', 's'], capsize=.1, ax=ax,
                        dodge=True, errwidth=1.5
                    )

                liste_m = metriques_domaines[domaine_cle]
                mets_fmt = (
                    ", ".join(liste_m[:2]) + " +\n" + ", ".join(liste_m[2:])
                    if len(liste_m) > 3
                    else " +\n".join(liste_m)
                )
                ax.set_title(
                    f"{domaine_cle.replace('_', ' ')} — {etiquettes_sexe.get(s, s)}\n"
                    f"({mets_fmt})",
                    fontsize=11, pad=12
                )
                ax.axhline(0, ls='--', color='black', alpha=0.3, lw=1)
                ax.set_ylabel(t["ylab"] if idx_sx == 0 else '')
                ax.set_xlabel('')
                if ax.get_legend() is not None:
                    ax.get_legend().remove()

                ylo, yhi = ax.get_ylim()
                y_range  = yhi - ylo if yhi != ylo else 1.0

                for idx_jour, jour in enumerate(phases_ordre):
                    rec = annot_lookup.get((nom_indice, s, jour))

                    if rec and rec['stars']:
                        p_adj = rec['p_adj']
                        label = (
                            f"{rec['stars']}\n({t['fdr_label']} p<0.001)"
                            if p_adj < 0.001
                            else f"{rec['stars']}\n({t['fdr_label']} p={p_adj:.3f})"
                        )
                        y_annot = rec['mean_cible'] + rec['std_cible'] * 0.4 + 0.3
                        ax.text(
                            idx_jour, y_annot, label,
                            color=couleur_cible, ha='center', va='bottom',
                            fontweight='bold', fontsize=taille_etoiles
                        )

                    if rec:
                        ax.text(
                            idx_jour,
                            ylo + 0.03 * y_range,
                            f"{t['n_label']}{rec['n_ctrl']}/{rec['n_cible']}",
                            ha='center', va='bottom', fontsize=7, color='#555555'
                        )

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, title="Treatment",
            loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=2
        )
        plt.tight_layout()
        fig.subplots_adjust(top=0.94, hspace=0.50)
        st.pyplot(fig)

        # ── Export ────────────────────────────────────────────────────────────
        st.header(t["exp_h"])
        col1, col2 = st.columns(2)

        buf_img = io.BytesIO()
        fig.savefig(buf_img, format='png', bbox_inches='tight', dpi=300)
        buf_img.seek(0)
        col1.download_button(
            label=t["dl_fig"], data=buf_img,
            file_name="LMT_Figure.png", mime="image/png"
        )

        buf_csv = io.StringIO()
        df_plot.to_csv(buf_csv, index=False)
        col2.download_button(
            label=t["dl_csv"], data=buf_csv.getvalue(),
            file_name="LMT_Data_Corrected.csv", mime="text/csv"
        )
