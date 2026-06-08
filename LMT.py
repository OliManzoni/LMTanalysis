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

        "rationale_md": """

# 🧠 Pipeline LMT — Guide Complet pour Étudiants


> **À qui s'adresse ce guide ?** À tout étudiant en master ou doctorat qui utilise le Live Mouse Tracker pour la première fois et qui veut comprendre *pourquoi* chaque étape existe, pas seulement *comment* cliquer.


---


## 🐭 Contexte : Qu'est-ce que le Live Mouse Tracker ?


Le **Live Mouse Tracker (LMT)** est un système automatisé qui enregistre en continu le comportement de groupes de souris (généralement 3–4 animaux par cage) grâce à des **puces RFID** sous-cutanées et une caméra infrarouge. Contrairement à l'observation manuelle, il génère des centaines de métriques comportementales sur des nuits entières, sans biais observateur.


Chaque ligne du fichier de données représente **un animal × une phase temporelle** (ex : Nuit 1). Les colonnes sont des métriques comme `MoveisolatedTotalLen` (distance parcourue seul), `Group2TotalLen` (temps passé en groupe de 2), ou `SAPNb` (nombre de postures d'auto-soin).


---


## 🛠️ PARTIE 1 : Mode d'Emploi Pas-à-Pas


### Étape 1 — Préparer et importer votre fichier


**Format attendu :** Fichier `.txt` ou `.csv` exporté depuis le logiciel LMT, avec un séparateur tabulation (détecté automatiquement).


**Convention de nommage obligatoire pour la colonne `genotype` :**

La colonne doit combiner le **sexe** et le **traitement** séparés par un caractère (par défaut `_`) :


| Valeur dans la colonne | Sexe détecté | Traitement détecté |

|---|---|---|

| `M_SHAM` | M (mâle) | SHAM |

| `F_THC` | F (femelle) | THC |

| `M_CBD` | M (mâle) | CBD |


> ⚠️ **Erreur fréquente :** Si vous avez nommé vos groupes `THC_M` au lieu de `M_THC`, le pipeline interprétera `THC` comme le sexe. Vérifiez l'ordre *avant* d'importer.


---


### Étape 2 — Comprendre la gestion des puces RFID


Le LMT identifie chaque souris par son **numéro de puce RFID**. Ces puces sont **réutilisées entre cohortes** (raisons économiques et pratiques). Cela signifie que le numéro `2029600582` dans le fichier de juin 2025 et le même numéro dans le fichier de juillet 2025 désignent **deux animaux biologiquement différents**.


Le pipeline résout cela en construisant des identifiants scopés au fichier source :


```

animal_id = nom_du_fichier.sqlite + "__" + numéro_RFID

cage_id   = nom_du_fichier.sqlite + "__" + numéro_de_cage

```


**Pourquoi la cage aussi ?** Parce que le numéro de cage dans le LMT correspond souvent au RFID de l'animal "alpha" de la cage — qui peut, lui aussi, être réutilisé. Sans ce scopage, deux cohortes différentes pourraient être fusionnées dans le même niveau d'effet aléatoire, corrompant silencieusement le modèle statistique.


> 💡 **Si vous voyez un message ℹ️ "X puces RFID réutilisées"** dans la barre latérale : c'est normal et attendu. Le pipeline l'a géré. Si vous voyez un ⚠️ "lignes dupliquées supprimées" : cela signifie qu'un même fichier a été exporté deux fois — vérifiez vos sources.


---


### Étape 3 — Configurer la cohorte globale pour le modèle LMM


**Règle absolue : incluez TOUS vos groupes expérimentaux ici** (SHAM, THC, CBD, Naïf…).


**Pourquoi ?** Le modèle mixte linéaire (LMM) doit estimer l'effet de cage (variance inter-portée) à partir de l'ensemble de la cohorte. Si vous excluez un groupe, vous réduisez le nombre de cages disponibles pour cette estimation, ce qui dégrade la précision de la correction statistique pour *tous* les groupes restants.


> 🎓 **Analogie pédagogique :** Imaginez que vous voulez mesurer la taille moyenne des étudiants d'une promo, en corrigeant pour l'effet "famille" (les frères/sœurs sont naturellement plus proches en taille). Si vous n'incluez que les étudiants blonds, votre estimation de la variance familiale sera biaisée et imprécise. Il faut toute la promo.


---


### Étape 4 — Définir les domaines comportementaux


Les **domaines composites** regroupent des métriques individuelles en indices biologiquement cohérents. Les 5 domaines par défaut sont :


| Domaine | Métriques clés | Ce qu'il mesure |

|---|---|---|

| **Locomotor Drive** | Distance totale, déplacement isolé | Niveau d'activité motrice général |

| **Exploratory Spatial Strategy** | Ratio centre/périphérie, redressements | Audace spatiale, exploration thigmotaxique |

| **Vigilance & Risk** | Postures SAP, arrêts | Hypervigilance, comportement défensif |

| **Social Tolerance** | Temps en groupe 2–3, contacts côte-à-côte | Proximité passive, tolérance sociale |

| **Active Social Engagement** | Contacts oro-génitaux, approches, trains | Interactions sociales actives et dirigées |


Vous pouvez ajouter ou retirer des métriques à la volée. En cas de doute, commencez avec les défauts.


---


### Étape 5 — Lancer l'analyse et lire les résultats


**Choisissez :**

- **Groupe contrôle** : c'est votre référence biologique (généralement SHAM ou WT). Son comportement à la **première phase** définira Z=0 pour tous les groupes.

- **Groupe cible** : le groupe dont vous voulez caractériser l'écart au contrôle.

- **Phases** : l'ordre des phases sur l'axe X (ex : Exploration → Nuit 1 → Nuit 2 → Nuit 3).


**Comment lire le graphique :**


```

Z-score > 0  →  le groupe est AU-DESSUS du niveau d'éveil initial du contrôle

Z-score = 0  →  comportement identique à l'état d'éveil aigu du contrôle

Z-score < 0  →  le groupe est EN-DESSOUS (hypoactivité, sédation, apathie)

```


Une courbe contrôle qui **descend progressivement** de Z≈0 vers Z négatif représente **l'habituation normale** : l'animal explore moins au fil des nuits car l'environnement devient familier. C'est de la **plasticité comportementale saine**.


Une courbe mutant/traité qui **reste proche de Z=0** alors que le contrôle s'habitue signe une **rigidité comportementale** (incapacité à s'habituer). Une courbe qui **s'effondre brutalement dans le négatif dès la 1ère nuit** suggère une **sédation** ou un déficit d'éveil.


**Les étoiles (*, **, ***)** indiquent des différences significatives entre les groupes à ce point temporel, après correction statistique pour tests multiples (voir Partie 2).


---


## 🔬 PARTIE 2 : Fondements Mathématiques


### Étape A — Correction de la pseudo-réplication par modèle mixte (LMM)


**Le problème :** Dans une cage LMT, 3 souris cohabitent. Leur comportement est statistiquement **non-indépendant** : une souris hyperactive va mécaniquement augmenter les interactions sociales de ses cagemates. Si on traite les 3 animaux comme indépendants, on **triple artificiellement la taille apparente de l'échantillon** (pseudo-réplication), ce qui gonfle la significativité statistique.


**La solution — modèle mixte :**


$$Y_{ijk} = \\beta_0 + \\beta_1 \\cdot \\text{Treatment}_i + \\gamma_j + \\varepsilon_{ijk}$$


| Terme | Signification |

|---|---|

| $Y_{ijk}$ | Valeur brute de la métrique pour l'animal $k$, du traitement $i$, dans la cage $j$ |

| $\\beta_0$ | Intercept global (valeur moyenne de référence) |

| $\\beta_1 \\cdot \\text{Treatment}_i$ | Effet fixe du traitement (ce qu'on veut mesurer) |

| $\\gamma_j$ | Effet aléatoire de la cage $j$ — estimé et soustrait |

| $\\varepsilon_{ijk}$ | Résidu individuel (variance non expliquée) |


Le comportement **résiduel corrigé** devient :


$$Y_{adj} = Y_{ijk} - \\hat{\\gamma}_j$$


> 🎓 **En clair :** Le modèle estime "combien cette cage, indépendamment du traitement, pousse les animaux à être plus ou moins actifs" et soustrait cette contribution. Ce qui reste reflète l'effet du traitement, pas l'effet de la cage.


---


### Étape B — Standardisation par Z-score ancré sur la baseline


**Pourquoi standardiser ?** Les métriques LMT ont des échelles très différentes (`totalDistance` est en centimètres, `SAPNb` est un comptage d'événements). Pour les combiner en domaines composites, elles doivent être sur une échelle commune.


**La formule :**


$$Z_{ijk} = \\frac{Y_{adj,ijk} - \\mu_{ctrl,\\, t_0}}{\\sigma_{ctrl,\\, t_0}}$$


où $\\mu_{ctrl,\\, t_0}$ et $\\sigma_{ctrl,\\, t_0}$ sont la **moyenne** et l'**écart-type du groupe contrôle à la première phase uniquement** (phase de référence), calculés séparément par sexe.


**Pourquoi cette baseline précise ?**

- La **première phase** (souvent "Exploration") correspond à l'introduction des animaux dans un environnement nouveau. C'est le moment où l'éveil comportemental est **maximal et uniforme** entre les groupes — avant que les effets du traitement ne se manifestent pleinement.

- Ancrer Z=0 sur cet état garantit que toutes les trajectoires ultérieures sont interprétables comme des **déviations par rapport à la réponse normale à la nouveauté**.


**Calcul séparé par sexe :** Les mâles et les femelles ont des niveaux d'activité basaux différents. Calculer la baseline séparément par sexe évite que la différence biologique male/femelle n'absorbe une partie du signal traitement.


---


### Étape C — Agrégation en domaines composites


Pour chaque domaine, on calcule la **moyenne des Z-scores** de ses métriques constitutives :


$$\\text{Domain\\_Index} = \\frac{1}{M} \\sum_{m=1}^{M} Z_m$$


Comme chaque $Z_m$ a une moyenne de 0 et un écart-type de 1 dans le groupe contrôle à la baseline, cette moyenne est mathématiquement valide — les métriques sont sur la même échelle avant d'être sommées.


**Avantage :** La variance de la moyenne de $M$ variables décorrélées diminue comme $1/M$, ce qui **améliore le ratio signal/bruit** par rapport à l'analyse de chaque métrique isolément.


---


### Étape D — Tests statistiques et correction pour tests multiples


**Test de Welch :** Pour chaque point temporel, on compare les deux groupes avec un test t de Welch (qui ne suppose **pas** l'égalité des variances entre groupes, contrairement au test t classique). C'est le test approprié en neurosciences comportementales où les groupes mutants ont souvent une variance très différente des contrôles.


**Le problème des tests multiples :** Avec 5 domaines × 2 sexes × 4 phases = **40 tests simultanés**, la probabilité d'obtenir au moins un faux positif par hasard est de $1 - 0.95^{40} \\approx 87\\%$ — ce qui est inacceptable.


**La solution — correction de Benjamini-Hochberg (FDR) :**


Plutôt que d'utiliser une correction de Bonferroni (trop conservative, qui divise α par le nombre de tests), on utilise le **False Discovery Rate (FDR)** :


1. Trier toutes les p-values brutes de la plus petite à la plus grande : $p_{(1)} \\leq p_{(2)} \\leq \\ldots \\leq p_{(m)}$

2. Chaque p-value ajustée est : $p_{adj,(i)} = p_{(i)} \\times \\frac{m}{i}$

3. Un test est déclaré significatif si $p_{adj} < 0.05$


**Interprétation :** Avec FDR=5%, on accepte que parmi tous les effets déclarés significatifs, **au maximum 5% soient de faux positifs**. C'est beaucoup plus puissant que Bonferroni tout en contrôlant rigoureusement l'inflation des erreurs de type I.


> ⚠️ **Erreur fréquente chez les étudiants :** Appliquer le test de Welch sur un seul point temporel en ignorant les autres. Cela revient à faire 40 paris en ne comptant que ceux qu'on gagne. **Toujours corriger pour l'ensemble des tests de la figure.**


---


## 📊 Glossaire Rapide


| Terme | Définition simple |

|---|---|

| **LMM** | Modèle mixte linéaire — sépare l'effet "cage" de l'effet "traitement" |

| **Effet aléatoire** | Variation due à un facteur de groupement (la cage) qu'on veut retirer |

| **Effet fixe** | L'effet qu'on veut mesurer (le traitement) |

| **Z-score** | Nombre d'écarts-types au-dessus/en-dessous de la moyenne de référence |

| **Pseudo-réplication** | Traiter des animaux de la même cage comme s'ils étaient indépendants |

| **FDR** | Taux de faux positifs contrôlé parmi les tests déclarés significatifs |

| **BH** | Benjamini-Hochberg, la procédure de correction FDR utilisée ici |

| **Thigmotaxie** | Tendance à rester proche des murs — signe d'anxiété chez la souris |

| **SAP** | Stretched Attend Posture — posture de surveillance du risque |

        """,

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

        "rationale_md": """

# 🧠 LMT Pipeline — Complete Student Guide


> **Who is this guide for?** Any Master's or PhD student using the Live Mouse Tracker for the first time who wants to understand *why* each step exists, not just *how* to click through it.


---


## 🐭 Background: What is the Live Mouse Tracker?


The **Live Mouse Tracker (LMT)** is an automated system that continuously records the behaviour of groups of mice (typically 3–4 animals per cage) using subcutaneous **RFID chips** and an infrared camera. Unlike manual observation, it generates hundreds of behavioural metrics across entire nights with no observer bias.


Each row in the data file represents **one animal × one temporal phase** (e.g. Night 1). Columns are metrics such as `MoveisolatedTotalLen` (distance travelled alone), `Group2TotalLen` (time spent in a group of 2), or `SAPNb` (number of stretched attend postures).


---


## 🛠️ PART 1: Step-by-Step Manual


### Step 1 — Prepare and import your file


**Expected format:** `.txt` or `.csv` exported from the LMT software, with a tab separator (auto-detected).


**Mandatory naming convention for the `genotype` column:**

The column must combine **sex** and **treatment** separated by a character (default `_`):


| Value in column | Sex detected | Treatment detected |

|---|---|---|

| `M_SHAM` | M (male) | SHAM |

| `F_THC` | F (female) | THC |

| `M_CBD` | M (male) | CBD |


> ⚠️ **Common mistake:** If your groups are named `THC_M` instead of `M_THC`, the pipeline will interpret `THC` as the sex. Check the order *before* importing.


---


### Step 2 — Understanding RFID chip management


The LMT identifies each mouse by its **RFID chip number**. These chips are **reused across cohorts** (for economic and practical reasons). This means that chip number `2029600582` in a June 2025 file and the same number in a July 2025 file refer to **two biologically different animals**.


The pipeline resolves this by building source-file-scoped identifiers:


```

animal_id = source_filename.sqlite + "__" + RFID_number

cage_id   = source_filename.sqlite + "__" + cage_number

```


**Why the cage too?** Because the cage number in LMT often corresponds to the RFID of the "alpha" animal of the cage — which can also be reused. Without this scoping, two different cohorts could be merged into the same random-effect level, silently corrupting the statistical model.


> 💡 **If you see an ℹ️ message "X RFID chips reused"** in the sidebar: this is normal and expected. The pipeline has handled it. If you see a ⚠️ "duplicate rows removed": the same file was exported twice — check your sources.


---


### Step 3 — Configure the global cohort for the LMM


**Absolute rule: include ALL your experimental groups here** (SHAM, THC, CBD, Naïve…).


**Why?** The linear mixed model (LMM) must estimate the cage effect (inter-litter variance) from the whole cohort. If you exclude a group, you reduce the number of cages available for this estimation, degrading the precision of the statistical correction for *all* remaining groups.


> 🎓 **Pedagogical analogy:** Imagine you want to measure the average height of students in a class, correcting for the "family" effect (siblings are naturally more similar in height). If you only include blonde students, your estimate of family variance will be biased and imprecise. You need the whole class.


---


### Step 4 — Define behavioural domains


**Composite domains** group individual metrics into biologically coherent indices. The 5 default domains are:


| Domain | Key metrics | What it measures |

|---|---|---|

| **Locomotor Drive** | Total distance, isolated movement | General motor activity level |

| **Exploratory Spatial Strategy** | Centre/periphery ratio, rearings | Spatial boldness, thigmotaxic exploration |

| **Vigilance & Risk** | SAP postures, stops | Hypervigilance, defensive behaviour |

| **Social Tolerance** | Time in groups of 2–3, side-by-side contacts | Passive proximity, social tolerance |

| **Active Social Engagement** | Oro-genital contacts, approaches, trains | Active and directed social interactions |


You can add or remove metrics on the fly. When in doubt, start with the defaults.


---


### Step 5 — Run the analysis and read the results


**Choose:**

- **Control group**: your biological reference (usually SHAM or WT). Its behaviour at the **first phase** will define Z=0 for all groups.

- **Target group**: the group whose deviation from control you want to characterise.

- **Phases**: the order of phases on the X axis (e.g. Exploration → Night 1 → Night 2 → Night 3).


**How to read the plot:**


```

Z-score > 0  →  the group is ABOVE the control's initial arousal level

Z-score = 0  →  behaviour identical to the control's acute arousal state

Z-score < 0  →  the group is BELOW (hypoactivity, sedation, apathy)

```


A control curve that **progressively descends** from Z≈0 towards negative values represents **normal habituation**: the animal explores less over successive nights because the environment becomes familiar. This is **healthy behavioural plasticity**.


A mutant/treated curve that **stays near Z=0** while the control habituates signals **behavioural rigidity** (inability to habituate). A curve that **collapses sharply into negative values from Night 1** suggests **sedation** or an arousal deficit.


**Stars (*, **, ***)** indicate significant differences between groups at that timepoint, after statistical correction for multiple comparisons (see Part 2).


---


## 🔬 PART 2: Mathematical Foundations


### Step A — Pseudo-replication correction via mixed model (LMM)


**The problem:** In an LMT cage, 3 mice cohabit. Their behaviour is statistically **non-independent**: a hyperactive mouse will mechanically increase the social interactions of its cagemates. If we treat the 3 animals as independent, we **artificially triple the apparent sample size** (pseudo-replication), inflating statistical significance.


**The solution — mixed model:**


$$Y_{ijk} = \\beta_0 + \\beta_1 \\cdot \\text{Treatment}_i + \\gamma_j + \\varepsilon_{ijk}$$


| Term | Meaning |

|---|---|

| $Y_{ijk}$ | Raw metric value for animal $k$, treatment $i$, cage $j$ |

| $\\beta_0$ | Global intercept (average reference value) |

| $\\beta_1 \\cdot \\text{Treatment}_i$ | Fixed effect of treatment (what we want to measure) |

| $\\gamma_j$ | Random effect of cage $j$ — estimated and subtracted |

| $\\varepsilon_{ijk}$ | Individual residual (unexplained variance) |


The **corrected residual behaviour** becomes:


$$Y_{adj} = Y_{ijk} - \\hat{\\gamma}_j$$


> 🎓 **In plain terms:** The model estimates "how much this cage, independently of treatment, pushes animals to be more or less active" and subtracts that contribution. What remains reflects the treatment effect, not the cage effect.


---


### Step B — Baseline-anchored Z-score standardisation


**Why standardise?** LMT metrics have very different scales (`totalDistance` is in centimetres, `SAPNb` is an event count). To combine them into composite domains, they must share a common scale.


**The formula:**


$$Z_{ijk} = \\frac{Y_{adj,ijk} - \\mu_{ctrl,\\, t_0}}{\\sigma_{ctrl,\\, t_0}}$$


where $\\mu_{ctrl,\\, t_0}$ and $\\sigma_{ctrl,\\, t_0}$ are the **mean and standard deviation of the control group at the first phase only** (reference phase), computed separately per sex.


**Why this specific baseline?**

- The **first phase** (usually "Exploration") corresponds to introducing the animals into a novel environment. This is when behavioural arousal is **maximal and uniform** across groups — before treatment effects have had time to fully manifest.

- Anchoring Z=0 to this state ensures that all subsequent trajectories are interpretable as **deviations from the normal response to novelty**.


**Computed separately per sex:** Males and females have different basal activity levels. Computing the baseline separately per sex prevents the biological sex difference from absorbing part of the treatment signal.


---


### Step C — Aggregation into composite domains


For each domain, we compute the **mean of the Z-scores** of its constituent metrics:


$$\\text{Domain\\_Index} = \\frac{1}{M} \\sum_{m=1}^{M} Z_m$$


Since each $Z_m$ has mean 0 and SD 1 in the control group at baseline, this average is mathematically valid — the metrics are on the same scale before being summed.


**Advantage:** The variance of the mean of $M$ decorrelated variables decreases as $1/M$, which **improves the signal-to-noise ratio** compared to analysing each metric in isolation.


---


### Step D — Statistical tests and multiple comparisons correction


**Welch's t-test:** At each timepoint, the two groups are compared with a Welch t-test (which does **not** assume equal variances between groups, unlike the classic t-test). This is the appropriate test in behavioural neuroscience where mutant groups often show very different variance from controls.


**The multiple comparisons problem:** With 5 domains × 2 sexes × 4 phases = **40 simultaneous tests**, the probability of obtaining at least one false positive by chance is $1 - 0.95^{40} \\approx 87\\%$ — which is unacceptable.


**The solution — Benjamini-Hochberg FDR correction:**


Rather than Bonferroni correction (too conservative — divides α by the number of tests), we use **False Discovery Rate (FDR)**:


1. Sort all raw p-values from smallest to largest: $p_{(1)} \\leq p_{(2)} \\leq \\ldots \\leq p_{(m)}$

2. Each adjusted p-value is: $p_{adj,(i)} = p_{(i)} \\times \\frac{m}{i}$

3. A test is declared significant if $p_{adj} < 0.05$


**Interpretation:** With FDR=5%, we accept that among all effects declared significant, **at most 5% are false positives**. This is far more powerful than Bonferroni while still rigorously controlling Type I error inflation.


> ⚠️ **Common student mistake:** Applying Welch's test to a single timepoint while ignoring the others. This is equivalent to placing 40 bets and only counting the wins. **Always correct for the full set of tests in the figure.**


---


## 📊 Quick Glossary


| Term | Simple definition |

|---|---|

| **LMM** | Linear Mixed Model — separates the "cage" effect from the "treatment" effect |

| **Random effect** | Variation due to a grouping factor (the cage) that we want to remove |

| **Fixed effect** | The effect we want to measure (the treatment) |

| **Z-score** | Number of standard deviations above/below the reference mean |

| **Pseudo-replication** | Treating animals from the same cage as if they were independent |

| **FDR** | False Discovery Rate — controls the proportion of false positives among significant results |

| **BH** | Benjamini-Hochberg, the FDR correction procedure used here |

| **Thigmotaxis** | Tendency to stay close to walls — a sign of anxiety in mice |

| **SAP** | Stretched Attend Posture — a risk-assessment surveillance posture |

        """,

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


    # ── FIX 1: Experiment-scoped unique identifiers ───────────────────────────

    #

    # RFID chips and cage/group numbers are REUSED across cohorts in LMT.

    # The same RFID integer in two different source SQLite files refers to

    # two completely different animals.  The raw `group` column has the same

    # reuse problem for cage IDs.

    #

    # Solution: scope both identifiers to the source file, producing:

    #   animal_id = source_filename + "__" + RFID   (unique per animal)

    #   cage_id   = source_filename + "__" + group  (unique per cage/litter)

    #

    # These replace the raw RFID and group columns everywhere downstream

    # (LMM grouping, Z-score loops, export).

    df['source_file'] = (

        df['file']

        .str.replace('\\\\', '/', regex=False)

        .str.split('/')

        .str[-1]

    )

    df['animal_id'] = df['source_file'] + '__' + df['RFID'].astype(str)

    df['cage_id']   = df['source_file'] + '__' + df['group'].astype(str)


    # Warn about genuine exact duplicates (same file + RFID + day exported twice)

    n_before = len(df)

    df = df.drop_duplicates(subset=['source_file', 'RFID', 'day'], keep='first').reset_index(drop=True)

    n_true_dups = n_before - len(df)

    if n_true_dups > 0:

        st.sidebar.warning(t["true_dup_warn"].format(n_true_dups))


    # Inform about RFID reuse across files (expected, not an error)

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


        with st.spinner(t["running"]):


            # ── Subset to selected cohort ─────────────────────────────────────

            df_calcul = df[df['Treatment'].isin(groupes_inclusion)].copy()

            df_calcul['day']     = pd.Categorical(

                df_calcul['day'], categories=phases_ordre, ordered=True

            )

            # Use cage_id as the LMM grouping variable (experiment-scoped cage)

            # and animal_id for all per-animal operations.

            df_calcul['cage_id']   = df_calcul['cage_id'].astype(str)

            df_calcul['animal_id'] = df_calcul['animal_id'].astype(str)


            # ── FIX 2: Drop all-NaN rows instead of filling NaN with 0 ───────

            # Imputing NaN as 0 conflates "missing observation" with "zero

            # behaviour" and biases LMM intercepts and Z-score baselines.

            df_calcul = df_calcul.dropna(subset=toutes_metriques, how='all')


            # ── Step 1: LMM cage correction ───────────────────────────────────

            # Model: metric ~ C(Treatment), random intercept per cage_id.

            # cage_id is experiment-scoped so reused cage numbers across

            # cohorts are treated as separate random-effect levels.

            #

            # The random_effects dict from statsmodels MixedLM uses the key

            # 'Group' for a simple random intercept (no custom exog_re).

            # We use .get() with a zero fallback so singleton cages that were

            # dropped during model fitting don't raise KeyError.

            #

            # FIX 3: Explicit error handling — failures surface as warnings

            # rather than being swallowed by bare except:pass.

            for m in toutes_metriques:

                df_calcul[f'{m}_adj'] = df_calcul[m].copy()

                rows_m = df_calcul.dropna(subset=[m])

                if rows_m.empty:

                    continue

                try:

                    modele   = smf.mixedlm(

                        f"Q('{m}') ~ C(Treatment)",

                        rows_m,

                        groups=rows_m['cage_id']

                    )

                    resultat = modele.fit(method='cg', disp=False)

                    re       = resultat.random_effects

                    df_calcul[f'{m}_adj'] = df_calcul.apply(

                        lambda row: row[m] - re.get(

                            row['cage_id'], pd.Series({'Group': 0})

                        ).get('Group', 0),

                        axis=1

                    )

                except Exception as e:

                    lmm_warnings.append(t["lmm_warn"].format(m, str(e)[:80]))


            # ── Step 2: Sex-stratified, baseline-anchored Z-score ─────────────

            # Anchor = mean and SD of control group at the first selected phase,

            # computed separately per sex to remove sex-level baseline offsets.

            # FIX 4: Visible warning when baseline SD is 0 or NaN.

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


        # ── Step 4: Welch t-tests + global BH-FDR correction ─────────────────

        # FIX 5: Collect ALL raw p-values across domains × sexes × phases,

        # apply Benjamini-Hochberg FDR correction globally, then annotate.

        # This prevents family-wise error inflation from 40+ simultaneous tests.


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


        # Collect raw p-values first

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


        # Apply BH-FDR globally across all tests

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

            squeeze=False   # always 2-D array regardless of shape

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


                # Annotate significant timepoints (BH-corrected)

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


                    # Sample sizes at bottom of each timepoint

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
