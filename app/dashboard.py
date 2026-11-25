"""Steam Lakehouse Analytics : dashboard décisionnel sur la couche Gold.

Lit UNIQUEMENT les tables Delta de `/lakehouse/gold/` produites par les notebooks J3 (Spark SQL) et J4
(tests statistiques + K-Means Spark MLlib). Ces tables sont petites (20 à 3 438 lignes) : elles sont lues
une fois par Spark (driver 512 Mo), converties en pandas et gardées en cache.
"""

import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyspark.errors import AnalysisException
from pyspark.sql import SparkSession

# Avertissements internes de PySpark 4.2 face à pandas 3 (conversion toPandas) : bruit sans impact ici
warnings.filterwarnings("ignore", module=r"pyspark\..*")

# ======================================================================================================
# Configuration
# ======================================================================================================
GOLD_TABLES = {
    "pareto": "/lakehouse/gold/genre_pareto",
    "trends": "/lakehouse/gold/release_year_trends",
    "top": "/lakehouse/gold/top_games_by_genre",
    "games": "/lakehouse/gold/game_segments",
}
CACHE_TTL_S = 600  # les tables Gold ne changent qu'à la ré-exécution des notebooks (bouton « Rafraîchir »)
SNAPSHOT_YEAR = 2019
ENGAGED_MINUTES = 227  # médiane du temps de jeu médian (seuil « engagé » des profils de vitalité, J4)
OLD_GAME_YEARS = 5

SEGMENT_ORDER = [
    "Hits premium à forte rétention",
    "Anciens classiques du catalogue",
    "Pépites de niche appréciées",
    "Free-to-play à faible rétention",
    "Jeux décevants & controversés",
]
SEGMENT_PLAYBOOK = {
    "Hits premium à forte rétention": (
        "Moteur du marché : la majorité des avis, des joueurs engagés et un prix premium accepté. "
        "Priorité : prolonger la durée de vie (contenus additionnels, saisons, communauté)."
    ),
    "Anciens classiques du catalogue": (
        "Fonds de catalogue bien noté mais peu rejoué aujourd'hui. Priorité : remises saisonnières, "
        "bundles et remasters pour réactiver une base de joueurs déjà acquise."
    ),
    "Pépites de niche appréciées": (
        "Très bien notés mais peu visibles (peu d'avis). Priorité : mise en avant éditoriale, curation "
        "et listes de souhaits pour convertir la satisfaction en volume."
    ),
    "Free-to-play à faible rétention": (
        "Beaucoup de joueurs, peu de temps de jeu. Priorité : améliorer la prise en main et la boucle de "
        "jeu des premières heures ; les avis négatifs y sont souvent jugés utiles."
    ),
    "Jeux décevants & controversés": (
        "Satisfaction minoritaire et avis polarisés. Priorité : corriger les défauts relevés (patchs, "
        "communication) avant toute dépense d'acquisition ; risque réputationnel."
    ),
}
VITALITY_ORDER = ["Classique durable", "Hit récent", "Vite abandonné / jeu mort", "Niche confidentielle"]
PLAYTIME_ORDER = ["< 1 h", "1-3 h", "3-10 h", "10-30 h", ">= 30 h"]
MODEL_FEATURES = {  # feature du K-Means J4 -> (libellé, transformation log1p ?)
    "n_reviews": ("Volume d'avis", True),
    "positive_ratio": ("Satisfaction", False),
    "median_playtime": ("Temps de jeu", True),
    "price": ("Prix", True),
    "game_age": ("Ancienneté", True),
}

# Palettes validées (séparation daltonisme, contraste) pour la surface sombre Steam et pour un thème clair
PALETTES = {
    "dark": {
        "segments": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "series": ["#3987e5", "#d95926", "#199e70"],
        "pareto": {"A": "#9ec5f4", "B": "#3987e5", "C": "#1c5cab"},
        "accent": "#66c0f4",
        "muted": "#4b5d6e",
        "pos": "#3987e5",
        "neg": "#e66767",
        "template": "plotly_dark",
        "grid": "rgba(199, 213, 224, 0.12)",
        "font": "#c7d5e0",
    },
    "light": {
        "segments": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "pareto": {"A": "#184f95", "B": "#3987e5", "C": "#86b6ef"},
        "accent": "#2a78d6",
        "muted": "#c3c2b7",
        "pos": "#2a78d6",
        "neg": "#e34948",
        "template": "plotly_white",
        "grid": "rgba(11, 11, 11, 0.08)",
        "font": "#0b0b0b",
    },
}

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem;}
[data-testid="stMetric"] {
    background: rgba(102, 192, 244, 0.06);
    border: 1px solid rgba(102, 192, 244, 0.22);
    border-radius: 12px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] p {font-size: 0.85rem; opacity: 0.85;}
[data-testid="stMetricValue"] {font-weight: 700;}
.hero-badges span {
    display: inline-block; margin: 0 6px 6px 0; padding: 2px 10px; border-radius: 999px;
    border: 1px solid rgba(102, 192, 244, 0.35); font-size: 0.8rem;
}
.insight {
    border-left: 3px solid #66c0f4; background: rgba(102, 192, 244, 0.07);
    padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 4px 0 18px 0; font-size: 0.93rem;
}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {padding: 8px 14px;}
</style>
"""

st.set_page_config(page_title="Steam Lakehouse Analytics", page_icon="🎮", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
# Thème imposé par la configuration (docker-compose : STREAMLIT_THEME_BASE=dark), sinon celui du navigateur
_theme_base = st.get_option("theme.base") or getattr(st.context.theme, "type", None)
THEME = "light" if _theme_base == "light" else "dark"
PAL = PALETTES[THEME]
SEGMENT_COLORS = dict(zip(SEGMENT_ORDER, PAL["segments"]))


# ======================================================================================================
# Accès aux données (couche Gold uniquement)
# ======================================================================================================
@st.cache_resource
def get_spark():
    return (
        SparkSession.builder.appName("StreamlitGoldReader")
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.shuffle.partitions", "4")
        # Petites tables : 1 partition pour reconstruire l'état du journal Delta (50 par défaut)
        .config("spark.databricks.delta.snapshotPartitions", "1")
        .getOrCreate()
    )


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def load_gold_table(hdfs_path: str) -> pd.DataFrame:
    spark = get_spark()
    return spark.read.format("delta").load(hdfs_path).toPandas()


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def model_space(games: pd.DataFrame) -> pd.DataFrame:
    """Espace du K-Means J4 : log1p des features asymétriques puis centrage-réduction (écart-type n-1)."""
    x = pd.DataFrame(
        {f: np.log1p(games[f]) if use_log else games[f] for f, (_, use_log) in MODEL_FEATURES.items()}
    )
    return (x - x.mean()) / x.std()


def fmt_int(n: float) -> str:
    return f"{n:,.0f}".replace(",", "\u202f")


def fmt_millions(n: float) -> str:
    return f"{num(n / 1e6, 2)} M"


def num(x: float, d: int = 1, sign: bool = False) -> str:
    """Nombre au format français (virgule décimale)."""
    return f"{x:{'+' if sign else ''}.{d}f}".replace(".", ",")


def pct(x: float, d: int = 1) -> str:
    return f"{num(x * 100, d)}\u202f%"


def wavg(df: pd.DataFrame, col: str) -> float:
    """Moyenne d'un taux par jeu pondérée par le nombre d'avis (= taux sur l'ensemble des avis)."""
    n = df["n_reviews"].sum()
    return float((df[col] * df["n_reviews"]).sum() / n) if n else float("nan")


def tier_label(tier: str) -> str:
    return tier.split(". ", 1)[-1]


def insight(text: str) -> None:
    """Encadré « éclairage métier » : le Markdown simple (gras, code) est converti en HTML."""
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
    st.markdown(f'<div class="insight">💡 {html}</div>', unsafe_allow_html=True)


def delta(value: float, unit: str, ref: str, d: int = 1) -> str | None:
    """Écart signé au format français ; None si l'écart arrondi est nul (pas de flèche trompeuse)."""
    return None if round(value, d) == 0 else f"{num(value, d, sign=True)} {unit} {ref}"


def explain(sql: str, justification: str) -> None:
    with st.expander("Requête SQL & Justification statistique"):
        st.code(sql.strip(), language="sql")
        st.markdown(justification)


def show(fig: go.Figure, height: int = 420, **layout) -> None:
    fig.update_layout(
        template=PAL["template"],
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": PAL["font"], "size": 13},
        margin={"l": 10, "r": 20, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": None},
        hoverlabel={"font_size": 13},
        separators=",\u202f",  # format français
    )
    fig.update_layout(**layout)
    # Titres de sous-graphiques / facettes : couleur inline (le thème Streamlit les force en gris foncé)
    fig.for_each_annotation(
        lambda a: (
            a.update(text=f'<span style="color:{PAL["font"]}">{a.text}</span>')
            if a.font.color is None and "<span" not in a.text
            else None
        )
    )
    if fig.layout.title.text:  # titre tout en haut, légende juste au-dessus de la zone de tracé
        fig.update_layout(margin_t=95, title={"y": 0.98, "yanchor": "top", "x": 0, "xanchor": "left"})
    fig.update_xaxes(gridcolor=PAL["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=PAL["grid"], zeroline=False)
    # Axes logarithmiques : graduations aux seules puissances de 10 (pas de « 2 5 » intermédiaires)
    fig.update_xaxes(dtick=1, selector={"type": "log"})
    fig.update_yaxes(dtick=1, selector={"type": "log"})
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def empty_guard(df: pd.DataFrame) -> bool:
    if df.empty:
        st.warning("Aucun jeu ne correspond aux filtres de la barre latérale : élargissez la sélection.")
        return True
    return False


try:
    with st.spinner("Lecture des tables Gold sur HDFS (premier chargement : environ 30 s)…"):
        gold = {name: load_gold_table(path) for name, path in GOLD_TABLES.items()}
except AnalysisException as exc:  # table absente : notebooks J3/J4 non exécutés
    st.error(
        f"Lecture de la couche Gold impossible ({type(exc).__name__}). Exécutez les notebooks J3 et J4."
    )
    st.stop()
pareto, trends, top, games_all = gold["pareto"], gold["trends"], gold["top"], gold["games"]
TOTAL_REVIEWS = int(pareto["total_reviews"].sum())
CATALOG_GAMES = int(trends["catalog_games"].sum())
REVIEWED_GAMES = int(pareto["n_games"].sum())
GLOBAL_RECO = float((pareto["recommendation_rate"] * pareto["total_reviews"]).sum() / TOTAL_REVIEWS)

# ======================================================================================================
# Barre latérale : filtres + périmètre
# ======================================================================================================
with st.sidebar:
    st.markdown("## 🎮 Steam Lakehouse")
    st.caption(f"Filtres appliqués aux {fmt_int(len(games_all))} jeux segmentés (Gold `game_segments`).")
    genres_all = pareto.sort_values("rank_by_reviews")["primary_genre"].tolist()
    genres_segmented = [g for g in genres_all if g in set(games_all["primary_genre"])]
    genres = st.multiselect("Genres", genres_segmented, default=genres_segmented, placeholder="Aucun genre")
    tiers_all = sorted(games_all["price_tier"].unique())
    tiers = st.multiselect("Gamme de prix", tiers_all, default=tiers_all, format_func=tier_label)
    y_min, y_max = int(games_all["release_year"].min()), int(games_all["release_year"].max())
    years = st.slider("Années de sortie", y_min, y_max, (y_min, y_max))
    segments = st.multiselect("Segments K-Means", SEGMENT_ORDER, default=SEGMENT_ORDER)
    vitality = st.multiselect("Profils de vitalité", VITALITY_ORDER, default=VITALITY_ORDER)

games = games_all[
    games_all["primary_genre"].isin(genres)
    & games_all["price_tier"].isin(tiers)
    & games_all["release_year"].between(*years)
    & games_all["segment"].isin(segments)
    & games_all["vitality_profile"].isin(vitality)
]

with st.sidebar:
    st.divider()
    st.markdown(
        f"**Périmètre filtré**  \n### {fmt_int(len(games))} jeux\n"
        f"**{fmt_millions(games['n_reviews'].sum())}** d'avis sur {fmt_millions(TOTAL_REVIEWS)}"
    )
    coverage = games["n_reviews"].sum() / TOTAL_REVIEWS
    st.progress(min(1.0, coverage), text=f"{pct(coverage, 1)} des avis du corpus")
    if st.button("Rafraîchir les données Gold", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.markdown(
        "**Walid Hdilou & Joud Atallah**  \nM2 IPSSI : Data Lake, Data Warehouse & Lakehouse  \n"
        "[Dépôt GitHub](https://github.com/Joud04/Archi_Data_Lake_Warehouse_LA)"
    )

# ======================================================================================================
# En-tête
# ======================================================================================================
st.title("Steam : ce qui fait le succès d'un jeu")
st.markdown(
    '<div class="hero-badges"><span>HDFS 3.5 · 3 DataNodes</span><span>Apache Spark 4.2</span>'
    "<span>Delta Lake 4.4</span><span>Spark MLlib K-Means</span><span>Streamlit + Plotly</span></div>",
    unsafe_allow_html=True,
)
st.caption(
    f"{fmt_millions(TOTAL_REVIEWS)} d'avis Kaggle joints au catalogue SteamSpy ({fmt_int(CATALOG_GAMES)} jeux, "
    f"snapshot {SNAPSHOT_YEAR}) · Architecture Médaillon Bronze → Silver → Gold · source : couche Gold uniquement"
)

tabs = st.tabs(
    [
        "📊 Vue exécutive",
        "🏆 Pareto & genres",
        "📈 Dynamique & vitalité",
        "🎯 Temps de jeu & segments",
        "🏗️ Architecture & méthode",
    ]
)

# ======================================================================================================
# 1. Vue exécutive
# ======================================================================================================
with tabs[0]:
    st.markdown("##### Corpus complet")
    spark_years = trends[trends["is_reliable"]].sort_values("release_year")
    k = st.columns(4)
    k[0].metric(
        "Avis analysés",
        fmt_millions(TOTAL_REVIEWS),
        help="Somme des avis (Gold genre_pareto). Courbe : avis par année de sortie (2006-2016).",
        chart_data=spark_years["total_reviews"].tolist(),
        chart_type="bar",
    )
    k[1].metric(
        "Jeux au catalogue", fmt_int(CATALOG_GAMES), help="Catalogue SteamSpy (Gold release_year_trends)"
    )
    k[2].metric(
        "Jeux commentés",
        fmt_int(REVIEWED_GAMES),
        f"{pct(REVIEWED_GAMES / CATALOG_GAMES, 0)} du catalogue",
        delta_color="off",
        delta_arrow="off",
    )
    k[3].metric(
        "Avis positifs",
        pct(GLOBAL_RECO),
        help="Part d'avis recommandant le jeu. Courbe : satisfaction par année de sortie (2006-2016).",
        chart_data=spark_years["recommendation_rate"].tolist(),
    )

    if not empty_guard(games):
        st.markdown("##### Périmètre filtré (jeux segmentés)")
        reco_f = wavg(games, "positive_ratio")
        seg_median_pt = games_all["median_playtime"].median() / 60
        pt_f = games["median_playtime"].median() / 60
        k = st.columns(5)
        k[0].metric(
            "Avis positifs (filtre)",
            pct(reco_f),
            delta((reco_f - GLOBAL_RECO) * 100, "pts", "vs global"),
        )
        k[1].metric(
            "Temps de jeu médian", f"{num(pt_f)} h", delta(pt_f - seg_median_pt, "h", "vs ensemble")
        )
        k[2].metric("Prix médian", f"{num(games['price'].median(), 2)} £")
        k[3].metric(
            "Avis jugés utiles", pct(wavg(games, "helpful_ratio")), help="Avis ayant reçu au moins un vote"
        )
        k[4].metric(
            "Avis à contre-courant",
            pct(wavg(games, "contrarian_ratio")),
            help="Avis opposé au consensus du jeu dans le catalogue Steam (polarisation)",
        )

        st.subheader("Prix et satisfaction")
        tier = (
            games.assign(
                pos=games["positive_ratio"] * games["n_reviews"],
                helpful=games["helpful_ratio"] * games["n_reviews"],
                contra=games["contrarian_ratio"] * games["n_reviews"],
            )
            .groupby("price_tier")
            .agg(
                jeux=("app_id", "size"),
                avis=("n_reviews", "sum"),
                pos=("pos", "sum"),
                helpful=("helpful", "sum"),
                contra=("contra", "sum"),
            )
            .reset_index()
        )
        tier["Avis positifs"] = tier["pos"] / tier["avis"]
        tier["Avis jugés utiles"] = tier["helpful"] / tier["avis"]
        tier["Avis à contre-courant"] = tier["contra"] / tier["avis"]
        tier["gamme"] = tier["price_tier"].map(tier_label)
        long = tier.melt(
            id_vars=["gamme", "jeux", "avis"],
            value_vars=["Avis positifs", "Avis jugés utiles", "Avis à contre-courant"],
            var_name="indicateur",
            value_name="part",
        )
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.bar(
                long,
                x="gamme",
                y="part",
                color="indicateur",
                barmode="group",
                color_discrete_sequence=PAL["series"],
                text=long["part"].map(lambda v: pct(v, 0)),
                hover_data={"jeux": True, "avis": ":,", "part": ":.1%"},
                labels={"gamme": "Gamme de prix", "part": "Part des avis", "indicateur": ""},
            )
            fig.update_traces(textposition="outside", cliponaxis=False, textfont_color=PAL["font"])
            # Étiquettes sélectives : seule la satisfaction est annotée, le reste est au survol
            fig.for_each_trace(lambda t: t.update(text=None) if t.name != "Avis positifs" else None)
            show(fig, 420, yaxis_tickformat=".0%", bargap=0.25)
        with c2:
            st.markdown("**Gratuit vs payant**")
            for label, df in [
                ("Gratuit", games[games["price"] == 0]),
                ("Payant", games[games["price"] > 0]),
            ]:
                with st.container(border=True):
                    if df.empty:
                        st.markdown(f"**{label}** : aucun jeu dans le périmètre")
                        continue
                    st.markdown(
                        f"**{label}** · {fmt_int(len(df))} jeux · {fmt_millions(df['n_reviews'].sum())} d'avis  \n"
                        f"### {pct(wavg(df, 'positive_ratio'))} d'avis positifs\n"
                        f"Temps de jeu médian **{num(df['median_playtime'].median() / 60)} h** · avis utiles "
                        f"**{pct(wavg(df, 'helpful_ratio'), 0)}** · à contre-courant "
                        f"**{pct(wavg(df, 'contrarian_ratio'), 0)}**"
                    )
        insight(
            "La satisfaction suit une **courbe en cloche** : les jeux à 5-15 £ sont les mieux notés, les gratuits "
            "et les plus chers (≥ 30 £) déçoivent davantage (Kruskal-Wallis, p ≈ 5·10⁻³⁷). Un avis négatif est "
            "**1,9 fois plus souvent jugé utile** qu'un avis positif (χ², V de Cramér = 0,12) : les critiques "
            "argumentées sont celles que la communauté retient."
        )
        explain(
            """
SELECT price_tier,
       SUM(positive_ratio * n_reviews)   / SUM(n_reviews) AS avis_positifs,
       SUM(helpful_ratio * n_reviews)    / SUM(n_reviews) AS avis_utiles,
       SUM(contrarian_ratio * n_reviews) / SUM(n_reviews) AS avis_contre_courant
FROM delta.`/lakehouse/gold/game_segments`
GROUP BY price_tier ORDER BY price_tier;
""",
            "Les taux sont **pondérés par le nombre d'avis** : ils décrivent l'ensemble des avis de la gamme et "
            "non un jeu moyen (sinon un jeu à 20 avis pèserait autant qu'un jeu à 50 000). Le périmètre est celui "
            "des jeux segmentés en J4 (≥ 20 avis, temps de jeu mesuré), qui couvre 94,5 % des avis. Au grain jeu, "
            "le test de Kruskal-Wallis confirme l'effet de la gamme de prix sur la satisfaction (ε² = 0,05 : effet "
            "réel mais modeste) ; les comparaisons deux à deux sont corrigées par Bonferroni.",
        )

# ======================================================================================================
# 2. Pareto & genres
# ======================================================================================================
with tabs[1]:
    st.subheader("Concentration du marché : courbe de Pareto par genre")
    par = pareto.sort_values("rank_by_reviews").reset_index(drop=True)
    selected = par["primary_genre"].isin(genres)
    fig = go.Figure()
    for cls in ["A", "B", "C"]:
        sub = par[par["pareto_class"] == cls]
        fig.add_bar(
            x=sub["primary_genre"],
            y=sub["share_reviews"],
            name=f"Classe {cls}",
            marker={"color": PAL["pareto"][cls], "opacity": np.where(selected[sub.index], 1.0, 0.3)},
            customdata=np.stack(
                [sub["n_games"], sub["recommendation_rate"], sub["median_reviews_per_game"]], axis=-1
            ),
            hovertemplate="<b>%{x}</b><br>Part des avis : %{y:.1%}<br>Jeux : %{customdata[0]}"
            "<br>Avis positifs : %{customdata[1]:.1%}<br>Avis médian / jeu : %{customdata[2]}<extra></extra>",
        )
    fig.add_scatter(
        x=par["primary_genre"],
        y=par["pareto_cum_share"],
        name="Part cumulée",
        mode="lines+markers",
        line={"color": PAL["series"][1], "width": 2},
        marker={"size": 8},
        hovertemplate="<b>%{x}</b><br>Cumul : %{y:.1%}<extra></extra>",
    )
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color=PAL["neg"],
        annotation_text="Seuil 80 %",
        annotation_position="top left",
        annotation_font_color=PAL["neg"],
    )
    show(fig, 460, yaxis_tickformat=".0%", yaxis_range=[0, 1.05], xaxis_tickangle=-35)
    n_a = int((pareto["pareto_class"] == "A").sum())
    share_a = pareto.loc[pareto["pareto_class"] == "A", "share_reviews"].sum()
    insight(
        f"**{n_a} genres sur {len(pareto)} concentrent {pct(share_a, 0)} des avis** (classe A), et Action en porte à "
        "lui seul 46 %. Au grain jeu, la concentration est encore plus forte : **8,6 % des jeux reçoivent 80 % "
        "des avis**. Les genres hors filtre apparaissent atténués."
    )
    explain(
        """
SELECT primary_genre, total_reviews,
       DENSE_RANK() OVER (ORDER BY total_reviews DESC)                     AS rank_by_reviews,
       total_reviews / SUM(total_reviews) OVER ()                          AS share_reviews,
       SUM(total_reviews) OVER (ORDER BY total_reviews DESC
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
         / SUM(total_reviews) OVER ()                                      AS pareto_cum_share
FROM by_genre;   -- table Gold : /lakehouse/gold/genre_pareto (J3)
""",
        "Le cumul ordonné (`ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`) donne la courbe de Pareto ; la "
        "classe A regroupe les genres nécessaires pour atteindre 80 % des avis, B jusqu'à 95 %, C le reste. Barres "
        "et courbe partagent la même échelle (part des avis), d'où un seul axe.",
    )

    st.subheader("Asymétrie : la moyenne trompe, la médiane décrit le jeu typique")
    c1, c2 = st.columns(2)
    with c1:
        top_par = par.head(10).iloc[::-1]
        fig = go.Figure()
        for _, r in top_par.iterrows():
            fig.add_scatter(
                x=[r["median_reviews_per_game"], r["mean_reviews_per_game"]],
                y=[r["primary_genre"]] * 2,
                mode="lines",
                line={"color": PAL["muted"], "width": 2},
                showlegend=False,
                hoverinfo="skip",
            )
        for col, name, color in [
            ("median_reviews_per_game", "Médiane", PAL["series"][0]),
            ("mean_reviews_per_game", "Moyenne", PAL["series"][1]),
        ]:
            fig.add_scatter(
                x=top_par[col],
                y=top_par["primary_genre"],
                mode="markers",
                name=name,
                marker={"color": color, "size": 11},
                hovertemplate="<b>%{y}</b><br>" + name + " : %{x:,.0f} avis / jeu<extra></extra>",
            )
        show(
            fig,
            430,
            xaxis_type="log",
            xaxis_title="Avis par jeu (échelle log)",
            title="Moyenne vs médiane (10 premiers genres)",
        )
    with c2:
        if not empty_guard(games):
            keep = games["primary_genre"].value_counts().head(8).index.tolist()
            fig = px.box(
                games[games["primary_genre"].isin(keep)],
                x="n_reviews",
                y="primary_genre",
                orientation="h",
                log_x=True,
                points=False,
                color_discrete_sequence=[PAL["accent"]],
                category_orders={"primary_genre": keep},
                labels={"n_reviews": "Avis par jeu (échelle log)", "primary_genre": ""},
                title="Distribution par genre : médiane et IQR (P25-P75)",
            )
            show(fig, 430)
    skew = games_all["n_reviews"].skew()
    insight(
        f"Le nombre d'avis par jeu est très asymétrique à droite (**skewness = {num(skew, 1)}** sur les jeux segmentés, "
        "14,6 sur les 8 935 jeux commentés) : dans Action, la moyenne (753 avis) est **17 fois** la médiane (43). "
        "Le dashboard compare donc les genres par la **médiane** et l'**IQR**, robustes aux blockbusters."
    )
    explain(
        """
SELECT primary_genre,
       ROUND(AVG(n_reviews), 1)                                                 AS mean_reviews_per_game,
       PERCENTILE_APPROX(n_reviews, 0.5)                                        AS median_reviews_per_game,
       PERCENTILE_APPROX(n_reviews, 0.75) - PERCENTILE_APPROX(n_reviews, 0.25)  AS iqr_reviews_per_game
FROM v_games GROUP BY primary_genre;   -- + SKEWNESS(n_reviews) = 14,6 (J3)
""",
        "La moyenne a un point de rupture de 0 % (une seule valeur extrême la déplace sans limite), la médiane "
        "de 50 % et l'IQR de 25 %. Les « outliers » au sens de Tukey (14,7 % des jeux) portent 87,8 % des avis : "
        "ce sont des valeurs légitimes d'un marché *winner takes all*, pas des erreurs à supprimer.",
    )

    st.subheader("Meilleurs jeux par genre (borne de Wilson)")
    top_genres = [g for g in genres_all if g in set(top["primary_genre"])]
    genre = st.selectbox("Genre", top_genres, index=0)
    sel = top[top["primary_genre"] == genre].sort_values("rank_in_genre")
    st.dataframe(
        sel[
            [
                "rank_in_genre",
                "game_name",
                "release_year",
                "price",
                "n_reviews",
                "recommendation_rate",
                "wilson_lower_bound",
                "volume_rank_in_genre",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "rank_in_genre": st.column_config.NumberColumn("Rang qualité", format="%d"),
            "game_name": "Jeu",
            "release_year": st.column_config.NumberColumn("Sortie", format="%d"),
            "price": st.column_config.NumberColumn("Prix (£)", format="%.2f"),
            "n_reviews": st.column_config.NumberColumn("Avis", format="%d"),
            "recommendation_rate": st.column_config.ProgressColumn(
                "Avis positifs", format="percent", min_value=0, max_value=1
            ),
            "wilson_lower_bound": st.column_config.NumberColumn("Borne de Wilson 95 %", format="%.3f"),
            "volume_rank_in_genre": st.column_config.NumberColumn("Rang volume", format="%d"),
        },
    )
    explain(
        """
SELECT *, ROW_NUMBER() OVER (PARTITION BY primary_genre
                             ORDER BY wilson_lower_bound DESC, n_reviews DESC, app_id) AS rank_in_genre
FROM scored WHERE n_reviews >= 50;   -- table Gold : /lakehouse/gold/top_games_by_genre (top 10)
""",
        "La borne basse de l'intervalle de Wilson à 95 % pénalise les petits effectifs : un jeu à 100 % sur 60 avis "
        "passe derrière Half-Life 2 (99,98 % sur 4 052 avis). Le *rang volume* montre que qualité et popularité "
        "divergent : pour 7 genres sur 18, le n°1 en qualité n'est pas dans le top 10 en volume.",
    )

# ======================================================================================================
# 3. Dynamique temporelle & vitalité
# ======================================================================================================
with tabs[2]:
    st.subheader("Cohortes par année de sortie (fenêtres LAG)")
    reliable_only = st.toggle(
        "Années fiables uniquement (≥ 30 jeux et couverture des avis ≥ 50 %)",
        value=True,
        help="Après 2016, le corpus d'avis ne couvre que 1 à 3 % des sorties : artefact de collecte.",
    )
    tr = trends[trends["release_year"].between(*years) & trends["recommendation_rate"].notna()]
    if reliable_only:
        tr = tr[tr["is_reliable"]]
    if tr.empty:
        st.warning("Aucune année ne correspond à la plage sélectionnée.")
    else:
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=["Volume d'avis", "Prix moyen (£)", "Part d'avis positifs"],
        )
        fig.add_bar(
            x=tr["release_year"],
            y=tr["total_reviews"],
            name="Avis",
            marker_color=PAL["accent"],
            customdata=tr[["reviews_yoy_pct", "n_games", "review_coverage"]],
            hovertemplate="%{x} : %{y:,} avis<br>vs N-1 : %{customdata[0]:+.1f} %<br>Jeux : %{customdata[1]}"
            "<br>Couverture : %{customdata[2]:.0%}<extra></extra>",
            showlegend=False,
            row=1,
            col=1,
        )
        fig.add_scatter(
            x=tr["release_year"],
            y=tr["avg_price"],
            name="Prix moyen",
            mode="lines+markers",
            line={"color": PAL["series"][2], "width": 2},
            marker={"size": 8},
            customdata=tr[["avg_price_delta"]],
            hovertemplate="%{x} : %{y:.2f} £<br>vs N-1 : %{customdata[0]:+.2f} £<extra></extra>",
            showlegend=False,
            row=2,
            col=1,
        )
        for col, name, color in [
            ("recommendation_rate", "Taux annuel", PAL["series"][0]),
            ("recommendation_rate_ma3", "Moyenne glissante 3 ans", PAL["series"][1]),
        ]:
            fig.add_scatter(
                x=tr["release_year"],
                y=tr[col],
                name=name,
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 8},
                hovertemplate="%{x} : %{y:.1%}<extra>" + name + "</extra>",
                row=3,
                col=1,
            )
        fig.update_yaxes(tickformat=".0%", row=3, col=1)
        show(fig, 640, hovermode="x unified")
        c1, c2 = st.columns([3, 2])
        with c1:
            d = tr.dropna(subset=["recommendation_delta_pts"])
            fig = go.Figure(
                go.Bar(
                    x=d["release_year"],
                    y=d["recommendation_delta_pts"],
                    marker_color=np.where(d["recommendation_delta_pts"] >= 0, PAL["pos"], PAL["neg"]),
                    hovertemplate="%{x} : %{y:+.2f} pts vs année précédente<extra></extra>",
                )
            )
            show(
                fig,
                300,
                title="Variation annuelle de la satisfaction (LAG, en points)",
                yaxis_title="Points de %",
            )
        with c2:
            insight(
                "Sur la période fiable, l'offre explose (395 jeux en 2013, 3 022 en 2016 : ouverture de Steam aux "
                "indépendants) pendant que la satisfaction lissée perd **≈ 8 points**. Plus de jeux, chacun moins "
                "visible et en moyenne moins bien reçu ; une partie de l'écart vient du **biais du survivant** "
                "(les vieux jeux encore commentés sont des classiques)."
            )
    explain(
        """
SELECT release_year, n_games, total_reviews, avg_price, recommendation_rate,
       ROUND((recommendation_rate - LAG(recommendation_rate, 1) OVER w) * 100, 2) AS recommendation_delta_pts,
       ROUND(try_divide(total_reviews - LAG(total_reviews, 1) OVER w, LAG(total_reviews, 1) OVER w) * 100, 1)
         AS reviews_yoy_pct,
       AVG(recommendation_rate) OVER (ORDER BY release_year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
         AS recommendation_rate_ma3
FROM filled WINDOW w AS (ORDER BY release_year);   -- Gold : /lakehouse/gold/release_year_trends
""",
        "Le calendrier complet (`SEQUENCE` + `LEFT JOIN`) garantit que `LAG` compare deux années consécutives. "
        "Les avis n'ont pas d'horodatage : l'axe est l'année de sortie (cohortes). Trois mesures d'unités "
        "différentes sont présentées en petits multiples alignés plutôt qu'avec un double axe trompeur.",
    )

    st.subheader("Jeu toujours joué ou jeu mort ? Profils de vitalité")
    if not empty_guard(games):
        vit = games.assign(
            temps_h=games["median_playtime"] / 60,
            taille=np.log10(games["owners_mid"]) - 3.5,
        )
        fig = px.scatter(
            vit,
            x="game_age",
            y="temps_h",
            size="taille",
            size_max=14,
            facet_col="vitality_profile",
            facet_col_wrap=2,
            facet_row_spacing=0.12,
            category_orders={"vitality_profile": VITALITY_ORDER},
            color_discrete_sequence=[PAL["accent"]],
            opacity=0.55,
            log_y=True,
            hover_name="game_name",
            hover_data={
                "owners_mid": ":,",
                "release_year": True,
                "n_reviews": ":,",
                "taille": False,
                "game_age": False,
                "temps_h": ":.1f",
                "vitality_profile": False,
            },
            labels={
                "game_age": "Ancienneté (ans, en 2019)",
                "temps_h": "Temps de jeu médian (h)",
                "owners_mid": "Propriétaires",
                "n_reviews": "Avis",
                "release_year": "Sortie",
            },
        )
        fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>"))
        fig.add_vline(x=OLD_GAME_YEARS, line_dash="dot", line_color=PAL["muted"], row="all", col="all")
        fig.add_hline(
            y=ENGAGED_MINUTES / 60, line_dash="dot", line_color=PAL["muted"], row="all", col="all"
        )
        show(fig, 620)
        vstats = (
            games.assign(pos=games["positive_ratio"] * games["n_reviews"])
            .groupby("vitality_profile")
            .agg(
                jeux=("app_id", "size"),
                avis=("n_reviews", "sum"),
                pos=("pos", "sum"),
                temps=("median_playtime", "median"),
                proprietaires=("owners_mid", "median"),
            )
            .reindex(VITALITY_ORDER)
            .dropna()
        )
        vstats["avis_positifs"] = vstats["pos"] / vstats["avis"]
        vstats["part_avis"] = vstats["avis"] / vstats["avis"].sum()
        vstats["temps"] = vstats["temps"] / 60
        st.dataframe(
            vstats.drop(columns="pos").rename_axis("Profil de vitalité"),
            width="stretch",
            column_config={
                "jeux": st.column_config.NumberColumn("Jeux", format="%d"),
                "avis": st.column_config.NumberColumn("Avis", format="%d"),
                "temps": st.column_config.NumberColumn("Temps de jeu médian (h)", format="%.1f"),
                "proprietaires": st.column_config.NumberColumn("Propriétaires (médiane)", format="%d"),
                "avis_positifs": st.column_config.ProgressColumn(
                    "Avis positifs", format="percent", min_value=0, max_value=1
                ),
                "part_avis": st.column_config.NumberColumn("Part des avis", format="percent"),
            },
        )
        insight(
            "Les **classiques durables** (≥ 5 ans et toujours joués) sont les mieux notés et pèsent le plus lourd "
            "en avis ; les **jeux de niche** (≤ 35 000 propriétaires) sont les moins bien reçus. Les écarts de "
            "satisfaction entre profils sont significatifs (Kruskal-Wallis H = 43,5, p ≈ 2·10⁻⁹)."
        )
    explain(
        """
SELECT vitality_profile, COUNT(*) AS jeux,
       SUM(positive_ratio * n_reviews) / SUM(n_reviews) AS avis_positifs
FROM delta.`/lakehouse/gold/game_segments` GROUP BY vitality_profile;
-- Règles J4 : Niche si owners_mid <= 35 000 ; sinon âge >= 5 ans et temps médian >= 227 min
--   -> Classique durable ; âge < 5 ans et temps >= 227 min -> Hit récent ; sinon Vite abandonné / jeu mort
""",
        "Sans historique d'activité, la vitalité combine **ancienneté**, **rétention** (temps de jeu médian) et "
        "**base de joueurs** (taille des points). Les seuils sont des médianes de l'échantillon (lignes "
        "pointillées). Un panneau par profil évite de superposer 4 couleurs difficiles à distinguer.",
    )

# ======================================================================================================
# 4. Temps de jeu & segmentation K-Means
# ======================================================================================================
with tabs[3]:
    if not empty_guard(games):
        st.subheader("Le temps de jeu fait-il la satisfaction ?")
        bands = (
            games.assign(pos=games["positive_ratio"] * games["n_reviews"])
            .groupby("playtime_band")
            .agg(
                jeux=("app_id", "size"),
                avis=("n_reviews", "sum"),
                pos=("pos", "sum"),
                catalogue=("catalog_positive_ratio", "median"),
            )
            .reindex(PLAYTIME_ORDER)
            .dropna()
            .reset_index()
        )
        bands["Avis du corpus (pondéré)"] = bands["pos"] / bands["avis"]
        bands["Note catalogue Steam (médiane)"] = bands["catalogue"]
        long = bands.melt(
            id_vars=["playtime_band", "jeux", "avis"],
            value_vars=["Avis du corpus (pondéré)", "Note catalogue Steam (médiane)"],
            var_name="source",
            value_name="part",
        )
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.bar(
                long,
                x="playtime_band",
                y="part",
                color="source",
                barmode="group",
                color_discrete_sequence=PAL["series"][:2],
                text=long["part"].map(lambda v: pct(v, 0)),
                hover_data={"jeux": True, "avis": ":,", "part": ":.1%"},
                labels={"playtime_band": "Temps de jeu médian par joueur", "part": "Part d'avis positifs"},
            )
            fig.update_traces(textposition="outside", cliponaxis=False, textfont_color=PAL["font"])
            show(fig, 400, yaxis_tickformat=".0%", yaxis_range=[0.5, 1], bargap=0.25)
        with c2:
            insight(
                "Le lien est **positif mais faible** (Spearman ρ = +0,13, p < 10⁻¹⁴) : la satisfaction grimpe "
                "jusqu'à 3-10 h puis plafonne. Les deux sources (corpus d'avis et note du catalogue) racontent la "
                "même histoire. `median_playtime = 0` signifie *non mesuré* par SteamSpy : ces jeux sont exclus, "
                "jamais imputés."
            )

        st.subheader("Segmentation K-Means (Spark MLlib, k = 5)")
        z = model_space(games_all).assign(segment=games_all["segment"])
        centroids = (
            z[z["segment"].isin(segments)]
            .groupby("segment")
            .mean()
            .reindex([s for s in SEGMENT_ORDER if s in segments])
            .rename(columns={f: lbl for f, (lbl, _) in MODEL_FEATURES.items()})
            .reset_index()
            .melt(id_vars="segment", var_name="feature", value_name="z")
        )
        fig = px.bar(
            centroids,
            x="feature",
            y="z",
            color="segment",
            barmode="group",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"segment": SEGMENT_ORDER},
            hover_data={"z": ":+.2f"},
            labels={"feature": "", "z": "Centroïde (écarts-types)", "segment": ""},
        )
        fig.add_hline(y=0, line_color=PAL["muted"])
        show(fig, 430, bargap=0.2, bargroupgap=0.05)
        insight(
            "Chaque barre est la position moyenne du segment sur une variable du modèle, en écarts-types "
            "(0 = jeu moyen). Le modèle place les **hits premium** très haut en volume et en temps de jeu, isole "
            "les **jeux décevants** par leur satisfaction et les **free-to-play** par le prix."
        )

        st.markdown("**Explorer les jeux d'un segment**")
        axes = {
            "Volume d'avis": ("n_reviews", True),
            "Satisfaction (avis positifs)": ("positive_ratio", False),
            "Temps de jeu médian (min)": ("median_playtime", True),
            "Prix (£)": ("price", False),
            "Propriétaires": ("owners_mid", True),
            "Avis jugés utiles": ("helpful_ratio", False),
        }
        c1, c2, c3 = st.columns(3)
        x_lbl = c1.selectbox("Axe horizontal", list(axes), index=0)
        y_lbl = c2.selectbox("Axe vertical", list(axes), index=1)
        focus = c3.selectbox("Segment mis en avant", [s for s in SEGMENT_ORDER if s in segments])
        (x_col, x_log), (y_col, y_log) = axes[x_lbl], axes[y_lbl]
        pts = games.assign(groupe=np.where(games["segment"] == focus, focus, "Autres segments"))
        pts = pts.sort_values("groupe", key=lambda s: s == focus)  # segment mis en avant dessiné au-dessus
        fig = px.scatter(
            pts,
            x=x_col,
            y=y_col,
            color="groupe",
            log_x=x_log,
            log_y=y_log,
            opacity=0.7,
            color_discrete_map={focus: SEGMENT_COLORS[focus], "Autres segments": PAL["muted"]},
            hover_name="game_name",
            hover_data={"segment": True, "primary_genre": True, "n_reviews": ":,", "groupe": False},
            labels={x_col: x_lbl, y_col: y_lbl, "segment": "Segment", "primary_genre": "Genre"},
        )
        fig.update_traces(marker={"size": 7, "line": {"width": 0}})
        for col, update_axis in [(x_col, fig.update_xaxes), (y_col, fig.update_yaxes)]:
            if col.endswith("_ratio"):  # taux -> axe en pourcentage
                update_axis(tickformat=".0%")
        show(fig, 470)

        st.markdown("**Profils et recommandations stratégiques**")
        prof = (
            games.assign(pos=games["positive_ratio"] * games["n_reviews"])
            .groupby("segment")
            .agg(
                jeux=("app_id", "size"),
                avis=("n_reviews", "sum"),
                pos=("pos", "sum"),
                temps=("median_playtime", "median"),
                prix=("price", "median"),
                age=("game_age", "median"),
                utiles=("helpful_ratio", "median"),
                contre=("contrarian_ratio", "median"),
            )
        )
        prof["avis_positifs"] = prof["pos"] / prof["avis"]
        prof["part_avis"] = prof["avis"] / prof["avis"].sum()
        cols = st.columns(2)
        for i, seg in enumerate([s for s in SEGMENT_ORDER if s in prof.index]):
            r = prof.loc[seg]
            with cols[i % 2].container(border=True):
                st.markdown(
                    f'<span style="color:{SEGMENT_COLORS[seg]};font-size:1.2rem">●</span> **{seg}**',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"### {pct(r['avis_positifs'], 0)} d'avis positifs\n"
                    f"**{fmt_int(r['jeux'])}** jeux · **{pct(r['part_avis'], 0)}** des avis · temps de jeu "
                    f"médian **{num(r['temps'] / 60, 1)} h**"
                )
                st.caption(
                    f"Prix médian {num(r['prix'], 2)} £ · âge médian {num(r['age'], 0)} ans · avis utiles "
                    f"{pct(r['utiles'], 0)} · avis à contre-courant {pct(r['contre'], 0)}"
                )
                st.markdown(SEGMENT_PLAYBOOK[seg])
        explain(
            """
-- Pipeline Spark MLlib (J4) sur les jeux >= 20 avis au temps de jeu mesuré :
VectorAssembler(log1p(n_reviews), positive_ratio, log1p(median_playtime), log1p(price), log1p(game_age))
  -> StandardScaler(withMean = true, withStd = true) -> KMeans(k = 5, seed = 42)
-- Profils affichés :
SELECT segment, COUNT(*) AS jeux, SUM(positive_ratio * n_reviews) / SUM(n_reviews) AS avis_positifs,
       PERCENTILE_APPROX(median_playtime, 0.5) / 60 AS temps_median_h
FROM delta.`/lakehouse/gold/game_segments` GROUP BY segment;
""",
            "Les 4 variables positives très asymétriques (skewness de 1,5 à 29) passent en `log1p` puis sont "
            "centrées-réduites : sinon la distance euclidienne du K-Means serait dominée par le volume d'avis. "
            "**k = 5** retenu par la méthode du coude et la silhouette (0,346) ; un K-Means scikit-learn "
            "indépendant (20 initialisations) retrouve la même partition (ARI = 0,983). Les centroïdes affichés "
            "sont recalculés depuis la table Gold dans le même espace standardisé.",
        )

# ======================================================================================================
# 5. Architecture & méthode
# ======================================================================================================
with tabs[4]:
    st.subheader("Architecture Lakehouse Médaillon")
    stages = [
        ("Sources Kaggle", "steam-reviews : 3,76 M avis<br>steam-store-games : 27 075 jeux", ""),
        (
            "Bronze · Delta",
            "données brutes + lignage<br><code>_ingested_at</code>, <code>_source_file</code>",
            "J1",
        ),
        (
            "Silver · Delta ACID",
            "BroadcastHashJoin, règles métier<br>Schema Enforcement, MERGE, Time Travel",
            "J2",
        ),
        (
            "Gold · Delta",
            "fenêtres SQL (Pareto, LAG, ROW_NUMBER)<br>OPTIMIZE, K-Means Spark MLlib",
            "J3 · J4",
        ),
        ("Streamlit + Plotly", "ce dashboard<br>lecture de la couche Gold seule", ""),
    ]
    cards = "".join(
        (f'<div class="arrow">→<span>{step}</span></div>' if i else "")
        + f'<div class="stage"><div class="t">{title}</div><div class="d">{desc}</div></div>'
        for i, (title, desc, step) in enumerate(stages)
    )
    st.markdown(
        f"""<style>
.pipeline {{display: flex; align-items: stretch; gap: 6px; flex-wrap: wrap; margin: 6px 0 4px 0;}}
.pipeline .stage {{flex: 1 1 150px; border: 1px solid rgba(102,192,244,.45); border-radius: 10px;
  padding: 10px 12px; background: rgba(42,71,94,.55);}}
.pipeline .t {{font-weight: 700; color: #66c0f4; margin-bottom: 4px;}}
.pipeline .d {{font-size: .82rem; line-height: 1.35;}}
.pipeline .arrow {{display: flex; flex-direction: column; justify-content: center; align-items: center;
  font-size: 1.4rem; opacity: .8;}}
.pipeline .arrow span {{font-size: .7rem; opacity: .8;}}
.hdfs-band {{border: 1px dashed rgba(139,163,184,.6); border-radius: 8px; padding: 6px 12px;
  font-size: .82rem; text-align: center; margin-bottom: 18px;}}
</style>
<div class="pipeline">{cards}</div>
<div class="hdfs-band">Stockage : cluster HDFS 3.5 · 1 NameNode + 3 DataNodes · blocs de 16 Mo · réplication 2
· tables Bronze / Silver / Gold au format Delta Lake</div>""",
        unsafe_allow_html=True,
    )
    c = st.columns(4)
    c[0].metric("Blocs HDFS", "16 Mo", "réplication × 2", delta_color="off", delta_arrow="off")
    c[1].metric(
        "Crash-test",
        "0 perte",
        "ré-réplication en 76 s",
        delta_color="off",
        delta_arrow="off",
    )
    c[2].metric(
        "OPTIMIZE Silver", "15 → 1", "fichiers actifs (small files)", delta_color="off", delta_arrow="off"
    )
    c[3].metric(
        "K-Means MLlib", "k = 5", "silhouette 0,35 · ARI 0,98", delta_color="off", delta_arrow="off"
    )

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown(
            "**Couches & garanties**\n"
            "- **Stockage distribué** : HDFS découpe `steam_reviews.csv` (100 Mo) en 7 blocs de 16 Mo, chacun "
            "répliqué sur 2 DataNodes ; la lecture continue pendant l'arrêt d'un nœud.\n"
            "- **Bronze** : tables Delta fidèles à la source, enrichies de `_ingested_at`, `_source_dataset`, "
            "`_source_file`.\n"
            "- **Silver** : jointure avis × catalogue en `BroadcastHashJoin` (aucun shuffle de la table de "
            "faits), 8 colonnes métier, rejet d'un schéma incompatible, upsert `MERGE` idempotent et lecture des "
            "versions passées.\n"
            "- **Gold** : 4 tables agrégées (`genre_pareto`, `release_year_trends`, `top_games_by_genre`, "
            "`game_segments`), quelques fichiers chacune, Silver compactée par `OPTIMIZE`."
        )
    with c2, st.container(border=True):
        st.markdown(
            "**Qualité des données & compromis**\n"
            "- **Pas de texte d'avis** : avec le texte, le fichier dépasse largement le gabarit du projet ; sans "
            "lui et après échantillonnage, il pèse ~100 Mo (3,76 M avis). Le contenu est approché par l'**utilité** "
            "des avis (`is_helpful`, issu de `review_votes`) et leur **polarisation** (`is_contrarian_review`, "
            "avis opposé au consensus du jeu).\n"
            "- **99,2 % de lignes identiques** dans les avis : ce sont des avis distincts sans identifiant, pas "
            "des doublons ; `dropDuplicates` n'est appliqué qu'à la clé du catalogue.\n"
            "- **Temps de jeu = 0** : non mesuré par SteamSpy pour les petits jeux → exclu, jamais imputé.\n"
            "- **Couverture temporelle** : après 2016, moins de 3 % des sorties ont des avis → années écartées "
            "des tendances (`is_reliable`)."
        )
    with st.expander("Sources des tables Gold de ce dashboard"):
        sources = [
            ("Pareto par genre (J3)", "pareto"),
            ("Cohortes annuelles (J3)", "trends"),
            ("Top 10 par genre (J3)", "top"),
            ("Segments de jeux (J4)", "games"),
        ]
        st.dataframe(
            pd.DataFrame(
                [
                    (label, GOLD_TABLES[key], len(gold[key]), len(gold[key].columns))
                    for label, key in sources
                ],
                columns=["Table", "Chemin Delta", "Lignes", "Colonnes"],
            ),
            width="stretch",
            hide_index=True,
        )
