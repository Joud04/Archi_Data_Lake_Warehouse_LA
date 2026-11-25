"""Dashboard Steam : marché du jeu vidéo et satisfaction des joueurs.

Lit UNIQUEMENT la couche Gold du Lakehouse (tables Delta sur HDFS, produites par les notebooks J3 et J4).
Toutes les tables sont petites (20 à 3 438 lignes) : elles tiennent dans le driver Spark de 512 Mo.
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from pyspark.errors import AnalysisException
from pyspark.sql import SparkSession

GOLD = "/lakehouse/gold"
GOLD_TABLES = {
    "pareto": f"{GOLD}/genre_pareto",
    "trends": f"{GOLD}/release_year_trends",
    "top": f"{GOLD}/top_games_by_genre",
    "segments": f"{GOLD}/game_segments",
}
SEGMENT_ORDER = [
    "Hits premium à forte rétention",
    "Anciens classiques du catalogue",
    "Pépites de niche appréciées",
    "Free-to-play à faible rétention",
    "Jeux décevants & controversés",
]
VITALITY_ORDER = ["Classique durable", "Hit récent", "Vite abandonné / jeu mort", "Niche confidentielle"]
PLAYTIME_ORDER = ["< 1 h", "1-3 h", "3-10 h", "10-30 h", ">= 30 h"]

# Palettes validées (séparation daltonisme + contraste), une variante par thème clair / sombre
PALETTES = {
    "light": {
        "segments": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        "vitality": ["#008300", "#6250d6", "#eda100", "#e34948"],
        "pareto": {"A": "#184f95", "B": "#3987e5", "C": "#86b6ef"},
        "line": ["#2a78d6", "#eb6834"],
        "bar": "#2a78d6",
    },
    "dark": {
        "segments": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "vitality": ["#0ca30c", "#9085e9", "#c98500", "#e66767"],
        "pareto": {"A": "#9ec5f4", "B": "#3987e5", "C": "#1c5cab"},
        "line": ["#3987e5", "#d95926"],
        "bar": "#3987e5",
    },
}

st.set_page_config(page_title="Steam Lakehouse Analytics", layout="wide")


@st.cache_resource
def get_spark():
    return (
        SparkSession.builder.appName("StreamlitGoldReader")
        .config("spark.driver.memory", "512m")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


@st.cache_data(ttl=60)
def load_gold_table(hdfs_path: str) -> pd.DataFrame:
    spark = get_spark()
    return spark.read.format("delta").load(hdfs_path).toPandas()


def weighted_rate(df: pd.DataFrame) -> float:
    """Taux de recommandation pondéré par les avis (= part d'avis positifs de l'ensemble)."""
    n = df["n_reviews"].sum()
    return float((df["positive_ratio"] * df["n_reviews"]).sum() / n) if n else float("nan")


def explain(sql: str, justification: str) -> None:
    with st.expander("Requête SQL & Justification statistique"):
        st.code(sql.strip(), language="sql")
        st.markdown(justification)


theme = "dark" if getattr(st.context.theme, "type", None) == "dark" else "light"
pal = PALETTES[theme]

try:
    gold = {name: load_gold_table(path) for name, path in GOLD_TABLES.items()}
except AnalysisException as exc:  # table absente : notebooks J3/J4 non exécutés
    st.error(
        f"Lecture de la couche Gold impossible ({type(exc).__name__}). Exécutez les notebooks J3 et J4."
    )
    st.stop()
pareto, trends, top, segments = gold["pareto"], gold["trends"], gold["top"], gold["segments"]

st.title("Steam : marché du jeu vidéo et satisfaction des joueurs")
st.caption(
    "Lakehouse Médaillon HDFS + Delta Lake : 3,76 M avis Kaggle joints au catalogue SteamSpy (snapshot 2019). "
    "Source du dashboard : couche Gold uniquement (`/lakehouse/gold/*`, cache 60 s)."
)

# --- Filtres (barre latérale) -------------------------------------------------------------------------
with st.sidebar:
    st.header("Filtres")
    genres_all = pareto.sort_values("rank_by_reviews")["primary_genre"].tolist()
    genres = st.multiselect("Genres principaux", genres_all, default=genres_all)
    y_min, y_max = int(segments["release_year"].min()), int(segments["release_year"].max())
    years = st.slider("Années de sortie", y_min, y_max, (y_min, y_max))
    seg_sel = st.multiselect("Segments K-Means", SEGMENT_ORDER, default=SEGMENT_ORDER)
    st.caption("Les filtres s'appliquent aux jeux segmentés (≥ 20 avis, temps de jeu mesuré).")

if not genres or not seg_sel:
    st.warning("Sélectionnez au moins un genre et un segment.")
    st.stop()

games = segments[
    segments["primary_genre"].isin(genres)
    & segments["release_year"].between(*years)
    & segments["segment"].isin(seg_sel)
]
if games.empty:
    st.warning("Aucun jeu ne correspond aux filtres.")
    st.stop()

# --- Indicateurs clés -----------------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Jeux", f"{len(games):,}".replace(",", " "))
k2.metric("Avis", f"{games['n_reviews'].sum():,}".replace(",", " "))
k3.metric("Avis positifs (pondéré)", f"{weighted_rate(games):.1%}")
k4.metric("Temps de jeu médian", f"{games['median_playtime'].median() / 60:.1f} h")
k5.metric("Prix médian", f"{games['price'].median():.2f} £")

# --- 1. Concentration du marché par genre (Pareto) ------------------------------------------------------
st.header("1. Concentration du marché par genre")
par = pareto[pareto["primary_genre"].isin(genres)].sort_values("rank_by_reviews")
fig = px.bar(
    par,
    x="share_reviews",
    y="primary_genre",
    orientation="h",
    color="pareto_class",
    color_discrete_map=pal["pareto"],
    category_orders={"primary_genre": par["primary_genre"].tolist(), "pareto_class": ["A", "B", "C"]},
    # Étiquette sélective : cumul affiché seulement pour les genres pesant >= 1 % des avis
    text=[
        f"cumul {c:.0%}" if s >= 0.01 else "" for c, s in zip(par["pareto_cum_share"], par["share_reviews"])
    ],
    hover_data={
        "n_games": True,
        "recommendation_rate": ":.1%",
        "median_reviews_per_game": True,
        "iqr_reviews_per_game": True,
        "median_price": ":.2f",
        "share_reviews": ":.1%",
    },
    labels={
        "share_reviews": "Part des avis",
        "primary_genre": "",
        "pareto_class": "Classe Pareto",
        "n_games": "Jeux",
        "recommendation_rate": "Avis positifs",
        "median_reviews_per_game": "Avis médian / jeu",
        "iqr_reviews_per_game": "IQR avis / jeu",
        "median_price": "Prix médian (£)",
    },
)
fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_layout(
    height=520,
    xaxis_tickformat=".0%",
    margin={"r": 80, "t": 50},
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
)
st.plotly_chart(fig, width="stretch")
class_a = pareto[pareto["pareto_class"] == "A"]
st.markdown(
    f"**Lecture :** {len(class_a)} genres sur {len(pareto)} (classe A) concentrent "
    f"{class_a['share_reviews'].sum():.0%} des avis. Le marché de l'attention est dominé par quelques genres, "
    "et au grain jeu 8,6 % des jeux reçoivent 80 % des avis."
)
explain(
    """
WITH by_genre AS (
    SELECT primary_genre, COUNT(*) AS n_games, SUM(n_reviews) AS total_reviews,
           ROUND(SUM(n_reviews * recommendation_rate) / SUM(n_reviews), 4) AS recommendation_rate,
           PERCENTILE_APPROX(n_reviews, 0.5) AS median_reviews_per_game,
           PERCENTILE_APPROX(n_reviews, 0.75) - PERCENTILE_APPROX(n_reviews, 0.25) AS iqr_reviews_per_game
    FROM v_games GROUP BY primary_genre
)
SELECT *, DENSE_RANK() OVER (ORDER BY total_reviews DESC) AS rank_by_reviews,
       total_reviews / SUM(total_reviews) OVER () AS share_reviews,
       SUM(total_reviews) OVER (ORDER BY total_reviews DESC
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
         / SUM(total_reviews) OVER () AS pareto_cum_share
FROM by_genre;   -- table Gold : /lakehouse/gold/genre_pareto
""",
    "**Justification :** le nombre d'avis par jeu est très asymétrique (`SKEWNESS` = 14,6 ; moyenne 421 "
    "contre médiane 37). La moyenne d'un genre décrit ses quelques blockbusters : on compare donc les genres "
    "par la **médiane** et l'**IQR** (survol des barres), et on garde les **sommes** pour les parts de marché "
    "et le cumul de Pareto, où les valeurs extrêmes sont justement l'information recherchée. "
    "Classe A = genres nécessaires pour atteindre 80 % des avis, B = 80 à 95 %, C = le reste.",
)

# --- 2. Dynamique par année de sortie (LAG) -------------------------------------------------------------
st.header("2. Satisfaction par année de sortie")
reliable_only = st.toggle("Années fiables uniquement (≥ 30 jeux et couverture des avis ≥ 50 %)", value=True)
tr = trends[trends["release_year"].between(*years) & trends["recommendation_rate"].notna()]
if reliable_only:
    tr = tr[tr["is_reliable"]]
c1, c2 = st.columns([3, 2])
long = tr.melt(
    id_vars="release_year",
    value_vars=["recommendation_rate", "recommendation_rate_ma3"],
    var_name="série",
    value_name="taux",
).replace({"recommendation_rate": "Taux annuel", "recommendation_rate_ma3": "Moyenne glissante 3 ans"})
fig = px.line(
    long,
    x="release_year",
    y="taux",
    color="série",
    markers=True,
    color_discrete_sequence=pal["line"],
    labels={"release_year": "Année de sortie", "taux": "Avis positifs (pondéré)", "série": ""},
    title="Taux de recommandation par cohorte de sortie",
)
fig.update_traces(line_width=2, marker_size=8)
fig.update_layout(
    height=380,
    yaxis_tickformat=".0%",
    hovermode="x unified",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
)
c1.plotly_chart(fig, width="stretch")
fig = px.bar(
    tr,
    x="release_year",
    y="n_games",
    color_discrete_sequence=[pal["bar"]],
    hover_data={"catalog_games": True, "review_coverage": ":.0%", "reviews_yoy_pct": ":+.1f"},
    labels={
        "release_year": "Année de sortie",
        "n_games": "Jeux avec avis",
        "catalog_games": "Jeux au catalogue",
        "review_coverage": "Couverture",
        "reviews_yoy_pct": "Avis vs N-1 (%)",
    },
    title="Jeux sortis et commentés par année",
)
fig.update_layout(height=380)
c2.plotly_chart(fig, width="stretch")
explain(
    """
SELECT release_year, n_games, catalog_games, try_divide(n_games, catalog_games) AS review_coverage,
       recommendation_rate,
       ROUND((recommendation_rate - LAG(recommendation_rate, 1) OVER w) * 100, 2) AS recommendation_delta_pts,
       ROUND(try_divide(total_reviews - LAG(total_reviews, 1) OVER w, LAG(total_reviews, 1) OVER w) * 100, 1)
         AS reviews_yoy_pct,
       AVG(recommendation_rate) OVER (ORDER BY release_year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
         AS recommendation_rate_ma3,
       n_games >= 30 AND try_divide(n_games, catalog_games) >= 0.5 AS is_reliable
FROM filled                       -- calendrier complet (SEQUENCE) : LAG compare 2 années consécutives
WINDOW w AS (ORDER BY release_year);   -- table Gold : /lakehouse/gold/release_year_trends
""",
    "**Justification :** les avis n'ont pas d'horodatage ; l'axe temporel est l'année de sortie du jeu "
    "(cohortes). La couverture du corpus d'avis s'effondre après 2016 (3 % des sorties 2017, 1 % en 2018-2019) : "
    "les variations calculées par `LAG` y sont un **artefact de collecte**, d'où le filtre `is_reliable`. "
    "Sur 2006-2016, la moyenne glissante perd environ 8 points pendant que l'offre est multipliée par 8 "
    "(ouverture de Steam aux indépendants) ; une partie de l'écart vient du **biais du survivant** "
    "(les vieux jeux encore commentés sont des classiques).",
)

# --- 3. Segmentation K-Means et temps de jeu -----------------------------------------------------------
st.header("3. Segmentation des jeux (K-Means) et temps de jeu")
seg_colors = dict(zip(SEGMENT_ORDER, pal["segments"]))
profile = (
    games.assign(n_pos=games["positive_ratio"] * games["n_reviews"])
    .groupby("segment")
    .agg(
        jeux=("app_id", "size"),
        avis=("n_reviews", "sum"),
        n_pos=("n_pos", "sum"),
        avis_median=("n_reviews", "median"),
        temps_median_h=("median_playtime", lambda s: s.median() / 60),
        prix_median=("price", "median"),
        age_median=("game_age", "median"),
        utiles_median=("helpful_ratio", "median"),
        contre_courant_median=("contrarian_ratio", "median"),
    )
    .reindex([s for s in SEGMENT_ORDER if s in seg_sel])
    .dropna(subset=["jeux"])
)
profile["part_avis"] = profile["avis"] / profile["avis"].sum()
profile["avis_positifs"] = profile["n_pos"] / profile["avis"]
profile = profile.drop(columns="n_pos")

c1, c2 = st.columns(2)
fig = px.box(
    games,
    x="positive_ratio",
    y="segment",
    orientation="h",
    color="segment",
    color_discrete_map=seg_colors,
    category_orders={"segment": SEGMENT_ORDER},
    points=False,
    hover_data={"segment": False},
    labels={"segment": "", "positive_ratio": "Avis positifs par jeu"},
    title="Satisfaction par segment (distribution par jeu)",
)
fig.update_layout(height=430, showlegend=False, xaxis_tickformat=".0%", xaxis_dtick=0.25)
c1.plotly_chart(fig, width="stretch")

bands = (
    games.assign(n_pos=games["positive_ratio"] * games["n_reviews"])
    .groupby("playtime_band")
    .agg(jeux=("app_id", "size"), avis=("n_reviews", "sum"), n_pos=("n_pos", "sum"))
    .reindex(PLAYTIME_ORDER)
    .dropna()
    .reset_index()
)
bands["avis_positifs"] = bands["n_pos"] / bands["avis"]
fig = px.bar(
    bands,
    x="playtime_band",
    y="avis_positifs",
    text=bands["avis_positifs"].map("{:.1%}".format),
    color_discrete_sequence=[pal["bar"]],
    hover_data={"jeux": True, "avis": True, "avis_positifs": ":.1%"},
    labels={"playtime_band": "Temps de jeu médian par joueur", "avis_positifs": "Avis positifs (pondéré)"},
    title="Temps de jeu et satisfaction",
)
fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_layout(height=430, yaxis_tickformat=".0%", yaxis_range=[0.5, 1])
c2.plotly_chart(fig, width="stretch")

st.dataframe(
    profile,
    width="stretch",
    column_config={
        "jeux": st.column_config.NumberColumn("Jeux", format="%d"),
        "avis": st.column_config.NumberColumn("Avis", format="%d"),
        "part_avis": st.column_config.NumberColumn("Part des avis", format="percent"),
        "avis_positifs": st.column_config.NumberColumn("Avis positifs (pondéré)", format="percent"),
        "utiles_median": st.column_config.NumberColumn("Avis utiles (médiane)", format="percent"),
        "contre_courant_median": st.column_config.NumberColumn(
            "Avis à contre-courant (méd.)", format="percent"
        ),
        "temps_median_h": st.column_config.NumberColumn("Temps de jeu médian (h)", format="%.1f"),
        "prix_median": st.column_config.NumberColumn("Prix médian (£)", format="%.2f"),
        "age_median": st.column_config.NumberColumn("Âge médian (ans)", format="%.0f"),
        "avis_median": st.column_config.NumberColumn("Avis médian / jeu", format="%.0f"),
    },
)
explain(
    """
SELECT segment, COUNT(*) AS jeux, SUM(n_reviews) AS avis,
       SUM(positive_ratio * n_reviews) / SUM(n_reviews) AS avis_positifs,
       PERCENTILE_APPROX(median_playtime, 0.5) / 60 AS temps_median_h,
       PERCENTILE_APPROX(price, 0.5) AS prix_median,
       PERCENTILE_APPROX(helpful_ratio, 0.5) AS utiles_median
FROM delta.`/lakehouse/gold/game_segments`
WHERE primary_genre IN (:genres) AND release_year BETWEEN :annee_min AND :annee_max
GROUP BY segment;
-- Segments produits en J4 par un Pipeline Spark ML :
-- VectorAssembler(log1p(n_reviews), positive_ratio, log1p(median_playtime), log1p(price), log1p(game_age))
--   -> StandardScaler(withMean, withStd) -> KMeans(k = 5, seed = 42)
""",
    "**Justification :** les 4 variables positives très asymétriques (skewness de 1,5 à 29) passent en `log1p` "
    "puis sont centrées-réduites, sinon le K-Means (distance euclidienne) serait dominé par le volume d'avis. "
    "**k = 5** : coude de l'inertie et silhouette de 0,346 ; un K-Means scikit-learn indépendant retrouve "
    "la même partition (ARI = 0,983). Les noms viennent des centroïdes et sont vérifiés par `assert`. "
    "Les profils comparent des **médianes** (distributions asymétriques) et un taux **pondéré par les avis**. "
    "Temps de jeu : `median_playtime = 0` signifie *non mesuré* par SteamSpy (petits jeux), ces jeux sont "
    "exclus et non imputés ; le lien temps de jeu ↔ satisfaction est significatif mais faible "
    "(Spearman ρ = +0,13, p < 10⁻¹⁴). Un avis négatif est 1,9 fois plus souvent jugé utile qu'un positif "
    "(χ², V de Cramér = 0,12).",
)

# --- 4. Vitalité des jeux (« toujours joué » vs « mort ») ------------------------------------------------
st.header("4. Vitalité des jeux : toujours joués ou abandonnés ?")
mix = (
    games.groupby(["segment", "vitality_profile"])
    .size()
    .rename("jeux")
    .reset_index()
    .assign(part=lambda d: d["jeux"] / d.groupby("segment")["jeux"].transform("sum"))
)
fig = px.bar(
    mix,
    x="part",
    y="segment",
    color="vitality_profile",
    orientation="h",
    color_discrete_sequence=pal["vitality"],
    category_orders={"segment": SEGMENT_ORDER, "vitality_profile": VITALITY_ORDER},
    text=mix["part"].map(lambda v: f"{v:.0%}" if v >= 0.08 else ""),
    hover_data={"jeux": True, "part": ":.1%"},
    labels={"part": "Part des jeux du segment", "segment": "", "vitality_profile": "Profil de vitalité"},
)
fig.update_traces(marker_line_width=2, marker_line_color="rgba(0,0,0,0)")
fig.update_layout(
    height=400, barmode="stack", xaxis_tickformat=".0%", legend={"orientation": "h", "y": 1.15}, bargap=0.3
)
st.plotly_chart(fig, width="stretch")
vit = (
    games.assign(n_pos=games["positive_ratio"] * games["n_reviews"])
    .groupby("vitality_profile")
    .agg(jeux=("app_id", "size"), avis=("n_reviews", "sum"), n_pos=("n_pos", "sum"))
    .reindex(VITALITY_ORDER)
    .dropna()
)
vit["avis_positifs"] = vit["n_pos"] / vit["avis"]
st.dataframe(
    vit.drop(columns="n_pos"),
    width="stretch",
    column_config={
        "avis_positifs": st.column_config.NumberColumn("Avis positifs (pondéré)", format="percent")
    },
)
explain(
    """
SELECT vitality_profile, COUNT(*) AS jeux, SUM(n_reviews) AS avis,
       SUM(positive_ratio * n_reviews) / SUM(n_reviews) AS avis_positifs
FROM delta.`/lakehouse/gold/game_segments`
GROUP BY vitality_profile;
-- Règles (J4) : Niche confidentielle si owners_mid <= 35 000 ; sinon âge (2019 - release_year) >= 5 ans
-- et temps de jeu médian >= 227 min (médiane de l'échantillon) -> Classique durable ;
-- âge < 5 ans et temps >= 227 min -> Hit récent ; sinon -> Vite abandonné / jeu mort
""",
    "**Justification :** sans historique d'activité, la vitalité combine l'**âge** du jeu, son **temps de jeu "
    "médian** (rétention) et sa **base de joueurs**. Les seuils sont des médianes de l'échantillon, donc "
    "robustes aux extrêmes. Les profils diffèrent significativement en satisfaction "
    "(Kruskal-Wallis H = 43,5, p ≈ 2·10⁻⁹) : les classiques durables sont les mieux notés (≈ 86 % d'avis "
    "positifs). Croisés avec les segments, ils montrent que les hits premium mêlent classiques et hits récents, "
    "alors que les free-to-play sont surtout vite abandonnés.",
)

# --- 5. Meilleurs jeux par genre (ROW_NUMBER + borne de Wilson) ----------------------------------------
st.header("5. Meilleurs jeux par genre")
top_genres = [g for g in genres if g in set(top["primary_genre"])]
if top_genres:
    genre = st.selectbox("Genre", top_genres)
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
            "rank_in_genre": "Rang qualité",
            "game_name": "Jeu",
            "release_year": st.column_config.NumberColumn("Sortie", format="%d"),
            "price": st.column_config.NumberColumn("Prix (£)", format="%.2f"),
            "n_reviews": "Avis",
            "recommendation_rate": st.column_config.NumberColumn("Avis positifs", format="percent"),
            "wilson_lower_bound": st.column_config.NumberColumn("Borne de Wilson 95 %", format="%.3f"),
            "volume_rank_in_genre": "Rang volume",
        },
    )
explain(
    """
WITH scored AS (
    SELECT *, (recommendation_rate + 1.9208 / n_reviews
               - 1.96 * SQRT((recommendation_rate * (1 - recommendation_rate) + 0.9604 / n_reviews) / n_reviews))
              / (1 + 3.8416 / n_reviews) AS wilson_lower_bound
    FROM v_games WHERE n_reviews >= 50
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY primary_genre
                                 ORDER BY wilson_lower_bound DESC, n_reviews DESC, app_id) AS rank_in_genre
    FROM scored
)
SELECT * FROM ranked WHERE rank_in_genre <= 10;   -- table Gold : /lakehouse/gold/top_games_by_genre
""",
    "**Justification :** trier sur le taux brut placerait en tête des jeux à 100 % sur une poignée d'avis. "
    "La **borne basse de l'intervalle de Wilson à 95 %** pénalise l'incertitude des petits effectifs "
    "(Half-Life 2, 99,98 % sur 4 052 avis, passe devant un jeu à 100 % sur 60 avis). `ROW_NUMBER` partitionné "
    "par genre donne exactement 10 jeux par genre ; le *rang volume* montre que qualité et popularité diffèrent.",
)
