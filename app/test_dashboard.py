"""Tests de fumée du dashboard sur les vraies tables Gold (lancés par `task test:dashboard`).

Le rendu est simulé sans navigateur par `streamlit.testing.v1.AppTest` : chaque test exécute le script
complet, vérifie l'absence d'exception puis manipule les filtres de la barre latérale.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).with_name("dashboard.py"))
TIMEOUT_S = 600  # premier rendu : démarrage de Spark + lecture des 4 tables Gold sur HDFS


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT_S).run()
    assert not at.exception, at.exception
    return at


def test_rendu_complet(app: AppTest) -> None:
    assert len(app.tabs) == 5
    assert len(app.get("plotly_chart")) >= 10
    assert len(app.expander) >= 7  # requête SQL + justification sous chaque analyse clé
    assert not app.error and not app.warning
    labels = {m.label: m.value for m in app.metric}
    assert labels["Jeux au catalogue"].replace("\u202f", "") == "27075"
    assert labels["Avis analysés"] == "3,76 M"


def test_filtres_reactifs(app: AppTest) -> None:
    before = {m.label: m.value for m in app.metric}["Avis positifs (filtre)"]
    genres = next(w for w in app.multiselect if w.label == "Genres")
    genres.set_value(["RPG"]).run()
    years = next(w for w in app.slider if w.label == "Années de sortie")
    years.set_value((2010, 2016)).run()
    assert not app.exception, app.exception
    after = {m.label: m.value for m in app.metric}["Avis positifs (filtre)"]
    assert after != before
    focus = next(w for w in app.selectbox if w.label == "Segment mis en avant")
    focus.set_value(focus.options[-1]).run()
    toggle = next(w for w in app.toggle if w.label.startswith("Années fiables"))
    toggle.set_value(False).run()
    assert not app.exception, app.exception


def test_filtre_vide_sans_exception(app: AppTest) -> None:
    segments = next(w for w in app.multiselect if w.label == "Segments K-Means")
    segments.set_value([]).run()
    assert not app.exception, app.exception
    assert app.warning, "Un filtre vide doit afficher un avertissement, pas une erreur"
