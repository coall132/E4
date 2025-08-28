"""
Tests d'intégration sur les endpoints UI de restaurants (/restaurant/{id}, /restaurant/{id}/reviews).
"""

from datetime import datetime
import pytest
from fastapi.testclient import TestClient
import pytest

@pytest.fixture
def seed_restaurant(db_session, app_and_db):
    _, _, models = app_and_db
    # crée un établissement minimal + options + horaires + 2 reviews
    etab = models.Etablissement(
        id_etab=12345,
        nom="Chez Test",
        adresse="10 rue de la Test, 37000 Tours",
        internationalPhoneNumber="+33 2 47 00 00 00",
        description="Cuisine locale",
        websiteUri="https://example.com",
        latitude=47.360,
        longitude=0.700,
        rating=4.2,
        priceLevel="2",
        start_price=18.0,
        end_price=32.0,
        editorialSummary_text="Une adresse sympa",
        google_place_id="test_place_12345",
    )
    db_session.add(etab)
    db_session.flush()

    options = models.Options(
        id_etab=etab.id_etab,
        delivery=True,
        goodForChildren=False,
        reservable=True,
        restroom=True,
        servesDinner=True,
    )
    db_session.add(options)

    # Horaires : lundi 12:00–14:00, 19:00–22:00
    from API.models import OpeningPeriod
    db_session.add_all([
        OpeningPeriod(id_etab=etab.id_etab, open_day=1, close_day=1,
                      open_hour=12, open_minute=0, close_hour=14, close_minute=0),
        OpeningPeriod(id_etab=etab.id_etab, open_day=1, close_day=1,
                      open_hour=19, open_minute=0, close_hour=22, close_minute=0),
    ])

    # Reviews (ordre par publishTime desc attendu)
    from API.models import Review
    db_session.add_all([
        Review(id_etab=etab.id_etab, original_languageCode="fr",
               original_text="Très bon!", publishTime="2024-12-12T12:00:00Z",
               rating=5.0, relativePublishTimeDescription="il y a 8 mois", author="Alice"),
        Review(id_etab=etab.id_etab, original_languageCode="fr",
               original_text="Correct", publishTime="2024-10-01T09:00:00Z",
               rating=3.5, relativePublishTimeDescription="il y a 10 mois", author="Bob"),
    ])

    db_session.commit()
    return etab.id_etab


def test_restaurant_detail_and_reviews(client, seed_restaurant):
    # client fourni ici via fixture locale pour éviter import circulaire
    def _client(app_and_db):
        app, _, _ = app_and_db
        return TestClient(app)

    app_and_db = pytest.global_dict["app_and_db"] if hasattr(pytest, "global_dict") else None
    assert app_and_db is not None, "fixture app_and_db introuvable"

    c = _client(app_and_db)

    # /restaurant/{id}
    r1 = c.get(f"/restaurant/{seed_restaurant}")
    assert r1.status_code == 200
    js = r1.json()
    assert js["id"] == seed_restaurant
    assert js["nom"] == "Chez Test"
    assert js["options"]["delivery"] is True
    # horaires formatés
    assert "Lundi" in js["horaires"]
    assert "12:00–14:00" in js["horaires"]["Lundi"]
    assert "19:00–22:00" in js["horaires"]["Lundi"]

    # /restaurant/{id}/reviews
    r2 = c.get(f"/restaurant/{seed_restaurant}/reviews")
    assert r2.status_code == 200
    reviews = r2.json()
    assert len(reviews) == 2
    assert reviews[0]["author"] == "Alice"
    assert reviews[1]["author"] == "Bob"
