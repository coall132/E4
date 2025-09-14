def test_restaurant_detail_and_reviews(client, db_session):
    # Import local pour éviter import au module
    from API import models

    # Seed : etab + options + horaires + 2 reviews
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
    db_session.add(etab); db_session.flush()

    options = models.Options(
        id_etab=etab.id_etab,
        delivery=True,
        goodForChildren=False,
        reservable=True,
        restroom=True,
        servesDinner=True,
    )
    db_session.add(options)

    from API.models import OpeningPeriod, Review
    db_session.add_all([
        OpeningPeriod(id_etab=etab.id_etab, open_day=1, close_day=1,
                      open_hour=12, open_minute=0, close_hour=14, close_minute=0),
        OpeningPeriod(id_etab=etab.id_etab, open_day=1, close_day=1,
                      open_hour=19, open_minute=0, close_hour=22, close_minute=0),
        Review(id_etab=etab.id_etab, original_languageCode="fr",
               original_text="Très bon!", publishTime="2024-12-12T12:00:00Z",
               rating=5.0, relativePublishTimeDescription="il y a 8 mois", author="Alice"),
        Review(id_etab=etab.id_etab, original_languageCode="fr",
               original_text="Correct", publishTime="2024-10-01T09:00:00Z",
               rating=3.5, relativePublishTimeDescription="il y a 10 mois", author="Bob"),
    ])
    db_session.commit()

    # /restaurant/{id}
    r1 = client.get(f"/restaurant/{etab.id_etab}")
    assert r1.status_code == 200
    js = r1.json()
    assert js["id"] == etab.id_etab
    assert js["nom"] == "Chez Test"
    assert js["options"]["delivery"] is True
    assert "Lundi" in js["horaires"]
    assert "12:00–14:00" in js["horaires"]["Lundi"]
    assert "19:00–22:00" in js["horaires"]["Lundi"]

    # /restaurant/{id}/reviews
    r2 = client.get(f"/restaurant/{etab.id_etab}/reviews")
    assert r2.status_code == 200
    reviews = r2.json()
    assert len(reviews) == 2
    assert reviews[0]["author"] in {"Alice", "Bob"}  # ordre défini par l'API
