"""
Crée une clé API, obtient un access token OAuth2 et appelle l'endpoint
'/history/predictions' protégé par Authorization: Bearer <token>.
"""

import os
import uuid


def test_history_predictions_flow(client, db_session):
    from API import models

    # 1) user web
    suffix = uuid.uuid4().hex[:8]
    u_payload = {
        "username": f"user_{suffix}",
        "email": f"user_{suffix}@example.com",
        "password": "P@ssw0rd!",
    }
    r_user = client.post("/users", json=u_payload)
    assert r_user.status_code == 200

    # 2) API key
    r_key = client.post("/auth/api-keys",
                        json={"email": u_payload["email"], "username": u_payload["username"]},
                        params={"password": os.getenv("API_STATIC_KEY", "coall")})
    assert r_key.status_code == 200
    key = r_key.json()["api_key"]

    # 3) token OAuth2
    r_tok = client.post("/auth/token", headers={"X-API-KEY": key})
    assert r_tok.status_code == 200
    token = r_tok.json()["access_token"]

    # 4) seed une prédiction liée à ce user
    user = db_session.query(models.User).filter(models.User.email == u_payload["email"]).first()
    assert user is not None

    form = models.FormDB(price_level=2, city="Tours", open="soir",
                         options={"reservable": True}, description="italien")
    db_session.add(form); db_session.flush()

    etab = models.Etablissement(id_etab=999001, nom="Test Resto", adresse="37000 Tours",
                                rating=4.5, priceLevel="2")
    db_session.add(etab); db_session.flush()

    pred = models.Prediction(form_id=form.id, user_id=user.id, k=2, model_version="dev",
                             latency_ms=12, status="ok")
    db_session.add(pred); db_session.flush()

    db_session.add_all([
        models.PredictionItem(prediction_id=pred.id, rank=1, etab_id=etab.id_etab, score=0.9),
        models.PredictionItem(prediction_id=pred.id, rank=2, etab_id=etab.id_etab, score=0.7),
    ])
    db_session.commit()

    # 5) call endpoint protégé
    r = client.get("/history/predictions", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    arr = r.json()
    assert isinstance(arr, list) and len(arr) >= 1
    assert any(x["id"] == str(pred.id) for x in arr)
