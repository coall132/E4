import os
import uuid
import numpy as np
import pandas as pd
from types import SimpleNamespace

from API import CRUD, utils
from API.main import app
from API import models

def _fake_catalog():
    df_etab = pd.DataFrame([
        {"id_etab": 1, "rating": 4.2, "priceLevel": "PRICE_LEVEL_MODERATE",
         "latitude": 47.39, "longitude": 0.69, "adresse": "20 Rue AAA, 37000 Tours",
         "editorialSummary_text": "Italien cosy", "start_price": 18.0},
        {"id_etab": 2, "rating": 3.6, "priceLevel": "PRICE_LEVEL_INEXPENSIVE",
         "latitude": 47.40, "longitude": 0.68, "adresse": "5 Ave BBB, 37100 Tours",
         "editorialSummary_text": "Terrasse sympa", "start_price": 12.0},
        {"id_etab": 3, "rating": 4.8, "priceLevel": "PRICE_LEVEL_EXPENSIVE",
         "latitude": 47.41, "longitude": 0.67, "adresse": "10 Rue CCC, 37200 Tours",
         "editorialSummary_text": "Gastro", "start_price": 35.0},
    ])
    df_options = pd.DataFrame([
        {"id_etab": 1, "servesVegetarianFood": True,  "outdoorSeating": False, "restroom": True},
        {"id_etab": 2, "servesVegetarianFood": False, "outdoorSeating": True,  "restroom": True},
        {"id_etab": 3, "servesVegetarianFood": True,  "outdoorSeating": True,  "restroom": True},
    ])
    df_embed = pd.DataFrame([
        {"id_etab": 1, "desc_embed": np.array([1,0,0], np.float32),
         "rev_embeds": [np.array([0.2,0.8,0], np.float32)]},
        {"id_etab": 2, "desc_embed": np.array([0,1,0], np.float32),
         "rev_embeds": [np.array([0.5,0.5,0], np.float32)]},
        {"id_etab": 3, "desc_embed": np.array([0,0,1], np.float32),
         "rev_embeds": [np.array([0.9,0.1,0], np.float32)]},
    ])
    df_open = pd.DataFrame([
        {"id_etab": 1, "open_day": 6, "open_hour": 19, "close_day": 6, "close_hour": 22},
        {"id_etab": 2, "open_day": 6, "open_hour": 12, "close_day": 6, "close_hour": 14},
        {"id_etab": 3, "open_day": 6, "open_hour": 12, "close_day": 6, "close_hour": 23},
    ])
    return df_etab, df_options, df_embed, df_open

def prepare_catalog_in_app(monkeypatch):
    """Construit DF_CATALOG via CRUD.load_df() en mockant la DB, puis initialise l'état app."""
    df_etab, df_options, df_embed, df_open = _fake_catalog()

    def fake_extract_table(_engine, name: str):
        if name == "etab":             return df_etab
        if name == "options":          return df_options
        if name == "etab_embedding":   return df_embed
        if name == "opening_period":   return df_open
        raise KeyError(name)

    monkeypatch.setattr(CRUD, "db", SimpleNamespace(extract_table=fake_extract_table, engine=None), raising=False)
    df = CRUD.load_df()

    # Prime l'état de l'app (warmup est désactivé dans les tests)
    app.state.DF_CATALOG = df
    app.state.SENT_MODEL = utils._StubSentModel(dim=3)  # embeddings 3D ci-dessus
    # Ancres optionnelles (peut être None)
    try:
        from API.benchmark_2_0 import pick_anchors_from_df
        app.state.ANCHORS = pick_anchors_from_df(df, n=4)
    except Exception:
        app.state.ANCHORS = None

def register_and_token(client):
    suffix = uuid.uuid4().hex[:8]
    payload = {"username": f"user_{suffix}", "email": f"user_{suffix}@ex.com", "password": "P@ssw0rd!"}
    r = client.post("/users", json=payload)
    assert r.status_code == 200

    r = client.post("/auth/api-keys",
                    json={"email": payload["email"], "username": payload["username"]},
                    params={"password": os.getenv("API_STATIC_KEY", "coall")})
    assert r.status_code == 200
    api_key = r.json()["api_key"]

    r = client.post("/auth/token", headers={"X-API-KEY": api_key})
    assert r.status_code == 200
    return r.json()["access_token"]

def test_predict(client, db_session, monkeypatch):
    prepare_catalog_in_app(monkeypatch)
    token = register_and_token(client)

    e = models.Etablissement(id_etab=1, nom="R1", adresse="37000 Tours", rating=4.2, priceLevel="2")
    db_session.add(e); db_session.commit()

    form = {
        "price_level": 2,
        "code_postal": "37000",
        "open": "ouvert_samedi_soir",
        "options": ["servesVegetarianFood","outdoorSeating"],
        "description": "italien calme terrasse",
    }
    r = client.post("/predict",
                    headers={"Authorization": f"Bearer {token}"},
                    json=form,
                    params={"k": 2, "use_ml": True})  
    assert r.status_code == 200
    data = r.json()

    assert "prediction_id" in data
    assert data.get("k") == 2
    assert "items" in data and isinstance(data["items"], list)
    assert len(data["items"]) <= 2
    # Tri décroissant par score
    scores = [it["score"] for it in data["items"]]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    # Persistance
    pred_id = data["prediction_id"]
    pred = db_session.query(models.Prediction).filter(models.Prediction.id == pred_id).first()
    assert pred is not None
    assert len(pred.items) == len(data["items"])

def test_predict_fails_when_catalog_empty(client, monkeypatch):
    # Force un catalogue vide
    app.state.DF_CATALOG = pd.DataFrame()
    token = register_and_token(client)

    r = client.post("/predict",
                    headers={"Authorization": f"Bearer {token}"},
                    json={"description": "test"})
    assert r.status_code == 500
    assert "Catalogue vide" in r.text