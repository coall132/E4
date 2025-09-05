# tests/test_crud_load_df.py
import numpy as np
import pandas as pd
from types import SimpleNamespace

from API import CRUD

def _fake_catalog():
    df_etab = pd.DataFrame([
        {"id_etab": 1, "rating": 4.2, "priceLevel": "PRICE_LEVEL_MODERATE",
         "latitude": 47.39, "longitude": 0.69, "adresse": "20 Rue AAA, 37000 Tours",
         "editorialSummary_text": "Italien cosy", "start_price": 18.0},
        {"id_etab": 2, "rating": 3.6, "priceLevel": "PRICE_LEVEL_INEXPENSIVE",
         "latitude": 47.40, "longitude": 0.68, "adresse": "5 Ave BBB, 37100 Tours",
         "editorialSummary_text": "Terrasse sympa", "start_price": 12.0},
        {"id_etab": 3, "rating": None, "priceLevel": None,  # test imputation
         "latitude": 47.41, "longitude": 0.67, "adresse": "10 Rue CCC, 37200 Tours",
         "editorialSummary_text": "Gastro", "start_price": None},              # test fillna
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

def test_load_df_happy_path(monkeypatch):
    df_etab, df_options, df_embed, df_open = _fake_catalog()

    def fake_extract_table(_engine, name: str):
        if name == "etab":             return df_etab
        if name == "options":          return df_options
        if name == "etab_embedding":   return df_embed
        if name == "opening_period":   return df_open
        raise KeyError(name)

    # CRUD.load_df() référence CRUD.db.extract_table → on monkeypatch "db"
    monkeypatch.setattr(CRUD, "db", SimpleNamespace(extract_table=fake_extract_table, engine=None), raising=False)

    out = CRUD.load_df()
    # Forme générale
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 3
    for col in ["id_etab", "rating", "priceLevel", "latitude", "longitude",
                "editorialSummary_text", "start_price", "code_postal",
                "desc_embed", "rev_embeds"]:
        assert col in out.columns

    # Nettoyage / imputation
    assert out["rating"].isna().sum() == 0
    assert out["start_price"].isna().sum() == 0
    assert out["priceLevel"].isna().sum() == 0  # imputation probabiliste
    assert out["code_postal"].str.match(r"^\d{5}$").all()

    # Types embeddings
    assert out["desc_embed"].apply(lambda v: isinstance(v, np.ndarray) and v.ndim == 1).all()
    assert out["rev_embeds"].apply(lambda L: isinstance(L, list) and all(isinstance(x, np.ndarray) for x in L)).all()
