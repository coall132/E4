from typing import Any, List
import json
import numpy as np
import pandas as pd
from API import models

def determine_price_level(row):
    if pd.notna(row['start_price']):
        price = row['start_price']
        if price < 15:
            return 1
        elif price > 15:
            return 2
        elif price > 20:
            return 3
    else :
        return np.nan
    
def calculer_profil_ouverture(row, df_horaires, jours, creneaux):
    etab_id = row['id_etab']  
    profil = {'id_etab': etab_id}
    
    for j in jours.values():
        for c in creneaux.keys():
            profil[f"ouvert_{j}_{c}"] = 0
    
    horaires_etab = df_horaires[df_horaires['id_etab'] == etab_id]
    
    if horaires_etab.empty:
        return pd.Series(profil)

    for _, periode in horaires_etab.iterrows():
        if periode['open_day'] != periode['close_day']:
            continue
        jour_nom = jours.get(periode['open_day'])
        if not jour_nom:
            continue
        for nom_creneau, (debut, fin) in creneaux.items():
            if periode['open_hour'] < fin and periode['close_hour'] > debut:
                profil[f"ouvert_{jour_nom}_{nom_creneau}"] = 1
                
    return pd.Series(profil)


def to_np1d(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str) and x.strip().startswith("["):
        try:
            return np.asarray(json.loads(x), dtype=np.float32)
        except Exception:
            return None
    return None

def to_list_np(x):
    if isinstance(x, list):
        out = []
        for e in x:
            if isinstance(e, np.ndarray):
                out.append(e.astype(np.float32))
            elif isinstance(e, list):
                out.append(np.asarray(e, dtype=np.float32))
            elif isinstance(e, str) and e.strip().startswith("["):
                try:
                    out.append(np.asarray(json.loads(e), dtype=np.float32))
                except Exception:
                    pass
        return out if out else None
    if isinstance(x, str) and x.strip().startswith("["):
        try:
            L = json.loads(x)
            return [np.asarray(e, dtype=np.float32) for e in L]
        except Exception:
            return None
    return None


