from fastapi import FastAPI, Depends, HTTPException, Security, status, Query, Request, Response
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import or_
from typing import List, Optional
import os, time
from datetime import timedelta, datetime, timezone
import pandas as pd
import numpy as np
from joblib import load
from sqlalchemy import MetaData, Table, select, outerjoin
import joblib
from pathlib import Path
import mlflow, os, uuid, json, time
from fastapi import Depends, Form
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, RedirectResponse

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI") 
MLFLOW_EXP = os.getenv("MLFLOW_EXPERIMENT", "reco-inference")

try :
    from . import utils
    from . import CRUD
    from . import models
    from . import schema
    from .database import engine, get_db, SessionLocal
    from .benchmark_2_0 import (
    score_func,
    build_item_features_df,
    aggregate_gains,
    W_eval,
    model as DEFAULT_SENT_MODEL,
    pick_anchors_from_df,
    build_item_features_df,
    make_preproc_final,  
)
except :
    from API import utils
    from API import CRUD
    from API import models
    from API import schema
    from API.database import engine, get_db, SessionLocal
    import API.benchmark_2_0 as bm
    score_func = bm.score_func
    build_item_features_df = bm.build_item_features_df
    aggregate_gains = bm.aggregate_gains
    W_eval = bm.W_eval
    DEFAULT_SENT_MODEL = getattr(bm, "model", None)
    pick_anchors_from_df = bm.pick_anchors_from_df
    make_preproc_final = bm.make_preproc_final

app = FastAPI(
    title="API Reco Restaurant",
    description="API pour recommander des restaurants",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

API_STATIC_KEY = os.getenv("API_STATIC_KEY", "coall")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

BASE_DIR = Path(__file__).resolve().parent  
STATIC_DIR = BASE_DIR / "static"                
TEMPLATES_DIR = BASE_DIR / "templates" 
PASSION_DIR = STATIC_DIR / "strategy"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/strategy", StaticFiles(directory=str(PASSION_DIR)), name="strategy")


models.ensure_ml_schema(engine)
models._attach_external_tables(engine)
models.Base.metadata.create_all(bind=engine)

@app.middleware("http")
async def check_auth_cookie(request: Request, call_next):
    # Protège uniquement les pages commençant par /app/
    if request.url.path.startswith("/app/"):
        token = request.cookies.get("auth_token")
        if not token:
            # Si pas de token, on redirige vers la page de connexion
            return RedirectResponse(url="/login")
    
    response = await call_next(request)
    return response

@app.on_event("startup")
def warmup():
    if os.getenv("DISABLE_WARMUP", "0") == "1":
        app.state.DF_CATALOG = pd.DataFrame()
        app.state.SENT_MODEL = utils._StubSentModel()
        app.state.PREPROC = None
        app.state.ML_MODEL = None
        app.state.FEATURE_COLS = []
        app.state.ANCHORS = None
        app.state.MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
        print("[startup] Warmup désactivé (DISABLE_WARMUP=1).")
        return
    
    try:
        df = CRUD.load_df()
        if df is None or df.empty:
            raise RuntimeError("DF_CATALOG vide")
        app.state.DF_CATALOG = df
    except Exception as e:
        print(f"[startup] Échec chargement DF_CATALOG: {e}")
        app.state.DF_CATALOG = pd.DataFrame()

    try:
        ml = CRUD.load_ML()
        app.state.PREPROC = getattr(ml, "preproc", None)
        app.state.PREPROC_FACTORY = getattr(ml, "preproc_factory", None)
        app.state.SENT_MODEL = getattr(ml, "sent_model", None)
        app.state.ML_MODEL = getattr(ml, "rank_model", None)
        # utile pour tracer la version
        app.state.MODEL_VERSION = (
            os.getenv("MODEL_VERSION")
            or getattr(ml, "rank_model_path", None)
            or "dev"
        )
        if app.state.SENT_MODEL is None:
            dim = utils._infer_embed_dim(app.state.DF_CATALOG, default=1024)
            app.state.SENT_MODEL = utils._StubSentModel(dim=dim)
        print(f"[startup] ML: PREPROC={type(app.state.PREPROC).__name__ if app.state.PREPROC else 'None'} | "
              f"MODEL={'ok' if app.state.ML_MODEL is not None else 'None'}")
    except Exception as e:
        print(f"[startup] Échec chargement ML: {e}")
        app.state.PREPROC = None
        app.state.SENT_MODEL = DEFAULT_SENT_MODEL
        app.state.ML_MODEL = None
        app.state.MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")

    df = app.state.DF_CATALOG
    anchors = pick_anchors_from_df(df, n=8) if not df.empty else None
    app.state.ANCHORS = anchors

    feature_cols = []
    if not df.empty:
        neutral_form = {
            "description": "",
            "price_level": np.nan,
            "code_postal": None,
            "options": [],
            "open": "",
        }
        X_probe_df, _ = build_item_features_df(df=df,form=neutral_form,sent_model=app.state.SENT_MODEL,
            include_query_consts=True,anchors=anchors,)
        feature_cols = [c for c in X_probe_df.columns if c != "id_etab"]
        app.state.FEATURE_COLS = feature_cols

        if app.state.PREPROC is None and feature_cols:
            app.state.PREPROC = make_preproc_final().fit(X_probe_df[feature_cols])
            print("[startup] PREPROC fallback créé et fit sur features de probe (DEV ONLY).")
    else:
        app.state.FEATURE_COLS = []

    anch_shape = None if anchors is None else getattr(anchors, "shape", None)
    print(f"[startup] OK | rows={len(df)} | features={len(feature_cols)} | anchors={anch_shape} |")

async def get_optional_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("auth_token")
    if not token:
        return None
    try:
        subject = await CRUD.get_current_subject(token)
        user_id = CRUD.current_user_id(subject)
        user = db.query(models.User).filter(models.User.id == user_id).first()
        return user
    except HTTPException:
        return None
    
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API des restaurants. Allez sur /docs pour voir les endpoints."}

@app.post("/auth/api-keys", response_model=schema.ApiKeyResponse, tags=["Auth"])
def create_api_key(API_key_in: schema.ApiKeyCreate,password:str, db: Session = Depends(get_db)):
    if password != API_STATIC_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Password invalide ou manquante.",
            headers={"WWW-Authenticate": "APIKey"},
        )
    user = db.query(models.User).filter(models.User.email == API_key_in.email).first()
    if user is None:
        if db.query(models.User).filter(models.User.username == API_key_in.username).first():
            raise HTTPException(status_code=409, detail="Ce username est déjà pris.")
        user = models.User(email=API_key_in.email, username=API_key_in.username)
        db.add(user)
        db.flush()

    api_key_plain, key_id, _secret = CRUD.generate_api_key()
    key_hash = CRUD.hash_api_key(api_key_plain)

    row = models.ApiKey(user_id=user.id,key_id=key_id,key_hash=key_hash or None)
    db.add(row)
    db.commit()
    return schema.ApiKeyResponse(api_key=api_key_plain, key_id=key_id)


@app.post("/auth/token", response_model=schema.TokenOut, tags=["Auth"])
def issue_token(API_key_in: Optional[str] = Security(api_key_header), db: Session = Depends(get_db)):
    row = CRUD.verify_api_key(db, API_key_in)
    token, exp_ts = CRUD.create_access_token(subject=f"user:{row.user_id}")
    return schema.TokenOut(access_token=token, expires_at=exp_ts)

@app.post("/predict", tags=["predict"], dependencies=[Depends(CRUD.get_current_subject)])
def predict(form: schema.Form,k: int = 3,use_ml: bool = True,user_id: int = Depends(CRUD.current_user_id),db: Session = Depends(get_db)):
    t0 = time.perf_counter()

    if (not hasattr(app.state, "DF_CATALOG")) or app.state.DF_CATALOG is None or app.state.DF_CATALOG.empty:
        raise HTTPException(500, "Catalogue vide/non chargé.")
    df = app.state.DF_CATALOG

    try:
        form_row = models.FormDB(price_level=form.price_level,city=form.city,open=form.open,options=form.options,description=form.description)
        db.add(form_row)
        db.flush()  
        form_id = form_row.id
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Insertion du formulaire impossible: {e}")

    anchors = getattr(app.state, "ANCHORS", None)
    X_df, gains_proxy = build_item_features_df(df=df,form=form.model_dump(),sent_model=app.state.SENT_MODEL,include_query_consts=True,anchors=anchors)

    used_ml = False
    scores = gains_proxy.copy()

    if use_ml and getattr(app.state, "ML_MODEL", None) is not None:
        model = app.state.ML_MODEL

        # Par défaut: on envoie les features BRUTES au modèle (DataFrame)
        X_input = X_df.copy()

        # N'utiliser notre PREPROC que si le modèle n’est PAS un pipeline scikit-learn
        is_pipeline = hasattr(model, "steps") or hasattr(model, "named_steps")
        if not is_pipeline and getattr(app.state, "PREPROC", None) is not None:
            feature_cols = getattr(app.state, "FEATURE_COLS", None) or [c for c in X_df.columns if c != "id_etab"]
            X_df_aligned = utils._align_df_to_cols(X_df.copy(), feature_cols)
            X_sp = app.state.PREPROC.transform(X_df_aligned)
            X_input = X_sp.toarray().astype(np.float32) if hasattr(X_sp, "toarray") else np.asarray(X_sp, dtype=np.float32)

        scores = utils._predict_scores(model, X_input)
        used_ml = True

    k = int(max(1, min(k, 50)))
    order = np.argsort(scores)[::-1]
    sel = order[:k]

    latency_ms = int((time.perf_counter() - t0) * 1000)
    model_version = os.getenv("MODEL_VERSION") or getattr(app.state, "MODEL_VERSION", None) or "dev"

    pred_row = models.Prediction(form_id=form_id,k=k,model_version=model_version,latency_ms=latency_ms,status="ok")
    if hasattr(models.Prediction, "user_id"):
        setattr(pred_row, "user_id", user_id)

    pred_row.items = []
    for r, i in enumerate(sel, start=1):
        etab_id = int(df.iloc[i]["id_etab"]) if "id_etab" in df.columns else int(i)
        pred_row.items.append(
            models.PredictionItem(rank=r,etab_id=etab_id,score=float(scores[i]),))

    try:
        db.add(pred_row)
        db.commit()
        db.refresh(pred_row)
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Insertion de la prédiction impossible: {e}")

    try:
        pyd_pred = schema.Prediction.model_validate(pred_row)
        CRUD.log_prediction_event(prediction=pyd_pred,form_dict=form.model_dump(),scores=np.asarray(scores, dtype=float),
            used_ml=used_ml,latency_ms=latency_ms,model_version=model_version,)
    except Exception as e:
        print(f"[mlflow] log_prediction_event failed: {e}")

    base = schema.Prediction.model_validate(pred_row).model_dump()
    pred_id = str(pred_row.id)
    base.setdefault("id", pred_id)
    base["prediction_id"] = pred_id

    ids = [int(it["etab_id"]) for it in base.get("items", [])]
    details_map = CRUD.get_etablissements_details_bulk(db, ids)

    items_rich = []
    for it in base.get("items", []):
        d = details_map.get(int(it["etab_id"]))
        items_rich.append({**it, "details": d})

    base["items_rich"] = items_rich
    base["message"] = "N’hésitez pas à donner un feedback (0 à 5) via /feedback en utilisant prediction_id."
    return base


@app.post("/feedback", response_model=schema.FeedbackOut, tags=["monitoring"])
def submit_feedback(payload: schema.FeedbackIn,sub: str = Depends(CRUD.get_current_subject),
                    db: Session = Depends(get_db),user_id: int = Depends(CRUD.current_user_id)):
    pred = db.query(models.Prediction).options(selectinload(models.Prediction.items)).filter(models.Prediction.id == payload.prediction_id).first()
    if not pred:
        raise HTTPException(404, "Prediction introuvable")
    
    if getattr(pred, "user_id", None) not in (None, user_id):
        raise HTTPException(status_code=403, detail="Cette prédiction n'appartient pas à l'utilisateur courant")
    

    row = models.Feedback(prediction_id=pred.id,rating=payload.rating,comment=payload.comment)
    db.add(row); db.commit()

    try:
        CRUD.log_feedback_rating(prediction_id=str(pred.id),rating=payload.rating,k=pred.k,model_version=pred.model_version,
        user_id=user_id,comment=payload.comment,use_active_run_if_any=True)
    except Exception as e:
        print(f"[mlflow] log_feedback_rating failed: {e}")

    return schema.FeedbackOut()

@app.post("/auth/web/token", response_model=schema.TokenOut, tags=["Auth"])
def login_for_web_access_token(
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    user = CRUD.get_user_by_username(db, username=form_data.username)
    if not user or not user.hashed_password or not CRUD.ph.verify(user.hashed_password, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, exp_ts = CRUD.create_access_token(subject=f"user:{user.id}")
    return schema.TokenOut(access_token=token, expires_at=exp_ts)


@app.post("/users", response_model=schema.UserOut, tags=["Auth"])
def register_user(user: schema.UserCreate, db: Session = Depends(get_db)):
    # Vérifie l’unicité du nom d’utilisateur et de l’email
    if CRUD.get_user_by_username(db, username=user.username):
        raise HTTPException(status_code=409, detail="Ce nom d'utilisateur est déjà pris.")
    existing = CRUD.get_user_by_email(db, email=user.email)
    if existing and existing.hashed_password:
        raise HTTPException(status_code=409, detail="Cet email est déjà utilisé.")
    return CRUD.create_user(db=db, user=user)

@app.post("/auth/web/login", tags=["Auth"])
def web_login_and_set_cookie(
    response: Response,
    db: Session = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    user = CRUD.get_user_by_username(db, username=form_data.username)
    if not user or not user.hashed_password or not CRUD.ph.verify(user.hashed_password, form_data.password):
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    token, exp_ts = CRUD.create_access_token(subject=f"user:{user.id}")
    # Pose un cookie HttpOnly
    response.set_cookie(
        key="auth_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,  # passe à True si tu es derrière HTTPS
        max_age=60*60*24*7,  # par ex. 7 jours
    )
    return {"ok": True, "expires_at": exp_ts}

# GET /login : page de connexion
@app.get("/login", name="login_page", response_class=HTMLResponse)
def ui_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "ACTIVE": "login"})

# GET /register : page de création de compte
@app.get("/register", name="register_page", response_class=HTMLResponse)
def ui_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "ACTIVE": "register"})

@app.post("/logout")
def logout(response: Response):
    response.delete_cookie("auth_token")
    return {"ok": True}
