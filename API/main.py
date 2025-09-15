from fastapi import FastAPI, Depends, HTTPException, Security, status, Query, Request, Response,Body
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from sqlalchemy.orm import Session, selectinload
from sqlalchemy import or_, cast, Integer
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
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Gauge
from redis.asyncio import Redis
from fastapi import Form, Request, HTTPException, Depends
from pydantic import ValidationError
import re, asyncio
from API.security.turnstile import verify_turnstile
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import PlainTextResponse
import redis.asyncio as redis
import logging
import sys
import traceback
from fastapi.openapi.utils import get_openapi
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI") 
MLFLOW_EXP = os.getenv("MLFLOW_EXPERIMENT", "reco-inference")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
BUCKET_SEC = int(os.getenv("SIGNUP_BUCKET_SEC", "10")) 
WINDOW_SEC = int(os.getenv("SIGNUP_UNIQUE_IPS_WINDOW_SEC", "300"))  # 5 min
NUM_BUCKETS = WINDOW_SEC // BUCKET_SEC
PATH_RE = re.compile(os.getenv("SIGNUP_PATH_REGEX", r"^/(users|register)$"))
METRICS_LEADER = os.getenv("METRICS_LEADER", "1") == "1"
PORT = int(os.getenv("EXPORTER_PORT", "9109"))
BYPASS = os.getenv("TURNSTILE_DEV_BYPASS", "0") 

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


tags_metadata = [
    {"name": "Auth", "description": "Endpoints d’inscription/connexion pour l’UI (captcha, etc.)."},
    {"name": "predict", "description": "Recommandations d’établissements."},
    {"name": "monitoring", "description": "Feedback et suivi qualité."},
    {"name": "ui", "description": "Pages HTML pour l’interface web."},
]

app = FastAPI(
    title="API Reco Restaurant (E4)",
    description="API pour recommander des restaurants (avec endpoints UI / rate-limit / captcha).",
    version="1.0.0",
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={
        "displayRequestDuration": True,
        "docExpansion": "list",
        "defaultModelsExpandDepth": 1,
        "defaultModelExpandDepth": 1,
        "tryItOutEnabled": True,
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
    comps = schema.setdefault("components", {}).setdefault("securitySchemes", {})
    comps["BearerAuth"] = {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
    schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

def client_ip(request):
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host

limiter = Limiter(key_func=client_ip,storage_uri=REDIS_URL,strategy="moving-window", default_limits=["60/minute"])
app.state.limiter = limiter
instrumentator = Instrumentator(should_group_status_codes=True,should_ignore_untemplated=True)
instrumentator.instrument(app).expose(app, endpoint="/metrics", tags=["metrics"])

API_STATIC_KEY = os.getenv("API_STATIC_KEY")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


BASE_DIR = Path(__file__).resolve().parent  
STATIC_DIR = BASE_DIR / "static"                
TEMPLATES_DIR = BASE_DIR / "templates" 
PASSION_DIR = STATIC_DIR / "strategy"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/strategy", StaticFiles(directory=str(PASSION_DIR)), name="strategy")



def safe_db_bootstrap(engine):
    if os.getenv("DISABLE_DB_INIT", "0") == "1":
        print("[startup] DB init disabled (DISABLE_DB_INIT=1)")
        return

    dialect = engine.dialect.name
    if dialect in ("postgresql", "postgres"):
        models.ensure_ml_schema(engine)
        if getattr(models, "_attach_external_tables", None):
            try:
                models._attach_external_tables(engine)
            except Exception as e:
                print(f"[startup] _attach_external_tables skipped: {e}")
        models.Base.metadata.create_all(bind=engine, checkfirst=True)
    else:
        models.Base.metadata.create_all(bind=engine, checkfirst=True)

app.state.redis = Redis.from_url(REDIS_URL, decode_responses=True)
Z_UNIQUE = "signup:ip_last_seen" 
H_BUCKET = "signup:b:"           

@app.exception_handler(RateLimitExceeded)
def ratelimit_handler(request: Request, exc):
    return PlainTextResponse("Too Many Requests", status_code=429)

def _client_ip(req: Request) -> str:
    xff = req.headers.get("x-forwarded-for")
    if xff: return xff.split(",")[0].strip()
    xr = req.headers.get("x-real-ip")
    if xr: return xr.strip()
    return req.client.host

def _curr_bucket(now: int) -> int:
    return (now // BUCKET_SEC) * BUCKET_SEC

async def _incr_ip_bucket(ip: str, now: int):
    b = _curr_bucket(now)
    hkey = f"{H_BUCKET}{b}"
    await redis.hincrby(hkey, ip, 1)
    await redis.expire(hkey, WINDOW_SEC + BUCKET_SEC + 5)


@app.on_event("shutdown")
async def shutdown():
    r = getattr(app.state, "redis", None)
    if r:
        await r.aclose()

@app.middleware("http")
async def block_banned_ip(request: Request, call_next):
    ip = request.client.host if request.client else ""
    r = getattr(request.app.state, "redis", None)
    try:
        if r and ip and await r.sismember("banned_ips", ip):
            return templates.TemplateResponse("ban.html")
    except Exception:
        pass
    return await call_next(request)

@app.middleware("http")
async def _track_signup(request: Request, call_next):
    try:
        if request.method == "POST" and PATH_RE.match(request.url.path):
            ip = _client_ip(request)
            now = int(time.time())
            await redis.zadd(Z_UNIQUE, {ip: now})
            if now % 5 == 0:
                await redis.zremrangebyscore(Z_UNIQUE, 0, now - WINDOW_SEC)
            await _incr_ip_bucket(ip, now)
    except Exception:
        pass 
    return await call_next(request)

@app.middleware("http")
async def check_auth_cookie(request: Request, call_next):
    if request.url.path.startswith("/app/"):
        token = request.cookies.get("auth_token")
        if not token:
            return RedirectResponse(url="/login")
    
    response = await call_next(request)
    return response


@app.on_event("startup")
def warmup():
    safe_db_bootstrap(engine)
    if os.getenv("DISABLE_WARMUP", "0") == "1":
        app.state.DF_CATALOG = pd.DataFrame()
        app.state.SENT_MODEL = utils._StubSentModel()
        app.state.PREPROC = None
        app.state.ML_MODEL = None
        app.state.FEATURE_COLS = []
        app.state.ANCHORS = None
        app.state.X_ITEMS = None
        app.state.MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")
        print("[startup] Warmup désactivé (DISABLE_WARMUP=1).")
        return
    
    try:
        df = CRUD.load_df()
        if df is None or df.empty:
            raise HTTPException(500, "Catalogue vide/non chargé.")
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
        app.state.MODEL_VERSION = (
            os.getenv("MODEL_VERSION")
            or getattr(ml, "rank_model_path", None)
            or "dev"
        )
        if app.state.SENT_MODEL is None:
            dim = utils._infer_embed_dim(app.state.DF_CATALOG, default=1024)
            app.state.SENT_MODEL = utils._StubSentModel(dim=dim)
            print("Model fake embedding ___________________")
        print(f"[startup] ML: PREPROC={type(app.state.PREPROC).__name__ if app.state.PREPROC else 'None'} | "
              f"MODEL={'ok' if app.state.ML_MODEL is not None else 'None'} |"
              f"EMbedding ={'ok' if app.state.SENT_MODEL is not None else 'None'}")
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

        if app.state.ML_MODEL is None:
            if app.state.PREPROC is None and feature_cols:
                app.state.PREPROC = make_preproc_final().fit(X_probe_df[feature_cols])
                print("[startup] PREPROC fallback créé et fit sur features de probe (dev only)")
    else:
        app.state.FEATURE_COLS = []

    anch_shape = None if anchors is None else getattr(anchors, "shape", None)
    print(f"[startup] OK | rows={len(df)} | features={len(feature_cols)} | anchors={anch_shape} |")

async def get_optional_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("auth_token") or request.cookies.get("ACCESS_TOKEN")
    if not token:
        return None
    try:
        subject = CRUD.decode_subject_from_token(token)
        user_id = CRUD.current_user_id(subject)
        return db.query(models.User).filter(models.User.id == user_id).first()
    except HTTPException:
        return None

@app.post("/auth/api-keys", response_model=schema.ApiKeyResponse, tags=["Auth"], include_in_schema=False)
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


@app.post("/auth/token", response_model=schema.TokenOut, tags=["Auth"], include_in_schema=False)
def issue_token(API_key_in: Optional[str] = Security(api_key_header), db: Session = Depends(get_db)):
    row = CRUD.verify_api_key(db, API_key_in)
    token, exp_ts = CRUD.create_access_token(subject=f"user:{row.user_id}")
    return schema.TokenOut(access_token=token, expires_at=exp_ts)

@app.post("/predict", tags=["predict"], dependencies=[Depends(CRUD.get_current_subject)],summary="Obtenir des recommandations",
    description=("Calcule un top-k d’établissements en fonction du formulaire utilisateur.\n\n"
                 "- `use_ml=true` active le modèle ML si disponible, sinon fallback proxy.\n"
                 "- `k` borné à [1, 50]."),
    response_model=schema.PredictionResponse,  
    response_model_exclude_none=True,
    responses={400: {"description": "Entrée invalide"},
               401: {"description": "Non autorisé (Bearer manquant/erroné)"},
               500: {"description": "Erreur serveur"},},
    openapi_extra={"requestBody": {"content": {"application/json": {"examples": {"simple": {"summary": "Recherche simple",
                            "value": {"description": "italien terrasse","price_level": 2,"city": "Tours","open": "soir_weekend",
                                "options": ["reservable","outdoorSeating"]}}}}}}})
def predict(form: schema.Form,k: int = 10,use_ml: bool = True,user_id: int = Depends(CRUD.current_user_id),db: Session = Depends(get_db),):
    try :   
        t0 = time.perf_counter()

        if (not hasattr(app.state, "DF_CATALOG")) or app.state.DF_CATALOG is None or app.state.DF_CATALOG.empty:
            raise HTTPException(500, "Catalogue vide/non chargé.")
        df = app.state.DF_CATALOG

        try:
            form_row = models.FormDB(
                price_level=getattr(form, "price_level", None),
                city=getattr(form, "city", None),
                open=getattr(form, "open", None),
                options=getattr(form, "options", None),
                description=getattr(form, "description", None),
            )
            db.add(form_row)
            db.flush()
            form_id = form_row.id
        except Exception as e:
            db.rollback()
            raise HTTPException(500, f"Insertion du formulaire impossible: {e}")

        anchors = getattr(app.state, "ANCHORS", None)
        X_df, gains_proxy = build_item_features_df(df=df,form=form.model_dump(),sent_model=app.state.SENT_MODEL,
            include_query_consts=True,anchors=anchors)

        used_ml = False
        scores = np.asarray(gains_proxy, dtype=float)  

        model  = getattr(app.state, "ML_MODEL", None)
        preproc = getattr(app.state, "PREPROC", None)
        X_items = getattr(app.state, "X_ITEMS", None)

        raw = X_df.drop(columns=["id_etab"], errors="ignore")
        raw_nb = CRUD._numeric_bool_to_float(raw)

        if use_ml and model is not None:
            try:
                if CRUD._is_sklearn_pipeline(model):
                    X_in = raw_nb

                    n_exp = getattr(model, "n_features_in_", None)
                    if n_exp is not None and X_in.shape[1] != n_exp:
                        if X_in.shape[1] > n_exp:
                            X_in = X_in.iloc[:, :n_exp]
                        else:
                            pad = np.zeros((X_in.shape[0], n_exp - X_in.shape[1]), dtype=float)
                            X_in = np.hstack([X_in.to_numpy(dtype=float), pad])

                    scores = utils._predict_scores(model, X_in)
                    used_ml = True

                else:
                    feat_cols_train = getattr(app.state, "FEATURE_COLS_TRAIN", None) \
                                    or getattr(app.state, "FEATURE_COLS", None)
                    X_align = utils._align_df_to_cols(raw_nb, feat_cols_train) if feat_cols_train else raw_nb

                    if preproc is not None:
                        X_sp = preproc.transform(X_align)
                        X_in = X_sp.toarray().astype(np.float32) if hasattr(X_sp, "toarray") else np.asarray(X_sp, dtype=np.float32)
                        scores = utils._predict_scores(model, X_in)
                        used_ml = True
                    else:
                        used_ml = False

            except Exception as e:
                print(f"[predict] chemin ML en échec, fallback proxy: {e}")
                used_ml = False


        k = int(max(1, min(k or 10, 50)))
        order = np.argsort(scores)[::-1]
        sel = order[:k]

        latency_ms = int((time.perf_counter() - t0) * 1000)
        model_version = os.getenv("MODEL_VERSION") or getattr(app.state, "MODEL_VERSION", None) or "dev"

        pred_row = models.Prediction(form_id=form_id,k=k,model_version=model_version,latency_ms=latency_ms,status="ok",)
        if hasattr(models.Prediction, "user_id"):
            setattr(pred_row, "user_id", user_id)

        candidate_ids = [
            int(df.iloc[i]["id_etab"]) if "id_etab" in df.columns else int(i)
            for i in sel
        ]

        rows = db.query(models.Etablissement.id_etab)\
                .filter(models.Etablissement.id_etab.in_(candidate_ids))\
                .all()
        existing_ids = {int(x[0]) for x in rows}

        pred_row.items = []
        rank = 1
        for i in sel:
            etab_id = int(df.iloc[i]["id_etab"]) if "id_etab" in df.columns else int(i)
            if etab_id not in existing_ids:
                print(f"[predict] skip etab_id={etab_id} (absent en DB)")
                continue
            pred_row.items.append(
                models.PredictionItem(rank=rank, etab_id=etab_id, score=float(scores[i]))
            )
            rank += 1

        try:
            db.add(pred_row)
            db.commit()
            db.refresh(pred_row)
        except Exception as e:
            db.rollback()
            raise HTTPException(500, f"Insertion de la prédiction impossible: {e}")

        try:
            pyd_pred = schema.Prediction.model_validate(pred_row)
            CRUD.log_prediction_event(
                prediction=pyd_pred,
                form_dict=form.model_dump(),
                scores=np.asarray(scores, dtype=float),
                used_ml=used_ml,
                latency_ms=latency_ms,
                model_version=model_version,
            )
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
    except Exception as e:
        print("\n---! ERREUR DANS L'ENDPOINT /predict !---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("-----------------------------------------\n", file=sys.stderr)
        
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@app.post("/feedback", response_model=schema.FeedbackOut, tags=["monitoring"],summary="Envoyer un feedback utilisateur",
    description="Attache une note/commentaire à une prédiction existante.",
    responses={401: {"description": "Non autorisé"},
               403: {"description": "La prédiction n’appartient pas à l’utilisateur courant"},
               404: {"description": "Prédiction introuvable"}})
def submit_feedback(payload: schema.FeedbackIn,sub: str = Depends(CRUD.get_current_subject),
                    db: Session = Depends(get_db),user_id: int = Depends(CRUD.current_user_id)):
    pred = db.query(models.Prediction).options(selectinload(models.Prediction.items)).filter(models.Prediction.id == payload.prediction_id).first()
    if not pred:
        raise HTTPException(404, "Prediction introuvable")
    
    if getattr(pred, "user_id", None) not in (None, user_id):
        raise HTTPException(status_code=403, detail="Cette prédiction n'appartient pas à l'utilisateur courant")
    
    existing = db.query(models.Feedback).filter(models.Feedback.prediction_id == payload.prediction_id).first()
    if existing:
        return schema.FeedbackOut(status="Feedback déjà existant pour cette prédiction")

    row = models.Feedback(prediction_id=pred.id,rating=payload.rating,comment=payload.comment)
    db.add(row); db.commit()

    try:
        CRUD.log_feedback_rating(prediction_id=str(pred.id),rating=payload.rating,k=pred.k,model_version=pred.model_version,
        user_id=user_id,comment=payload.comment,use_active_run_if_any=True)
    except Exception as e:
        print(f"[mlflow] log_feedback_rating failed: {e}")

    return schema.FeedbackOut()

@limiter.limit("15/minute; 2/10second")
@app.post("/auth/web/token", response_model=schema.TokenOut, tags=["Auth"],openapi_extra={"security": []})
async def login_for_web_access_token(request: Request,db: Session = Depends(get_db),form_data: OAuth2PasswordRequestForm = Depends(),
                                     cf_token: str = Form(alias="cf-turnstile-response", default="")):

    remote_ip = request.client.host if request.client else None

    ok, details = await verify_turnstile(cf_token, remote_ip)
    if not ok and BYPASS==0:
        raise HTTPException(status_code=400, detail="CAPTCHA failed")

    user = CRUD.get_user_by_username(db, username=form_data.username)
    if not user or not user.hashed_password or not CRUD.ph.verify(user.hashed_password, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d'utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token, exp_ts = CRUD.create_access_token(subject=f"user:{user.id}")
    return schema.TokenOut(access_token=token, expires_at=exp_ts)


@limiter.limit("15/minute; 2/10second")
@app.post("/users", response_model=schema.UserOut, tags=["Auth"], openapi_extra={"security": []})
async def register_user(request: Request,db: Session = Depends(get_db),form: Optional[schema.RegisterIn] = Body(None)):
    if form is not None:
        data = form.model_dump(exclude_unset=True, by_alias=False)
        cf_token = form.cf_turnstile_response or form.captcha_token
    else:
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(400, "Corps JSON invalide ou manquant.")
        cf_token = (data.pop("cf-turnstile-response", None)
                    or data.pop("captcha_token", None))

    dev_bypass = int(os.getenv("TURNSTILE_DEV_BYPASS", "0"))
    if dev_bypass == 0:
        if not cf_token:
            raise HTTPException(400, "Vérification anti-robot manquante.")
        ok, _ = await verify_turnstile(cf_token, _client_ip(request))
        if not ok:
            raise HTTPException(400, "CAPTCHA failed")
    try:
        user_in = schema.UserCreate(**data)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    if CRUD.get_user_by_username(db, username=user_in.username):
        raise HTTPException(409, "Ce nom d'utilisateur est déjà pris.")
    existing = CRUD.get_user_by_email(db, email=user_in.email)
    if existing and existing.hashed_password:
        raise HTTPException(409, "Cet email est déjà utilisé.")
    return CRUD.create_user(db=db, user=user_in)

@app.get("/login", name="login_page", response_class=HTMLResponse, include_in_schema=False)
def ui_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "ACTIVE": "login", "TURNSTILE_SITEKEY": os.getenv("TURNSTILE_SITEKEY","")})

@app.get("/register", name="register_page", response_class=HTMLResponse, include_in_schema=False)
def ui_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "ACTIVE": "register", "TURNSTILE_SITEKEY": os.getenv("TURNSTILE_SITEKEY","")})

@app.get("/logout", include_in_schema=False)
def logout_get(request: Request):
    target = request.url_for("home") if hasattr(request.app.router, "url_path_for") else "/"
    resp = RedirectResponse(target, status_code=303)
    resp.delete_cookie("ACCESS_TOKEN") 
    return resp

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request,current_user: Optional[models.User] = Depends(get_optional_current_user),):
    if not current_user:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("home.html",{"request": request,"ACTIVE": "home","user": current_user,},)

@app.get("/predict", response_class=HTMLResponse, name="ui_predict", include_in_schema=False)
def ui_predict(request: Request):
    token = request.cookies.get("ACCESS_TOKEN")
    if not token:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/history", response_class=HTMLResponse, name="history", include_in_schema=False)
def history_page(request: Request):
    token = request.cookies.get("ACCESS_TOKEN")
    if not token:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("history.html", {"request": request, "ACTIVE": "history"})

@app.get("/data", response_class=HTMLResponse, name="data", tags=["ui"], include_in_schema=False)
def data_page(request: Request):
    token = request.cookies.get("ACCESS_TOKEN") or request.cookies.get("auth_token")
    if not token:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("data.html", {"request": request, "ACTIVE": "data"})

@app.get("/restaurant/{etab_id}", tags=["ui"], include_in_schema=False)
def restaurant_detail(etab_id: int, db: Session = Depends(get_db)):
    etab = (
        db.query(models.Etablissement)
        .filter(models.Etablissement.id_etab == etab_id)
        .first()
    )
    if not etab:
        raise HTTPException(status_code=404, detail="Restaurant introuvable")

    opt = (
        db.query(models.Options)
        .filter(models.Options.id_etab == etab_id)
        .first()
    )
    options = {
        "allowsDogs": bool(opt.allowsDogs) if opt and opt.allowsDogs is not None else False,
        "delivery": bool(opt.delivery) if opt and opt.delivery is not None else False,
        "goodForChildren": bool(opt.goodForChildren) if opt and opt.goodForChildren is not None else False,
        "goodForGroups": bool(opt.goodForGroups) if opt and opt.goodForGroups is not None else False,
        "goodForWatchingSports": bool(opt.goodForWatchingSports) if opt and opt.goodForWatchingSports is not None else False,
        "outdoorSeating": bool(opt.outdoorSeating) if opt and opt.outdoorSeating is not None else False,
        "reservable": bool(opt.reservable) if opt and opt.reservable is not None else False,
        "restroom": bool(opt.restroom) if opt and opt.restroom is not None else False,
        "servesVegetarianFood": bool(opt.servesVegetarianFood) if opt and opt.servesVegetarianFood is not None else False,
        "servesBrunch": bool(opt.servesBrunch) if opt and opt.servesBrunch is not None else False,
        "servesBreakfast": bool(opt.servesBreakfast) if opt and opt.servesBreakfast is not None else False,
        "servesDinner": bool(opt.servesDinner) if opt and opt.servesDinner is not None else False,
        "servesLunch": bool(opt.servesLunch) if opt and opt.servesLunch is not None else False,
    }

    periods = (
        db.query(models.OpeningPeriod)
        .filter(models.OpeningPeriod.id_etab == etab_id)
        .order_by(models.OpeningPeriod.open_day, models.OpeningPeriod.open_hour, models.OpeningPeriod.open_minute)
        .all()
    )
    heures_formatees = utils._format_opening_periods(periods)

    return {
        "id": etab.id_etab,
        "nom": etab.nom,
        "adresse": etab.adresse,
        "telephone": etab.internationalPhoneNumber,
        "site_web": etab.websiteUri,
        "rating": float(etab.rating) if etab.rating is not None else None,
        "price_level": utils._pricelevel_to_int(etab.priceLevel),
        "description": etab.description,
        "options": options,
        "horaires": heures_formatees,
        "latitude": etab.latitude,
        "longitude": etab.longitude,
    }

@app.get("/restaurant/{etab_id}/reviews", tags=["ui"], include_in_schema=False)
def restaurant_reviews(etab_id: int, db: Session = Depends(get_db)):
    rows = (
        db.query(models.Review)
        .filter(models.Review.id_etab == etab_id)
        .order_by(models.Review.publishTime.desc())
        .all()
    )
    return [{"date": r.publishTime,"rating": float(r.rating) if r.rating is not None else None,
            "comment": r.original_text,"author": r.author,}for r in rows]

@app.get("/history/predictions", tags=["ui"],include_in_schema=False)
def history_predictions(skip: int = 0,limit: int = 30,db: Session = Depends(get_db),user_id: int = Depends(CRUD.current_user_id)):
    rows = (db.query(models.Prediction).options(
                selectinload(models.Prediction.items),
                selectinload(models.Prediction.form),)
        .filter(models.Prediction.user_id == user_id)
        .order_by(models.Prediction.created_at.desc())  
        .offset(skip)
        .limit(limit)
        .all())

    def summarize_form(form):
        if not form:
            return {}
        return {"created_at": form.created_at.isoformat() if getattr(form, "created_at", None) else None,
            "price_level": form.price_level,
            "city": form.city,
            "open": form.open,
            "options": form.options,
            "description": form.description,
        }

    out = []
    for p in rows:
        fb = db.query(models.Feedback).filter(models.Feedback.prediction_id == p.id).first()
        out.append({
            "id": str(p.id),
            "k": p.k,
            "model_version": p.model_version,
            "latency_ms": p.latency_ms,
            "items_count": len(p.items),
            "created_at": p.created_at.isoformat() if getattr(p, "created_at", None) else None, 
            "form": summarize_form(p.form),
            "feedback": {
                "rating": fb.rating if fb else None,
                "comment": fb.comment if fb else None,
            },
        })
    return out

@app.get("/history/prediction/{pred_id}", tags=["ui"], include_in_schema=False)
def get_prediction_detail(pred_id: str,db: Session = Depends(get_db),user_id: int = Depends(CRUD.current_user_id),):
    try:
        pred_uuid = uuid.UUID(pred_id)
    except Exception:
        raise HTTPException(400, "pred_id invalide")

    pred = (db.query(models.Prediction)
        .options(
            selectinload(models.Prediction.items),
            selectinload(models.Prediction.form),)
        .filter(models.Prediction.id == pred_uuid,
            models.Prediction.user_id == user_id).first())
    if not pred:
        raise HTTPException(404, "Prédiction introuvable")

    fb = (
        db.query(models.Feedback)
        .filter(models.Feedback.prediction_id == pred_uuid)
        .first()
    )

    form = pred.form
    form_json = {
        "created_at": form.created_at.isoformat() if getattr(form, "created_at", None) else None,
        "price_level": form.price_level if form else None,
        "city": form.city if form else None,
        "open": form.open if form else None,
        "options": form.options if form else None,
        "description": form.description if form else None,
    }

    etab_ids = [itm.etab_id for itm in pred.items]
    name_map = {}
    if etab_ids:
        rows = (
            db.query(models.Etablissement.id_etab, models.Etablissement.nom)
            .filter(models.Etablissement.id_etab.in_(etab_ids))
            .all())

        name_map = {row[0]: row[1] for row in rows}

    items_out = [
        {"rank": itm.rank,
        "etab_id": itm.etab_id,
        "score": float(itm.score),
        "name": name_map.get(itm.etab_id),}
        for itm in sorted(pred.items, key=lambda i: i.rank)]

    return {
        "id": str(pred.id),
        "created_at": pred.created_at.isoformat() if getattr(pred, "created_at", None) else None,
        "k": pred.k,
        "model_version": pred.model_version,
        "latency_ms": pred.latency_ms,
        "form": form_json,
        "feedback": {
            "rating": fb.rating if fb else None,
            "comment": fb.comment if fb else None},"items": items_out,}


@app.get("/ui/api/restaurants", tags=["ui"])
def list_restaurants(q: Optional[str] = None,city: Optional[str] = None,price_level: Optional[int] = Query(None, ge=1, le=4),open_day: Optional[str] = None,
    options: Optional[List[str]] = Query(None),sort_by: Optional[str] = None,sort_dir: Optional[str] = "asc",skip: int = 0,limit: int = 50,
    db: Session = Depends(get_db)):

    q0 = db.query(models.Etablissement)

    if options and len(options) == 1 and isinstance(options[0], str) and "," in options[0]:
        options = [s for s in options[0].split(",") if s]

    joined_opts = False
    if options:
        q0 = q0.join(models.Options)
        joined_opts = True
    if open_day:
        q0 = q0.join(models.OpeningPeriod)

    if q:
        q0 = q0.filter(models.Etablissement.nom.ilike(f"%{q.lower()}%"))
    if city:
        q0 = q0.filter(models.Etablissement.adresse.ilike(f"%{city}%"))
    if price_level:
        q0 = q0.filter(models.Etablissement.priceLevel == str(price_level))
    if options:
        for opt in options:
            if hasattr(models.Options, opt):
                q0 = q0.filter(getattr(models.Options, opt) == True)
    if open_day:
        jours = {
            "lundi": 1, "mardi": 2, "mercredi": 3, "jeudi": 4,
            "vendredi": 5, "samedi": 6, "dimanche": 0,
            "monday": 1, "tuesday": 2, "wednesday": 3,
            "thursday": 4, "friday": 5, "saturday": 6, "sunday": 0,
        }
        d = jours.get(open_day.lower())
        if d is not None:
            q0 = q0.filter(models.OpeningPeriod.open_day == d)

    ids_subq = q0.with_entities(models.Etablissement.id_etab).distinct().subquery()
    total = db.query(ids_subq).count()

    rows_q = (
        db.query(models.Etablissement)
        .join(ids_subq, ids_subq.c.id_etab == models.Etablissement.id_etab)
        .options(
            selectinload(models.Etablissement.options),
            selectinload(models.Etablissement.opening_periods),
        )
    )

    dir_desc = (str(sort_dir).lower() == "desc")
    if sort_by in ("nom", "rating", "price", "price_level"):
        if sort_by == "nom":
            col = models.Etablissement.nom
        elif sort_by == "rating":
            col = models.Etablissement.rating
        else:
            col = cast(models.Etablissement.priceLevel, Integer)
        try:
            order_expr = (col.desc() if dir_desc else col.asc()).nulls_last()
        except Exception:
            order_expr = col.desc() if dir_desc else col.asc()
        rows_q = rows_q.order_by(order_expr, models.Etablissement.id_etab.asc())
    else:
        rows_q = rows_q.order_by(models.Etablissement.id_etab.asc())

    rows = rows_q.offset(skip).limit(limit).all()

    opt_fields = [
        "allowsDogs","delivery","goodForChildren","goodForGroups",
        "goodForWatchingSports","outdoorSeating","reservable","restroom",
        "servesVegetarianFood","servesBrunch","servesBreakfast",
        "servesDinner","servesLunch",
    ]

    def build_options_map(opt_row) -> dict:
        out = {}
        if not opt_row:
            for f in opt_fields: out[f] = False
            return out
        for f in opt_fields:
            val = getattr(opt_row, f, None)
            out[f] = bool(val) if val is not None else False
        return out

    def options_true_list(opt_row) -> list[str]:
        if not opt_row: return []
        return [f for f in opt_fields if getattr(opt_row, f, None)]

    results = []
    for e in rows:
        try:
            price_lvl_int = utils._pricelevel_to_int(e.priceLevel)
        except Exception:
            price_lvl_int = None

        horaires_fmt = utils._format_opening_periods(e.opening_periods or [])
        opts_map = build_options_map(e.options)

        results.append({
            "id": e.id_etab,
            "nom": e.nom,
            "adresse": e.adresse,
            "telephone": e.internationalPhoneNumber,
            "site_web": e.websiteUri,
            "description": e.description or e.editorialSummary_text,
            "rating": float(e.rating) if e.rating is not None else None,
            "price_level": price_lvl_int,
            "start_price": e.start_price,
            "end_price": e.end_price,
            "latitude": e.latitude,
            "longitude": e.longitude,
            "options": opts_map,                 
            "options_actives": options_true_list(e.options),  
            "horaires": horaires_fmt,
        })

    return {"total": total, "items": results}
