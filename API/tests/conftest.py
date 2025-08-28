"""
Fixtures communes pour les tests d'intégration.

- Lance un Postgres éphémère via testcontainers.
- Configure les variables d'env consommées par API/database.py.
- Initialise les schémas/tables SQLAlchemy.
- Expose un TestClient FastAPI déjà prêt.
"""

import os
import importlib
import contextlib
import uuid
import pytest

# testcontainers[postgres] est listé dans requirements_dev.txt
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def pg_container():
    # image légère et récente
    with PostgresContainer("postgres:16-alpine") as pg:
        pg.start()
        yield pg


def _export_pg_env(pg: PostgresContainer):
    host = pg.get_container_host_ip()
    port = pg.get_exposed_port(5432)
    os.environ["POSTGRES_USER"] = pg.username
    os.environ["POSTGRES_PASSWORD"] = pg.password
    os.environ["POSTGRES_DB"] = pg.dbname
    os.environ["POSTGRES_HOST"] = host
    os.environ["POSTGRES_PORT"] = str(port)


@pytest.fixture(scope="session")
def app_and_db(pg_container):
    # Empêche le warmup ML/DF au démarrage de l'app
    os.environ["DISABLE_WARMUP"] = "1"
    # Secret & clé statique par défaut (comme dans ci.yml)
    os.environ.setdefault("API_STATIC_KEY", "coall")
    os.environ.setdefault("JWT_SECRET", "coall")

    # Configure la BDD pour ce process AVANT tout import de l'app
    _export_pg_env(pg_container)

    # (Ré)importe les modules dépendants de la config BDD
    database = importlib.import_module("API.database")
    models = importlib.import_module("API.models")

    # Crée les schémas et tables
    models.ensure_ml_schema(database.engine)  # crée "ml" & "user_base"
    models.Base.metadata.create_all(database.engine)  # crée aussi les tables "public"

    # Charge l'app après l'init DB
    main = importlib.import_module("API.main")
    app = main.app

    return app, database, models


@pytest.fixture
def db_session(app_and_db):
    _, database, _ = app_and_db
    session = database.SessionLocal()
    try:
        yield session
    finally:
        with contextlib.suppress(Exception):
            session.rollback()
        session.close()
