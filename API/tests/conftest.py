"""
Fixtures communes d'intégration:
- Postgres éphémère via testcontainers
- Variables d'env pour API/database.py
- Création des schémas/tables
- Fixtures app/client/db_session
"""

import os
import importlib
import contextlib
import pytest
from testcontainers.postgres import PostgresContainer
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def _db_env():
    # Démarre un Postgres jetable (Docker requis sur le runner)
    with PostgresContainer("postgres:16-alpine") as pg:
        pg.start()
        os.environ["POSTGRES_USER"] = pg.username
        os.environ["POSTGRES_PASSWORD"] = pg.password
        os.environ["POSTGRES_DB"] = pg.dbname
        os.environ["POSTGRES_HOST"] = pg.get_container_host_ip()
        os.environ["POSTGRES_PORT"] = pg.get_exposed_port(5432)

        # autres env attendues par l'appli/tests
        os.environ.setdefault("DISABLE_WARMUP", "1")
        os.environ.setdefault("API_STATIC_KEY", "coall")
        os.environ.setdefault("JWT_SECRET", "coall")
        yield


@pytest.fixture(scope="session")
def app(_db_env):
    # Importe DB + modèles une fois l'env posée
    database = importlib.import_module("API.database")
    models = importlib.import_module("API.models")

    # Crée les schémas + tables
    models.ensure_ml_schema(database.engine)
    models.Base.metadata.create_all(database.engine)

    # Charge l'app seulement après la BDD prête
    main = importlib.import_module("API.main")
    return main.app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def db_session(_db_env):
    database = importlib.import_module("API.database")
    s = database.SessionLocal()
    try:
        yield s
    finally:
        with contextlib.suppress(Exception):
            s.rollback()
        s.close()
