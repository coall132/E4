"""
Fixtures d'intégration :
- Postgres éphémère via testcontainers
- Override explicite de API.database.engine / SessionLocal
- Création des schémas/tables
- Fixtures app / client / db_session
"""

import os
import contextlib
import importlib
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from testcontainers.postgres import PostgresContainer
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def pg():
    """Démarre un Postgres jetable une seule fois pour toute la session."""
    container = PostgresContainer("postgres:16-alpine")
    container.start()
    try:
        yield container
    finally:
        # Arrêt propre à la fin de la session
        with contextlib.suppress(Exception):
            container.stop()


def _engine_url_from_pg(pg: PostgresContainer) -> str:
    """
    Construit une URL SQLAlchemy pour psycopg2 à partir du container.
    testcontainers retourne typiquement 'postgresql://...'; on force
    'postgresql+psycopg2://' pour SQLAlchemy.
    """
    url = pg.get_connection_url()  # ex: postgresql://test:test@127.0.0.1:32777/test
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


@pytest.fixture(scope="session")
def app(pg):
    """
    Configure l'env et remplace l'engine AVANT d'importer l'app.
    """
    # Exporte aussi les env au cas où du code s'y réfère
    os.environ["POSTGRES_USER"] = pg.username
    os.environ["POSTGRES_PASSWORD"] = pg.password
    os.environ["POSTGRES_DB"] = pg.dbname
    os.environ["POSTGRES_HOST"] = pg.get_container_host_ip()
    os.environ["POSTGRES_PORT"] = pg.get_exposed_port(5432)
    os.environ.setdefault("DISABLE_WARMUP", "1")
    os.environ.setdefault("API_STATIC_KEY", "coall")
    os.environ.setdefault("JWT_SECRET", "coall")

    engine_url = _engine_url_from_pg(pg)
    # Pour tout code qui lirait une URL unique :
    os.environ["DATABASE_URL"] = engine_url
    os.environ["SQLALCHEMY_DATABASE_URL"] = engine_url

    # Importe le module DB et **override** l'engine + SessionLocal
    database = importlib.import_module("API.database")

    with contextlib.suppress(Exception):
        database.engine.dispose()

    database.engine = create_engine(engine_url, future=True)
    database.SessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=database.engine
    )

    # Crée schémas + tables avant de charger l'app
    models = importlib.import_module("API.models")
    models.ensure_ml_schema(database.engine)
    models.Base.metadata.create_all(database.engine)

    # Charge l'app seulement maintenant
    main = importlib.import_module("API.main")
    return main.app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def db_session(app):
    """Session courte pour chaque test, rollback à la fin."""
    database = importlib.import_module("API.database")
    s = database.SessionLocal()
    try:
        yield s
    finally:
        with contextlib.suppress(Exception):
            s.rollback()
        s.close()
