# models_ml.py
import uuid
from sqlalchemy import (
    Column, Integer, Text, ForeignKey, TIMESTAMP, func,
    UniqueConstraint, Index, Boolean,String, JSON, DateTime,Float
)
from sqlalchemy.dialects.postgresql import UUID, DOUBLE_PRECISION, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import text
from sqlalchemy import Column, Integer, TIMESTAMP, func, ForeignKey, Table

try:
    from .database import Base
except:
    from API.database import Base

def ensure_ml_schema(engine):
    with engine.connect() as conn:
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "ml"'))
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "user_base"'))
        conn.commit()

def _attach_external_tables(engine):
    # Reflète la table déjà existante en base dans la metadata des modèles
    Table("etab", Base.metadata, schema="public", autoload_with=engine)
    print("[startup] Attached external table public.etab to metadata")

class FormDB(Base):
    __tablename__ = "form"
    __table_args__ = {'schema': 'ml'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    price_level = Column(Integer)
    city = Column(Text)
    open = Column(Text, nullable=True)        
    options = Column(JSONB)             
    description = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    predictions = relationship("Prediction", back_populates="form")

class Prediction(Base):
    __tablename__ = "prediction"
    __table_args__ = {'schema': 'ml'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    form_id = Column(UUID(as_uuid=True), ForeignKey("ml.form.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("user_base.user.id", ondelete="RESTRICT"), nullable=False)
    k = Column(Integer, nullable=False)
    model_version = Column(Text, nullable=False)
    latency_ms = Column(Integer)
    status = Column(Text, default="ok")
    created_at = Column(TIMESTAMP(timezone=True),server_default=func.now(),default=func.now(),nullable=True)

    form = relationship("FormDB", back_populates="predictions")
    items = relationship("PredictionItem", back_populates="prediction", cascade="all, delete-orphan")

class PredictionItem(Base):                         
    __tablename__ = "prediction_item"
    __table_args__ = (
        UniqueConstraint("prediction_id", "rank", name="uq_prediction_item_rank"),
        Index("ix_prediction_item_prediction", "prediction_id"),
        Index("ix_prediction_item_etab", "etab_id"),
        {'schema': 'ml'},
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("ml.prediction.id", ondelete="CASCADE"), nullable=False)

    rank = Column(Integer, nullable=False)           
    etab_id = Column(Integer, ForeignKey("public.etab.id_etab"), nullable=False)  
    score = Column(DOUBLE_PRECISION, nullable=False)

    prediction = relationship("Prediction", back_populates="items")

class User(Base):
    __tablename__ = "user"
    __table_args__ = {'schema': 'user_base'}

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(80), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True) 
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")

class ApiKey(Base):
    __tablename__ = "api_key"
    __table_args__ = (
        UniqueConstraint("key_id", name="uq_api_key_keyid"),
        Index("ix_api_key_user", "user_id"),{'schema': 'user_base'}
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey("user_base.user.id", ondelete="CASCADE"), nullable=False)
    key_id = Column(String(32), nullable=False, index=True)
    key_hash = Column(Text, nullable=False)       
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    last_used_at = Column(TIMESTAMP(timezone=True), nullable=True)

    user = relationship("User", back_populates="api_keys")

class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = {'schema': 'ml'}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    prediction_id = Column(UUID(as_uuid=True),ForeignKey("ml.prediction.id", ondelete="CASCADE"),
                           nullable=False, index=True)
 
    rating = Column(Integer, nullable=True)           
    comment = Column(Text, nullable=True)

class Etablissement(Base):
    __tablename__ = "etab"
    __table_args__ = {'schema': 'public'}
    id_etab = Column(Integer, primary_key=True, index=True)
    nom = Column(String, index=True)
    internationalPhoneNumber = Column(String, nullable=True)
    adresse = Column(String, nullable=True)
    description = Column(String, nullable=True)
    websiteUri = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    rating = Column(Float, nullable=True)
    priceLevel = Column(String, nullable=True)
    start_price = Column(Float, nullable=True)
    end_price = Column(Float, nullable=True)
    editorialSummary_text = Column(String, nullable=True)
    google_place_id = Column(String, unique=True, index=True)

    reviews = relationship("Review", back_populates="etablissement")
    options = relationship("Options", back_populates="etablissement", uselist=False)
    opening_periods = relationship("OpeningPeriod", back_populates="etablissement")

class Options(Base):
    __tablename__ = "options"
    __table_args__ = {'schema': 'public'}
    id_etab = Column(Integer, ForeignKey("public.etab.id_etab"), primary_key=True)
    allowsDogs = Column(Boolean, nullable=True)
    delivery = Column(Boolean, nullable=True)
    goodForChildren = Column(Boolean, nullable=True)
    goodForGroups = Column(Boolean, nullable=True)
    goodForWatchingSports = Column(Boolean, nullable=True)
    outdoorSeating = Column(Boolean, nullable=True)
    reservable = Column(Boolean, nullable=True)
    restroom = Column(Boolean, nullable=True)
    servesVegetarianFood = Column(Boolean, nullable=True)
    servesBrunch = Column(Boolean, nullable=True)
    servesBreakfast = Column(Boolean, nullable=True)
    servesDinner = Column(Boolean, nullable=True)
    servesLunch = Column(Boolean, nullable=True)

    etablissement = relationship("Etablissement", back_populates="options")

class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = {'schema': 'public'}
    id_review = Column(Integer, primary_key=True, index=True)
    id_etab = Column(Integer, ForeignKey("public.etab.id_etab"))
    original_languageCode = Column(String, nullable=True)
    original_text = Column(String, nullable=True)
    publishTime = Column(String, nullable=True)
    rating = Column(Float, nullable=True)
    relativePublishTimeDescription = Column(String, nullable=True)
    author = Column(String, nullable=True)

    etablissement = relationship("Etablissement", back_populates="reviews")

class OpeningPeriod(Base):
    __tablename__ = "opening_period"
    __table_args__ = {'schema': 'public'}
    id_period = Column(Integer, primary_key=True, index=True)
    id_etab = Column(Integer, ForeignKey("public.etab.id_etab"))
    open_day = Column(Integer, nullable=True)
    open_hour = Column(Integer, nullable=True)
    open_minute = Column(Integer, nullable=True)
    close_day = Column(Integer, nullable=True)
    close_hour = Column(Integer, nullable=True)
    close_minute = Column(Integer, nullable=True)
    
    etablissement = relationship("Etablissement", back_populates="opening_periods")