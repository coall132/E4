from __future__ import annotations
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: int

class ApiKeyCreate(BaseModel):
    email: str
    username: str = Field(min_length=3, max_length=50)

class ApiKeyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    api_key: str                 
    key_id: str              


class Form(BaseModel):
    price_level:Optional[int] = None
    city:Optional[str] = None
    open:Optional[str] = None
    options:Optional[list] = None
    description:Optional[str] = None
    created_at: Optional[datetime] = Field(default=None, json_schema_extra={"readOnly": True})

class Geo(BaseModel):
    lat: Optional[float] = None
    lng: Optional[float] = None

class EtablissementDetails(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id_etab: int
    nom: Optional[str] = None
    adresse: Optional[str] = None
    telephone: Optional[str] = None
    site_web: Optional[str] = None
    description: Optional[str] = None
    rating: Optional[float] = None
    priceLevel: Optional[int] = None
    priceLevel_symbole: Optional[str] = None
    startPrice: Optional[float] = None
    endPrice: Optional[float] = None
    geo: Optional[Geo] = None
    options_actives: List[str] = []
    horaires: List[str] = []

class PredictionItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    rank: int
    etab_id: int
    score: float

class PredictionItemRich(PredictionItem):
    details: Optional[EtablissementDetails] = None

class Prediction(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    form_id: UUID
    k: int
    model_version: Optional[str] = None
    latency_ms: Optional[int] = None
    status: Optional[str] = "ok"
    items: List[PredictionItem] = []

class PredictionResponse(Prediction):
    prediction_id: UUID
    items_rich: List[PredictionItemRich] = []
    message: Optional[str] = None

class FeedbackIn(BaseModel):
    prediction_id: UUID  
    rating: Optional[int] = None     
    comment: Optional[str] = None

class FeedbackOut(BaseModel):
    status: str = "ok"

class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str
    password: str

class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    email: str
    created_at: datetime

class OpeningPeriod(BaseModel):
    open_day: Optional[int]
    open_hour: Optional[int]
    open_minute: Optional[int]
    close_day: Optional[int]
    close_hour: Optional[int]
    close_minute: Optional[int]

    class Config: orm_mode = True

class Review(BaseModel):
    original_languageCode: Optional[str]
    original_text: Optional[str]
    publishTime: Optional[str]
    rating: Optional[float]
    relativePublishTimeDescription: Optional[str]
    author: Optional[str]

    class Config: orm_mode = True

class Options(BaseModel):
    allowsDogs: Optional[bool]
    delivery: Optional[bool]
    goodForChildren: Optional[bool]
    goodForGroups: Optional[bool]
    goodForWatchingSports: Optional[bool]
    outdoorSeating: Optional[bool]
    reservable: Optional[bool]
    restroom: Optional[bool]
    servesVegetarianFood: Optional[bool]
    servesBrunch: Optional[bool]
    servesBreakfast: Optional[bool]
    servesDinner: Optional[bool]
    servesLunch: Optional[bool]
    
    class Config: orm_mode = True

class EtablissementBase(BaseModel):
    id_etab: int
    nom: Optional[str]
    adresse: Optional[str]
    rating: Optional[float]

    class Config: orm_mode = True

class Etablissement(EtablissementBase):
    internationalPhoneNumber: Optional[str]
    description: Optional[str]
    websiteUri: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    priceLevel: Optional[str]
    start_price: Optional[float]
    end_price: Optional[float]
    editorialSummary_text: Optional[str]
    google_place_id: Optional[str]
    reviews: List[Review] = Field(default_factory=list)
    options: Optional[Options] = None
    opening_periods: List[OpeningPeriod] = Field(default_factory=list)

    class Config: orm_mode = True

class EtablissementWithOptions(EtablissementBase):
    internationalPhoneNumber: Optional[str]
    description: Optional[str]
    websiteUri: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    priceLevel: Optional[str]
    start_price: Optional[float]
    end_price: Optional[float]
    editorialSummary_text: Optional[str]
    google_place_id: Optional[str]
    options: Optional[Options] = None  # On inclut seulement les options

    class Config:
        orm_mode = True

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: int
    
@dataclass
class MLState:
    preproc: object | None = None          
    preproc_factory: object | None = None 
    sent_model: object | None = None
    rank_model: object | None = None
    rank_model_path: str | None = None