
from typing import Literal
from pydantic import BaseModel, Field

class PassengerInput(BaseModel):
    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: float = Field(ge=0, le=120)
    SibSp: int = Field(ge=0)
    Parch: int = Field(ge=0)
    Fare: float = Field(ge=0)
    Embarked: Literal["C", "Q", "S"]

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    survival_probability: float
    non_survival_probability: float
