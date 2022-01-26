from typing import List

from abalone_prediction import modeling as md
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


class AbaloneIn(BaseModel):
    sex: str
    length: float
    diameter: float
    height: float
    whole_weight: float
    shucked_weight: float
    viscera_weight: float
    shell_weight: float


class UserRequestInput(BaseModel):
    inputs: List[AbaloneIn]


class AbaloneOut(BaseModel):
    label: int
    probability: float


class OutputPrediction(BaseModel):
    outputs: List[AbaloneOut]


app = FastAPI()

model = md.load_model("data/pickles/model_RF_abalone_v1.pkl")


@app.post("/abalones", response_model=OutputPrediction)
def make_prediction(user_request: UserRequestInput):
    inputs_client = user_request.inputs  # List of AbaloneIn object
    abalones = [md.predict_abalone(i.dict(), model) for i in inputs_client]
    return {"outputs": abalones}


@app.get("/", response_class=HTMLResponse)
def index():
    return (
        "<p>Welcome to the REST API used for \
            age prediction of abalone :)</p>"

    )
