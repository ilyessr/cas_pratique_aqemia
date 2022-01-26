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
        """
    <html>
        <head>
            <style>
                p{font-family: "Gill Sans", sans-serif;}
            </style>
            <title> Age abalone predictor </title>

        </head>
        <body >

            <h1>Welcome</h1>
            <p>
            The goal of this REST API is to serve predictions about the age 
            of the abalones from an existing machine learning model. </p>
            <p>Click on the button to test it.</p>

            <button onclick="window.location.href = 
            '/docs#/default/make_prediction_abalones_post';">
                Click here
            </button>
        </body>
    </html>
        """

    )
