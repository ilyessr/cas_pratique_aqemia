#!/bin/sh

cd abalone_app
uvicorn abalone_prediction.main:app --reload
