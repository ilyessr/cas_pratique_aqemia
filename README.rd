# Prediction API

This a REST API for serving predictions of abalone age based on an existing
machine learning model.

- './run_app.sh' : Run the application locally.
- './run_docker.sh' : Run the docker locally.
- './run_docker.sh' : Have some documentation about the app


The api has been deployed in EC2 : http://34.193.161.120:8000/docs#/default/make_prediction_abalones_post

Github action has been used to create a deployment workflow.
Link of the github : https://github.com/ilyessr/cas_pratique_aqemia



# Organisation of the directory

- abalone_app :

        --- abalone_prediction :
            --- main.py : API (FastAPI + Pydantic)
            --- modeling.py : Python code based on the jupyter notebook

        -- data :
            --- pickles : Contains the model.
            --- test_data : Contain expected data for units test + script to generate them

        -- docs : Generated from docstring with Sphinx

        --tests : Units tests

 - Dockerfile
 - docker-compose.yaml
 - run_app.sh
 - run_docker.sh


