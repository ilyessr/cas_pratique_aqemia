# Prediction API

The goal of this exercise is to serve predictions from an existing machine learning model through a REST API.

You are provided with a Jupyter notebook that trains and serializes a machine learning model. You are tasked with building an inference REST API that receives POST HTTP queries and returns predictions. Below is an example of input and output payloads.

### Input
```json
{    
  "inputs": [
    {
      "sex":"M",
      "length": 0.815000,
      "diameter": 0.055000,
      "height": 1.130000,
      "whole_weight": 2.825500,
      "shucked_weight": 1.488000,
      "viscera_weight": 0.760000,
      "shell_weight": 0.001500
    },
    {
      "sex":"F",
      "length": 0.815000,
      "diameter": 1.055000,
      "height": 1.130000,
      "whole_weight": 2.825500,
      "shucked_weight": 1.488000,
      "viscera_weight": 1.760000,
      "shell_weight": 0.001500
    }
  ]
}

```

### Output
```json
{
  "outputs": [
    {
      "label": 1,
      "probability": 0.109
    },
    {
      "label": 1,
      "probability": 0.183
    }
  ]
}
```

## Steps 

1. Choose a Python web app framework to build the API.
2. Implement the application. Make sure to follow conventional style guides for your Python code (e.g PEP8), to lint and format your code, and to include static types for your objects.
3. Provide a shell script to run the application locally. Querties can be tested using a visual REST client such as Postman or Insomnia. The `interface.py` script contains a helper function that can also be used to make test calls to the API.
4. Implement a Docker container to enable running the application anywhere and provide a shell script to run the container locally.
5. (bonus) Create an AWS account and deploy the API to AWS Elastic Beanstalk (Docker mode). You can keep the hardware specifications low enough to remain within the AWS Free Tier so that you do not incur any costs.