# Dockerization

This document outlines the steps to dockerize the Predict Xplore application.

- Our main component is an API which allows users to upload data and test available machine learning models.
- Each model has its own loading code, configuration files (e.g., hyperparameters, dataset info), and weight files.
- The models are dynamically loaded and used as requested by the API.

The system will be containerized into a single Docker image, and the API will handle calls to individual models, loading them with their respective configuration files.

## Project Structure

The project will have the following directory layout. Note that this is a very basic directory structure. A model can include as many files as it needs, but the file and class that provides the loading and inference functionality should be specified in `/app/config.json`.

```
/app
  ├── api/                     # API code (Flask/FastAPI/Django, etc.)
  │   ├── main.py              # Main API logic
  │   └── model_loader.py      # Model loader utility functions
  │   └── ...                  # Additional files as required
  ├── models/
  │   ├── model1/
  │   │   ├── model1.py        # Model 1's Python code
  │   │   ├── config/          # Configuration files for Model 1
  │   │   │   ├── cfg.yaml
  │   │   │   ├── data.yaml
  │   │   ├── weights/         # Weights for Model 1
  │   │       └── best.pt
  │   ├── model2/              # Model 2 directory
  │       ├── model2.py        # Model 2's Python code
  │       ├── config/
  │       │   ├── cfg.json
  │       │   ├── dataset.json
  │       ├── weights/
  │           └── best.pt
  ├── config.json              # Config file for model associations
  ├── Dockerfile               # Dockerfile for containerization
  ├── requirements.txt         # Dependencies (API and model-related)
  └── ...                      # Additional files as required
```

## Dockerization Process

### API Setup

1. **Install the API framework** (Flask/FastAPI/Django). For example, in `requirements.txt`, include:

   ```py
   fastapi
   uvicorn
   torch               # PyTorch for model loading
   ```

2. **API Code**: 
   - Your API (e.g., in `api/main.py`) should include endpoints as defined in [the API documentation.](API.md)
   - Example structure:
    ```python
    from fastapi import FastAPI
    from model_loader import load_model

    app = FastAPI()

    @app.post(...)
    async def predict(...):
        model = load_model(model_name)
        result = model.predict(dataset)
        return result
    ```

### Model-Specific Code

Each model should have its own loading and prediction logic. It should *NOT* include any training code. For example:

- **Model 1 (`models/model1/model1.py`)**:
   ```python
   import torch
   import yaml

   class Model1:
       def __init__(self):
           # Load model weights and config files
           self.model = YOLOv8("/app/models/model1/config/data.yaml")
           self.model.load_state_dict(torch.load("/app/models/model1/weights/best.pt"))

       def predict(self, data):
           # Use self.model for inference after preprocessing (as required)
           return self.model.predict(data)
   ```

- **Model 2 (`models/model2/model2.py`)**:
   ```python
   import torch
   import json

   class Model2:
       def __init__(self):
           self.model = torch.load("/app/models/model2/weights/best.pt")
           with open("/app/models/model2/config/cfg.json", "r") as f:
               self.hyperparams = json.load(f)

       def predict(self, dataset):
           return self.model.infer(dataset)
   ```

The requirements of each model should be included in `/app/requirements.txt`. Their associations should be defined in `/app/config.json` as follows:

```json
{
  "model1": {
    "path": "models/model1/model1.py",
    "class": "Model1"
  },
  "model2": {
    "path": "models/model2/model2.py",
    "class": "Model2"
  }
}
```

### Model Loader Code

The `model_loader.py` file will dynamically load the correct model, getting associations between model names, their directory structures, and class names from `/app/config.json`:

```python
import importlib.util
import json
import os

CONFIG_PATH = "/app/config.json"

def load_model(model_name: str):
    with open(CONFIG_PATH, "r") as f:
        models_config = json.load(f)

    if model_name not in models_config:
        raise ValueError(f"Model {model_name} not found in configuration file.")

    model_info = models_config[model_name]
    model_path = model_info["path"]
    model_class = model_info["class"]

    # Dynamically load the model class
    spec = importlib.util.spec_from_file_location(model_class, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class from the module
    model_cls = getattr(module, model_class)

    # Return an instance of the model
    return model_cls()
```

## Dockerfile

The `Dockerfile` defines how to build and run the container:

```Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ /app/api/
COPY models/ /app/models/
COPY config.json .

# Expose the port (todo: adjust based on your API setup)
EXPOSE 8000

# Command to run the API (todo: adjust based on your API setup)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Deployment and Running the Docker Container

Use the following command to build the Docker image from the root of the project:

```bash
docker build -t predict-xplore .
```

Once the image is built, you can run the container with:

```bash
docker run -p 8000:8000 predict-xplore
```

This will start the API and expose it on port `8000`.

## Notes for the Developer

1. **Model-Specific Loading**: Each model's loading and inference logic should reside in its own Python file (e.g., `model1.py`, `model2.py`). The API will call the appropriate model by dynamically importing it using the `load_model` function.

2. **Config Files**: Each model can have one or more configuration files (e.g., for hyperparameters and dataset info). These should be placed in the `config` folder within each model's directory. The model's Python code should be responsible for loading and using these config files.

3. **Model Weights**: The weights of each model should be stored in the `weights` directory inside the model’s folder. These weights can be large, so some weights can be dynamically downloaded them from a remote storage service (like AWS S3) if needed.

4. **API Design**: The API will handle requests for different models via endpoints like `/test`. The user can upload datasets, and the API will load the correct model using the `model_loader.py` logic and return predictions.

5. **Scaling Considerations**: While this setup works well in a single container, if needed, you could move to a multi-container architecture later, where each model might be hosted in its own container. For now, this single-container design simplifies deployment.
