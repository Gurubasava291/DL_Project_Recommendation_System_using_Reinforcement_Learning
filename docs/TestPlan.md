# Test Plan
## Full-Stack MLOps Recommender System

### 1. Introduction
This document describes the testing strategy for the MLOps Recommender System, covering unit testing for models, integration testing for the API, and validation for the MLOps pipeline.

### 2. Test Scope
- **Unit Tests**: Verification of PyTorch and PennyLane model components.
- **Integration Tests**: Verification of the FastAPI backend endpoints (`/recommend`, `/metrics`).
- **MLOps Validation**: Ensuring MLflow tracks experiments correctly and Docker containers build without errors.

### 3. Test Strategy
#### 3.1 Unit Testing (Models)
- **Framework**: `pytest`
- **DL Tier (PyTorch)**: Ensure that the DQN agent outputs the correct Q-value tensor shapes given a batch of state inputs. Ensure the loss function converges on a dummy dataset.
- **QML Tier (PennyLane)**: Verify that the quantum circuit produces valid expectation values (between -1 and 1) and that gradients can be computed.

#### 3.2 Integration Testing (API)
- **Framework**: `pytest` with `fastapi.testclient.TestClient`
- **Endpoints to Test**:
  - `GET /`: Should return the static HTML frontend.
  - `POST /recommend`: Should accept a state vector and return a list of recommended song IDs.
  - `GET /metrics`: Should return tracking metrics from the latest MLflow run.

#### 3.3 MLOps Pipeline Testing
- **Docker**: Run `docker build -t mlops-recommender .` to ensure all dependencies resolve and the image builds successfully.
- **MLflow tracking**: Run the training pipeline on a small subset of data and assert that an MLflow run is created with the required parameters, metrics, and models.

### 4. Test Environment
- Python 3.9+
- Docker
- Pytest

### 5. Execution
Run tests locally via:
```bash
pytest tests/ -v
```
