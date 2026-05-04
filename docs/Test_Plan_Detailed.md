# Test Plan: Full-Stack MLOps Recommender System

## 1. Introduction
This document defines the comprehensive testing strategy for the Full-Stack MLOps Recommender System. It outlines the scope, approach, resources, and schedule of testing activities to ensure the reliability of the Machine Learning, Deep Learning, and Quantum Machine Learning models, alongside the backend infrastructure.

## 2. Test Items
The following components are subject to testing:
- **FastAPI Backend Services** (`app/main.py`)
- **Deep Q-Network Agent** (`src/dqn_agent.py`)
- **Quantum Variational Classifier** (`src/qml_classifier.py`)
- **MLflow Training Pipeline** (`train_pipeline.py`)
- **Docker Containerization Build** (`Dockerfile`)

## 3. Features to be Tested
- **Unit Logic**: Tensor shape consistency, loss function gradient propagation, and quantum circuit expectation values.
- **API Functionality**: HTTP request parsing, payload validation, and correct model routing for the `/recommend` and `/metrics` endpoints.
- **MLOps Integrity**: MLflow tracking metrics (loss, reward) during a dry-run of the training pipeline. Container build success without dependency conflicts.

## 4. Testing Approach
Testing is executed predominantly via the `pytest` framework, organized into unit and integration suites.

### 4.1 Unit Testing (Models)
- **Target**: `tests/test_models.py`
- **Criteria**: PyTorch DQN must output exactly $N$ Q-values where $N$ is the catalog size (200). PennyLane QML must map outputs rigorously between $0.0$ and $1.0$.

### 4.2 Integration Testing (API)
- **Target**: `tests/test_api.py`
- **Criteria**: The FastAPI endpoints must return HTTP 200 OK for valid inputs and appropriate HTTP 400 Client Error responses for malformed states (e.g., passing 2 items instead of the required sequence length of 3).

### 4.3 Environment Testing (MLOps)
- **Target**: Docker and MLflow configuration.
- **Criteria**: `docker build -t mlops-recommender .` must exit with code 0.

## 5. Pass/Fail Criteria
- A test suite passes if 100% of the assertions in `pytest` are successful.
- MLOps tests pass if the model artifact is successfully generated and visible in the `mlruns/` directory following a `train_pipeline.py` execution.

## 6. Test Environment
- **Operating System**: Agnostic (via Docker), natively developed on Windows/Linux.
- **Python Version**: 3.9+
- **Key Dependencies**: `pytest`, `fastapi.testclient`, `torch`, `pennylane`.

## 7. Deliverables
- Automated test scripts (`tests/` directory).
- This formal test plan document.
- CI/CD build logs (when integrated into GitHub Actions or similar).
