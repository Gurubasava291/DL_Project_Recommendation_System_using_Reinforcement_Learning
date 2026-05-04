# Project Report: Full-Stack MLOps Recommender System

## 1. Executive Summary
This project aims to build a comprehensive, production-ready music recommendation system integrating Machine Learning (ML), Deep Learning (DL), and Quantum Machine Learning (QML) paradigms. By adopting an MLOps lifecycle, the system guarantees reproducibility, scalable deployment, and strict metric tracking, transitioning a traditional Jupyter Notebook-style RL model into a robust Full-Stack web application.

## 2. Introduction
Music recommendation represents a sequential decision-making problem. While earlier iterations of this project utilized a basic tabular Q-Learning approach, the state space limitations necessitated a shift to neural network approximations. Concurrently, the operational overhead required a formal MLOps methodology to manage training runs, model artifacts, and service deployment.

## 3. System Architecture
The system follows a three-tier architecture:
- **Frontend Layer**: A lightweight Vanilla HTML/CSS/JS interface providing a sleek, dark-themed UI for users to query the recommendation API.
- **Backend Service Layer**: A FastAPI server handling HTTP requests, performing data validation using Pydantic, and routing requests to the appropriate inference model.
- **Model / MLOps Layer**: 
  - PyTorch for Deep Q-Networks.
  - PennyLane for Quantum Variational Classifiers.
  - MLflow for experiment tracking and model registry.
  - Docker for unified containerization.

## 4. Implementation Details
### 4.1 Deep Q-Network (DQN)
The DQN agent (`src/dqn_agent.py`) replaces the legacy Q-table. It accepts an input state consisting of an embedded sequence of the user's last 3 interacted songs. A multi-layer perceptron predicts the Q-values for all 200 available songs, optimized via Mean Squared Error against the Bellman target.

### 4.2 Quantum Machine Learning (QML) Classifier
To investigate quantum utility in recommendation tasks, `src/qml_classifier.py` introduces a Parameterized Quantum Circuit. Using PennyLane, it encodes continuous latent features into quantum states and applies strongly entangling layers to classify user segmentation based on Pauli-Z expectation measurements.

### 4.3 Training Pipeline
The `train_pipeline.py` script orchestrates the data loading, state formulation, and model training. It seamlessly integrates with MLflow (`mlflow.start_run()`) to log critical hyperparameters (learning rate, discount factor $\gamma$, exploration rate $\epsilon$) and step-wise metrics (loss, reward).

## 5. Deployment and Operations
The entire stack is encapsulated within a `Dockerfile`. The image is built upon a lightweight Python 3.9 image, installing all required dependencies (FastAPI, PyTorch, PennyLane) defined in `requirements.txt`. The container exposes port 8000 for the Uvicorn server, ensuring consistent execution across any Docker-compatible infrastructure.

## 6. Conclusion
The Full-Stack MLOps Recommender System successfully demonstrates the convergence of advanced AI methodologies (DL, QML) with modern software engineering practices. The resulting application is highly modular, easily testable, and prepared for continuous integration and delivery.
