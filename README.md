<<<<<<< HEAD
# Q-Learning based Music Recommendation System
=======
# Recommendation System using Reinforcement Learning
>>>>>>> 531c183ab7cf8914981d7373c877a42a5a2aa93d

## Abstract

This project implements a Reinforcement Learning-based music recommendation system from scratch using Python. By formulating the recommendation problem as a Markov Decision Process (MDP), the system models user interactions as a sequence of states and actions. The recommendation engine is powered by the **Q-Learning** algorithm, which learns to recommend songs by maximizing the expected cumulative reward based on the user's sequential interaction history.

The project includes:
- A **Synthetic Dataset Generator** that mimics user interactions and latent genre preferences.
- A **Popularity Baseline** model for performance comparison.
- A **Q-Learning Agent** that explores and exploits user preferences using an epsilon-greedy strategy and Bellman updates.
- An **Evaluation Pipeline** measuring performance using standard ranking metrics: Precision@K, Recall@K, F1 Score@K, and NDCG@K.

## Project Structure

- `main.py`: The entry point of the application. It handles data loading, state conversion, model training, and evaluation.
- `src/dataset_generator.py`: Generates a synthetic dataset of user-song interactions and saves it to `data/dataset.csv`.
- `src/q_learning.py`: Contains the implementation of the Q-Learning agent, including Q-table management, action selection, and Q-value updates.
- `src/baseline.py`: Implements a simple popularity-based recommendation model to serve as a baseline.
- `src/evaluation.py`: Provides functions to compute ranking metrics (Precision, Recall, F1, NDCG).
- `src/dataset_prep.py`: Handles data splitting and converting sequential interaction logs into MDP states (state transitions).

## Setup Instructions

### Prerequisites

Ensure you have Python 3.7+ installed. The project relies on standard data science libraries.

<<<<<<< HEAD
You can install the required dependencies using `pip`:
=======
You can install the required dependencies using `pip` or requirements.txt: 
>>>>>>> 531c183ab7cf8914981d7373c877a42a5a2aa93d

```bash
pip install pandas numpy matplotlib tqdm
```

### Running the Project

**1. Generate the Dataset**

First, you need to generate the synthetic dataset that the models will train and test on. Run the dataset generator script:

```bash
python -m src.dataset_generator
```
This will create a `data/dataset.csv` file containing the simulated user interactions.

**2. Train and Evaluate Models**

Run the main script to start the training process for the Q-Learning agent, evaluate it against the Popularity Baseline, and generate performance plots:

```bash
python main.py
```

### Outputs

After running `main.py`, the script will output the evaluation metrics directly to the console. Additionally, it will generate and save two visual plots in the root directory:
- `training_rewards.png`: A line plot showing the total reward accumulated by the Q-Learning agent per epoch during training.
- `evaluation_comparison.png`: A bar chart comparing the Precision, Recall, F1 Score, and NDCG between the Q-Learning agent and the Popularity Baseline at K=5.
