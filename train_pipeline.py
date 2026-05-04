import mlflow
import mlflow.pytorch
import pandas as pd
from src.dataset_prep import load_and_split_data, get_mdp_states
from src.dqn_agent import DQNAgent
import numpy as np

def train_model():
    print("Starting MLOps Training Pipeline...")
    
    # 1. Load Data
    train_df, test_df = load_and_split_data('data/dataset.csv', test_ratio=0.2)
    STATE_LENGTH = 3
    NUM_SONGS = 200
    train_mdp = get_mdp_states(train_df, state_length=STATE_LENGTH)
    
    # 2. Setup MLflow tracking
    mlflow.set_experiment("Music_Recommender_DQN")
    
    with mlflow.start_run(run_name="DQN_Run"):
        # Log parameters
        epochs = 10
        alpha = 0.001
        gamma = 0.9
        epsilon = 0.2
        mlflow.log_params({"epochs": epochs, "lr": alpha, "gamma": gamma, "epsilon": epsilon})
        
        # 3. Initialize Agent
        agent = DQNAgent(num_songs=NUM_SONGS, state_length=STATE_LENGTH, lr=alpha, gamma=gamma, epsilon=epsilon)
        
        # 4. Train Model
        print("Training Deep Q-Network...")
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_reward = 0
            np.random.shuffle(train_mdp)
            
            # Simple batching approximation
            for user_id, state, action, reward in train_mdp[:1000]: # Train on a subset for speed in this demo
                next_state = list(state[1:]) + [action]
                loss = agent.update(state, action, reward, next_state)
                epoch_loss += loss
                epoch_reward += reward
                
            # Log metrics
            avg_loss = epoch_loss / 1000
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("reward", epoch_reward, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Reward: {epoch_reward}")
            
        # 5. Save Model
        mlflow.pytorch.log_model(agent.model, "dqn_model")
        print("Training complete. Model logged to MLflow.")

        mlflow.set_tracking_uri("file:./mlruns")

if __name__ == "__main__":
    train_model()
