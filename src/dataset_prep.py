import pandas as pd
import numpy as np

def load_and_split_data(filepath='data/dataset.csv', test_ratio=0.2):
    """
    Loads dataset and splits into train and test without scikit-learn.
    We split chronologically per user to simulate real-world sequence prediction.
    """
    df = pd.read_csv(filepath)
    
    train_data = []
    test_data = []
    
    # Sort by user_id to ensure consistency
    for user_id, group in df.groupby('user_id'):
        split_idx = int(len(group) * (1 - test_ratio))
        user_train = group.iloc[:split_idx]
        user_test = group.iloc[split_idx:]
        
        train_data.append(user_train)
        test_data.append(user_test)
        
    train_df = pd.concat(train_data).reset_index(drop=True)
    test_df = pd.concat(test_data).reset_index(drop=True)
    
    return train_df, test_df

def get_mdp_states(df, state_length=3):
    """
    Converts user interactions into MDP states.
    State S_t = [song_{t-k}, ..., song_{t-1}]
    Action A_t = song_t
    Reward R_t = reward_t
    """
    states = []
    
    for user_id, group in df.groupby('user_id'):
        songs = group['song_id'].tolist()
        rewards = group['reward'].tolist()
        
        # We need at least state_length + 1 items to form one state transition
        if len(songs) <= state_length:
            continue
            
        for i in range(len(songs) - state_length):
            state = tuple(songs[i : i + state_length])
            action = songs[i + state_length]
            reward = rewards[i + state_length]
            states.append((user_id, state, action, reward))
            
    return states
