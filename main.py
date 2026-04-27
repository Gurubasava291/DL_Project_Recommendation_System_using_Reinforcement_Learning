import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.dataset_prep import load_and_split_data, get_mdp_states
from src.baseline import PopularityBaseline
from src.q_learning import QLearningRecommender
from src.evaluation import precision_at_k, recall_at_k, f1_score_at_k, ndcg_at_k
from tqdm import tqdm
import warnings

# Suppress plot warnings for clean output
warnings.filterwarnings("ignore")

def main():
    print("==================================================")
    print(" Q-Learning based Music Recommendation System")
    print("==================================================")
    
    print("\n--- Step 2: Dataset Preprocessing ---")
    train_df, test_df = load_and_split_data('data/dataset.csv', test_ratio=0.2)
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    STATE_LENGTH = 3
    num_songs = 200 # As defined in dataset_generator
    
    print("\nConverting to MDP states...")
    train_mdp = get_mdp_states(train_df, state_length=STATE_LENGTH)
    test_mdp = get_mdp_states(test_df, state_length=STATE_LENGTH)
    
    print(f"Train MDP transitions: {len(train_mdp)}")
    print(f"Test MDP transitions: {len(test_mdp)}")
    
    print("\n--- Step 3: Popularity Baseline ---")
    baseline = PopularityBaseline()
    baseline.fit(train_df)
    print("Baseline model trained.")
    
    print("\n--- Step 4 & 5: Q-Learning Training ---")
    agent = QLearningRecommender(num_songs=num_songs, state_length=STATE_LENGTH, alpha=0.1, gamma=0.9, epsilon=0.2)
    
    epochs = 15
    rewards_history = []
    
    for epoch in range(epochs):
        epoch_reward = 0
        np.random.shuffle(train_mdp) # Shuffle transitions for training
        
        for user_id, state, action, reward in train_mdp:
            # Approximate next_state by shifting the window. 
            next_state = list(state[1:]) + [action]
            next_state = tuple(next_state)
            
            agent.update(state, action, reward, next_state)
            epoch_reward += reward
            
        rewards_history.append(epoch_reward)
        print(f"Epoch {epoch+1}/{epochs} | Total Reward: {epoch_reward}")
        
    print("\n--- Step 6: Evaluation ---")
    
    # Organize test data by user state to evaluate
    test_relevant_items = {}
    for user_id, group in test_df.groupby('user_id'):
        songs = group['song_id'].tolist()
        rewards = group['reward'].tolist()
        
        if len(songs) <= STATE_LENGTH:
            continue
            
        for i in range(len(songs) - STATE_LENGTH):
            state = tuple(songs[i : i + STATE_LENGTH])
            action = songs[i + STATE_LENGTH]
            reward = rewards[i + STATE_LENGTH]
            
            if reward == 1: # Relevant item
                if state not in test_relevant_items:
                    test_relevant_items[state] = set()
                test_relevant_items[state].add(action)

    # Evaluate Baseline
    base_precisions, base_recalls, base_f1s, base_ndcgs = [], [], [], []
    ql_precisions, ql_recalls, ql_f1s, ql_ndcgs = [], [], [], []
    
    K = 5
    print(f"Evaluating both models at K={K}...")
    for state, relevant in test_relevant_items.items():
        relevant_list = list(relevant)
        
        # Baseline recs
        base_recs = baseline.recommend(k=K)
        base_precisions.append(precision_at_k(base_recs, relevant_list, K))
        base_recalls.append(recall_at_k(base_recs, relevant_list, K))
        base_f1s.append(f1_score_at_k(base_precisions[-1], base_recalls[-1]))
        base_ndcgs.append(ndcg_at_k(base_recs, relevant_list, K))
        
        # Q-Learning recs
        ql_recs = agent.recommend(state, k=K)
        ql_precisions.append(precision_at_k(ql_recs, relevant_list, K))
        ql_recalls.append(recall_at_k(ql_recs, relevant_list, K))
        ql_f1s.append(f1_score_at_k(ql_precisions[-1], ql_recalls[-1]))
        ql_ndcgs.append(ndcg_at_k(ql_recs, relevant_list, K))
        
    print(f"\n--- Final Results @ K={K} ---")
    print(f"Popularity Baseline:")
    print(f"Precision: {np.mean(base_precisions):.4f}")
    print(f"Recall:    {np.mean(base_recalls):.4f}")
    print(f"F1 Score:  {np.mean(base_f1s):.4f}")
    print(f"NDCG:      {np.mean(base_ndcgs):.4f}")
    
    print(f"\nQ-Learning Agent:")
    print(f"Precision: {np.mean(ql_precisions):.4f}")
    print(f"Recall:    {np.mean(ql_recalls):.4f}")
    print(f"F1 Score:  {np.mean(ql_f1s):.4f}")
    print(f"NDCG:      {np.mean(ql_ndcgs):.4f}")
    
    # Plotting Total Rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), rewards_history, marker='o', label='Total Reward', color='blue')
    plt.title('Q-Learning Training: Total Reward per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()
    plt.savefig('training_rewards.png')
    
    # Plotting Evaluation Comparison
    metrics = ['Precision', 'Recall', 'F1', 'NDCG']
    base_scores = [np.mean(base_precisions), np.mean(base_recalls), np.mean(base_f1s), np.mean(base_ndcgs)]
    ql_scores = [np.mean(ql_precisions), np.mean(ql_recalls), np.mean(ql_f1s), np.mean(ql_ndcgs)]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, base_scores, width, label='Popularity Baseline', color='salmon')
    plt.bar(x + width/2, ql_scores, width, label='Q-Learning', color='skyblue')
    plt.ylabel('Score')
    plt.title(f'Evaluation Metrics @ K={K}')
    plt.xticks(x, metrics)
    plt.legend()
    plt.savefig('evaluation_comparison.png')
    print("\nVisualizations saved as 'training_rewards.png' and 'evaluation_comparison.png'")

if __name__ == "__main__":
    main()
