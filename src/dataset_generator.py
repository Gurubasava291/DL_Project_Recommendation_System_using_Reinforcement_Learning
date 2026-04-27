import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_music_dataset(num_users=100, num_songs=200, interactions_per_user=50, output_path='data/dataset.csv'):
    """
    Generates a synthetic music recommendation dataset mimicking an MDP structure.
    Users have underlying preferences for certain 'genres' (latent features).
    """
    print(f"Generating synthetic dataset with {num_users} users and {num_songs} songs...")

    # Assign each song to a "genre" (0 to 4)
    song_genres = {song_id: random.randint(0, 4) for song_id in range(num_songs)}
    
    # Assign each user a preferred genre
    user_pref_genres = {user_id: random.randint(0, 4) for user_id in range(num_users)}

    data = []
    
    for user_id in range(num_users):
        preferred_genre = user_pref_genres[user_id]
        
        for _ in range(interactions_per_user):
            # 70% chance to be recommended a song from their preferred genre, 30% random
            if random.random() < 0.7:
                # Find songs matching preferred genre
                matching_songs = [s_id for s_id, g in song_genres.items() if g == preferred_genre]
                song_id = random.choice(matching_songs)
            else:
                song_id = random.randint(0, num_songs - 1)
                
            # If the song matches user's preferred genre, high chance of liking (reward = 1)
            # Else, lower chance
            if song_genres[song_id] == preferred_genre:
                reward = 1 if random.random() < 0.8 else 0  # 80% chance of liking
            else:
                reward = 1 if random.random() < 0.2 else 0  # 20% chance of liking
                
            data.append([user_id, song_id, reward])
            
    df = pd.DataFrame(data, columns=['user_id', 'song_id', 'reward'])
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully and saved to {output_path}")
    print(df.head(10))
    print(f"\nTotal interactions: {len(df)}")
    print(f"Positive rewards (likes): {df['reward'].sum()} ({(df['reward'].sum()/len(df))*100:.2f}%)")

if __name__ == "__main__":
    generate_music_dataset()
