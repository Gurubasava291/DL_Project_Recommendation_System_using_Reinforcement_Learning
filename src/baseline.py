import pandas as pd

class PopularityBaseline:
    def __init__(self):
        self.popular_songs = []
        
    def fit(self, train_df):
        # Calculate popularity by summing rewards (likes)
        popularity = train_df.groupby('song_id')['reward'].sum().sort_values(ascending=False)
        self.popular_songs = popularity.index.tolist()
        
    def recommend(self, k=5):
        # Recommend the top k popular songs
        return self.popular_songs[:k]
