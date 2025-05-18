import pandas as pd
import torch
from sklearn import model_selection
from torch.utils.data import Dataset

ratings = pd.read_csv('data/ratings.csv')

class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, item):
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]

        return {
            "users": torch.tensor(users, dtype=torch.long),
            "movies": torch.tensor(movies, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.float)
        }
    
def preprocess_data(label_encoder_user, label_encoder_movie):
    ratings.userId = label_encoder_user.fit_transform(ratings.userId.values)
    ratings.movieId = label_encoder_movie.fit_transform(ratings.movieId.values)

    train, val = model_selection.train_test_split(ratings, test_size=0.1, random_state=3, stratify=ratings.rating.values)

    train_dataset = MovieDataset(users=train.userId.values,
                                movies=train.movieId.values,
                                ratings=train.rating.values
                                )

    valid_dataset = MovieDataset(users=val.userId.values,
                                movies=val.movieId.values,
                                ratings=val.rating.values
                                )
    
    return train_dataset, valid_dataset