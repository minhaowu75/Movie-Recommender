import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(
            self,
            num_users,
            num_movies,
            embedding_size=256,
            hidden_dim=256,
            dropout_rate=0.2
    ):
        super(MLPModel, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim

        self.user_embedding = nn.Embedding(num_embeddings=self.num_users, embedding_dim = self.embedding_size)
        self.movie_embedding = nn.Embedding(num_embeddings=self.num_movies, embedding_dim=self.embedding_size)

        self.lin1 = nn.Linear(2 * self.embedding_size, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, 1)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.relu = nn.ReLU()
    
    def forward(self, users, movies):
        user_embed = self.user_embedding(users)
        movie_embed = self.movie_embedding(movies)

        concat_embed = torch.cat([user_embed, movie_embed], dim=1)

        input = self.relu(self.lin1(concat_embed))
        input = self.dropout(input)
        res = self.lin2(input)

        return res