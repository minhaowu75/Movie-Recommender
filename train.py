from data import preprocess_data
from log_training import log
from model import MLPModel

from collections import defaultdict
from sklearn import preprocessing
from sklearn.metrics import root_mean_squared_error
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

def calculate_precision_recall(user_ratings, k, threshold):
    user_ratings.sort(key=lambda x: x[0], reverse=True)
    n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
    n_rec_k = sum(est >= threshold for est, _ in user_ratings[:k])
    n_rel_and_rec_k = sum((true_r >= threshold) and (est >= threshold) for est, true_r in user_ratings[:k])

    precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
    recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return precision, recall

def train():
    if torch.cuda.is_available():
        print('cuda is available')
        device = torch.device('cuda')
    else:
        print('cpu is enabled')
        device = torch.device('cpu')

    BATCH_SIZE = 32
    EPOCHS = 2

    label_encoder_user = preprocessing.LabelEncoder()
    label_encoder_movie = preprocessing.LabelEncoder()

    train_dataset, valid_dataset = preprocess_data(label_encoder_user, label_encoder_movie)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    model = MLPModel(
        num_users=len(label_encoder_user.classes_),
        num_movies=len(label_encoder_movie.classes_),
        embedding_size=128,
        hidden_dim=256,
        dropout_rate=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()

    total_loss = 0
    log_progress_step = 100
    losses = []
    train_dataset_size = len(train_dataset)
    print(f"Training on {train_dataset_size} samples...")

    model.train()

    for e in range(EPOCHS):
        step_count = 0
        for i, train_data in enumerate(train_loader):
            #print(train_data['movies'])
            out = model(users=train_data['users'].to(device), movies=train_data['movies'].to(device))

            out = out.squeeze()
            ratings = (train_data['ratings'].to(torch.float32).to(device))

            l = loss_func(out, ratings)
            total_loss += l.sum().item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            step_count += len(train_data["users"])

            if (step_count % log_progress_step == 0 or i == len(train_loader) - 1):
                log(e, step_count, total_loss, log_progress_step, train_dataset_size, losses, EPOCHS)
                total_loss = 0

    user_ratings_comparison = defaultdict(list)
    user_precisions = dict()
    user_based_recalls = dict()

    k = 50
    threshold = 3

    val_pred = []
    gold = []

    with torch.inference_mode():
        for i, valid_data in enumerate(val_loader):
            users = valid_data['users'].to(device)
            movies = valid_data['movies'].to(device)
            ratings = valid_data['ratings'].to(device)
            out = model(users, movies)

            for user, pred, true in zip(users, out, ratings):
                user_ratings_comparison[user.item()].append((pred[0].item(), true.item()))
            
            val_pred.extend(out.cpu().numpy())
            gold.extend(ratings.cpu().numpy())

    for user_id, user_ratings in user_ratings_comparison.items():
        precision, recall = calculate_precision_recall(user_ratings, k, threshold)
        user_precisions[user_id] = precision
        user_based_recalls[user_id] = recall

    average_precision = sum(prec for prec in user_precisions.values()) / len(user_precisions)
    average_recall = sum(rec for rec in user_based_recalls.values()) / len(user_based_recalls)

    rms = root_mean_squared_error(gold, val_pred)
    print(f"\nRMS Error: {rms:.4f}")
    print(f"\nprecision @ {k}: {average_precision:.4f}")
    print(f"\nrecall @ {k}: {average_recall:.4f}")

if __name__ == "__main__":
    train()
    


