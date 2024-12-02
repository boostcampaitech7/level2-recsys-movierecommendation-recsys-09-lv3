import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix
import torch
import seaborn as sns
import matplotlib.pyplot as plt

class iALS:
    def __init__(self, num_users, num_items, num_factors, reg_lambda, alpha, genre_tensor=None, device='cpu'):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.reg_lambda = reg_lambda
        self.alpha = alpha
        self.genre_tensor = genre_tensor
        self.device = device
        
        self.P = torch.rand((num_users, num_factors), requires_grad=False, device=device)
        self.Q = torch.rand((num_items, num_factors), requires_grad=False, device=device)
        
        if self.genre_tensor is not None:
            genre_factors = torch.rand((genre_tensor.shape[1], num_factors), requires_grad=False, device=device)
            self.Q += torch.matmul(self.genre_tensor, genre_factors)
    
    def train_step(self, interaction_matrix, num_iterations=10):
        confidence_matrix = 1 + self.alpha * interaction_matrix

        for _ in range(num_iterations):
            for u in range(self.num_users):
                Cu = torch.diag(confidence_matrix[u, :])
                A = self.Q.T @ Cu @ self.Q + self.reg_lambda * torch.eye(self.num_factors, device=self.device)
                b = self.Q.T @ Cu @ interaction_matrix[u, :].t()
                self.P[u] = torch.linalg.solve(A, b)
            
            for i in range(self.num_items):
                Ci = torch.diag(confidence_matrix[:, i])
                A = self.P.T @ Ci @ self.P + self.reg_lambda * torch.eye(self.num_factors, device=self.device)
                b = self.P.T @ Ci @ interaction_matrix[:, i]
                self.Q[i] = torch.linalg.solve(A, b)
    
    def predict(self):
        return self.P @ self.Q.t()
    
    def save_factors(self, path):
        torch.save({'P': self.P, 'Q': self.Q}, path)
    
    def load_factors(self, path):
        data = torch.load(path)
        self.P = data['P']
        self.Q = data['Q']

def recommend_items(model, user_id, k=10, train_interactions=None, valid_interactions=None):
    user_embedding = model.P[user_id]
    scores = torch.matmul(user_embedding, model.Q.t()).detach().cpu().numpy()
    
    if train_interactions is not None and valid_interactions is not None:
        interacted_items = (train_interactions + valid_interactions).cpu().numpy().astype(bool)
    elif train_interactions is not None:
        interacted_items = train_interactions.cpu().numpy().astype(bool)
    elif valid_interactions is not None:
        interacted_items = valid_interactions.cpu().numpy().astype(bool)
    else:
        interacted_items = np.zeros_like(scores, dtype=bool)
    
    scores[interacted_items] = -np.inf
    recommended_items = np.argsort(scores)[-k:][::-1]
    return recommended_items

def calculate_normalized_recall(model, valid_tensor, k=10):
    total_recall = 0
    num_users = valid_tensor.shape[0]

    for user_id in range(num_users):
        actual_items = valid_tensor[user_id].nonzero(as_tuple=True)[0].tolist()
        if not actual_items:
            continue
        recommended_items = recommend_items(model, user_id, k)
        hits = len(set(actual_items) & set(recommended_items))
        denominator = min(k, len(actual_items))
        if denominator > 0:
            total_recall += hits / denominator

    return total_recall / num_users

def plot_user_item_genre(user_data, user_id, ax):
    user_items = user_data[user_data['user'] == user_id]
    user_genres = user_items.explode('genre')['genre'].value_counts()

    sns.barplot(x=user_genres.index, y=user_genres.values, palette='viridis', ax=ax)
    ax.set_title(f'User {user_id} - Item Genre Distribution')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# 데이터 준비 및 전처리
def load_or_prepare_data():
    if os.path.exists('train_matrix.npy') and os.path.exists('valid_matrix.npy'):
        train_matrix = np.load('train_matrix.npy')
        valid_matrix = np.load('valid_matrix.npy')
        genre_matrix_full = np.load('genre_matrix_full.npy')
    else:
        train_matrix, valid_matrix, genre_matrix_full = prepare_data()
    return train_matrix, valid_matrix, genre_matrix_full


def prepare_data():
    ratings = pd.read_csv('./data/train/train_ratings.csv')
    user_map = {u: i for i, u in enumerate(ratings["user"].unique())}
    item_map = {i: j for j, i in enumerate(ratings["item"].unique())}
    ratings["user_id"] = ratings["user"].map(user_map)
    ratings["item_id"] = ratings["item"].map(item_map)

    genres = pd.read_csv('./data/train/genres.tsv', sep='\t')
    genres_grouped = genres.groupby('item')['genre'].apply(list).reset_index()
    genres_grouped['item_id'] = genres_grouped['item'].map(item_map)
    genres_grouped = genres_grouped.dropna(subset=['item_id']).astype({'item_id': int})

    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(genres_grouped['genre'])
    genre_matrix = genre_encoded.astype(np.float32)

    genre_matrix_full = np.zeros((len(item_map), len(mlb.classes_)), dtype=np.float32)
    genre_matrix_full[genres_grouped['item_id'], :] = genre_matrix

    train_data, valid_data = split_train_valid(ratings)

    train_matrix = create_interaction_matrix(train_data, user_map, item_map)
    valid_matrix = create_interaction_matrix(valid_data, user_map, item_map)

    np.save('train_matrix.npy', train_matrix)
    np.save('valid_matrix.npy', valid_matrix)
    np.save('genre_matrix_full.npy', genre_matrix_full)

    return train_matrix, valid_matrix, genre_matrix_full


def split_train_valid(ratings, train_ratio=0.8):
    train_data_list, valid_data_list = [], []
    for user in ratings['user'].unique():
        user_data = ratings[ratings['user'] == user].sort_values('time')
        n_train = int(len(user_data) * train_ratio)
        train_data_list.append(user_data.iloc[:n_train])
        valid_data_list.append(user_data.iloc[n_train:])
    return pd.concat(train_data_list), pd.concat(valid_data_list)


def create_interaction_matrix(data, user_map, item_map):
    num_users = len(user_map)
    num_items = len(item_map)
    matrix = np.zeros((num_users, num_items), dtype=np.float32)
    for _, row in data.iterrows():
        matrix[row["user_id"], row["item_id"]] = 1.0
    return matrix


# 모델 학습 및 평가
def train_and_evaluate_model(train_tensor, valid_tensor, genre_tensor, device):
    num_users, num_items = train_tensor.shape
    num_factors = 10
    reg_lambda = 0.1
    alpha = 40

    model = iALS(num_users, num_items, num_factors, reg_lambda, alpha, genre_tensor=genre_tensor, device=device)
    model.train_step(train_tensor, num_iterations=10)
    model.save_factors("latent_factors.pth")

    normalized_recall = calculate_normalized_recall(model, valid_tensor, k=10)
    print(f"Normalized Recall@10: {normalized_recall:.4f}")
    return model


# 추천 결과 생성 및 저장
def generate_recommendations(model, train_tensor, valid_tensor, user_map, item_map, k=10):
    reverse_user_map = {i: u for u, i in user_map.items()}
    reverse_item_map = {i: item for item, i in item_map.items()}
    recommendations = []

    for user_id in range(model.num_users):
        original_user_id = reverse_user_map[user_id]
        train_interaction = train_tensor[user_id]
        valid_interaction = valid_tensor[user_id]
        recommended_items = recommend_items(model, user_id, k, train_interaction, valid_interaction)
        for item_id in recommended_items:
            original_item_id = reverse_item_map[item_id]
            recommendations.append([original_user_id, original_item_id])

    recommendations_df = pd.DataFrame(recommendations, columns=['user', 'item'])
    recommendations_df.to_csv('recommendations.csv', index=False)
    print("추천 결과가 'recommendations.csv' 파일로 저장되었습니다.")
    return recommendations_df


# 메인 실행 함수
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_matrix, valid_matrix, genre_matrix_full = load_or_prepare_data()
    train_tensor = torch.FloatTensor(train_matrix).to(device)
    valid_tensor = torch.FloatTensor(valid_matrix).to(device)
    genre_tensor = torch.FloatTensor(genre_matrix_full).to(device)

    model = train_and_evaluate_model(train_tensor, valid_tensor, genre_tensor, device)
    ratings = pd.read_csv('./data/train/train_ratings.csv')
    user_map = {u: i for i, u in enumerate(ratings["user"].unique())}
    item_map = {i: j for j, i in enumerate(ratings["item"].unique())}

    generate_recommendations(model, train_tensor, valid_tensor, user_map, item_map, k=10)


if __name__ == "__main__":
    main()