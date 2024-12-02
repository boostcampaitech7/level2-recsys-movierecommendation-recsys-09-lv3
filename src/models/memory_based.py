from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
from src.evaluation.metrics import precision_at_k, recall_at_k

def calculate_genre_similarity(genre_data):
    """
    장르 유사도 구하기
    
    Parameters:
    genre_data (pd.DataFrame)
    
    Returns:
    np.ndarray: Genre-based item similarity matrix.
    """
    item_genres = genre_data.groupby('item')['genre'].apply(list).reset_index()

    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(
        mlb.fit_transform(item_genres['genre']),
        index=item_genres['item'],
        columns=mlb.classes_
    )

    genre_similarity = cosine_similarity(genre_encoded)
    
    

    return genre_similarity 


def calculate_item_similarity(train_data):
    """
    아이템 유사도 구하기

    Parameters:
    train_data (pd.DataFrame)

    Returns:
    np.ndarray: Item-item similarity matrix.
    """
    # 1. 사용자-아이템 상호작용 행렬 생성
    user_item_matrix = train_data.pivot(index='user', columns='item', values='time').notna().astype(int)

    # 2. CSR 형식으로 변환
    sparse_matrix = csr_matrix(user_item_matrix)

    # 3. 아이템-사용자 행렬로 전치
    item_user_matrix = user_item_matrix.T

    # 4. 코사인 유사도 계산
    similarity_matrix = cosine_similarity(item_user_matrix)

    return user_item_matrix, similarity_matrix


def recommend_with_time_weight(user_id, user_item_matrix, similarity_matrix, time_weights, k=10):
    """
    시간 가중치 적용한 Top10개 추천

    Parameters:
        user_id: ID of the user for whom to generate recommendations.
        user_item_matrix: User-item interaction matrix (DataFrame format).
        similarity_matrix: Item-item similarity matrix.
        time_weights: Time weight vector (per item).
        k: Number of items to recommend (default is 10).

    Returns:
        List of recommended items (Top-K).
    """
    # User's rated items
    user_ratings = user_item_matrix.loc[user_id].values
    rated_items = np.where(user_ratings > 0)[0]  # Indices of items the user interacted with

    # Summing similarities for rated items
    scores = similarity_matrix[rated_items].sum(axis=0)

    # Exclude items the user has already rated
    scores[rated_items] = -1

    # Apply time weights
    scores = scores * time_weights

    # Get top-K recommended items
    top_k_items = np.argsort(scores)[-k:][::-1]  # Sort in descending order and get top K
    return user_item_matrix.columns[top_k_items]


def recommend_all_users(user_item_matrix, similarity_matrix, time_weights, k=10):
    """
    모든 유저에 대한 추천
    
    Parameters:
        user_item_matrix: User-item interaction matrix (DataFrame format).
        similarity_matrix: Item-item similarity matrix.
        time_weights: Time weight vector (per item).
        k: Number of items to recommend (default is 10).
    
    Returns:
        dict: Recommended items for each user.
    """
    recommendations = {}
    for user_id in user_item_matrix.index:
        top_k_items = recommend_with_time_weight(user_id, user_item_matrix, similarity_matrix, time_weights, k)
        recommendations[user_id] = top_k_items
    return recommendations


def evaluate_model(user_item_matrix, similarity_matrix, time_weights, val_data, test_data, k=10):
    """
    Valid/Test 셋에 대한 Precision@K and Recall@K.
    
    Parameters:
        user_item_matrix (pd.DataFrame): User-item interaction matrix.
        similarity_matrix (np.ndarray): Item-item similarity matrix.
        time_weights (np.ndarray): Time weight vector.
        val_data (pd.DataFrame): Validation data.
        test_data (pd.DataFrame): Test data.
        k (int): Top-K value for evaluation.
    
    Returns:
        dict: Validation and Test Precision@K and Recall@K scores.
    """
    # Prepare actual data
    val_actual = val_data.groupby('user')['item'].apply(list).tolist()
    test_actual = test_data.groupby('user')['item'].apply(list).tolist()

    # Generate predictions for all users
    recommendations = recommend_all_users(user_item_matrix, similarity_matrix, time_weights, k)
    predicted = [recommendations[user_id] for user_id in user_item_matrix.index]

    # Calculate scores for Validation
    val_precision = precision_at_k(val_actual, predicted, k)
    val_recall = recall_at_k(val_actual, predicted, k)

    # Calculate scores for Test
    test_precision = precision_at_k(test_actual, predicted, k)
    test_recall = recall_at_k(test_actual, predicted, k)

    return {
        "Validation Precision@K": val_precision,
        "Validation Recall@K": val_recall,
        "Test Precision@K": test_precision,
        "Test Recall@K": test_recall,
    }
