from src.data.memory_data import calculate_time_weights, create_train_val_test
#from src.utils import generate_rating_matrix_valid, generate_rating_matrix_test
from src.models.memory_based import  recommend_all_users, calculate_item_similarity, calculate_genre_similarity, evaluate_model
from src.utils.utils import generate_submission_file
import pandas as pd

def main():
    # 데이터 로드
    data = pd.read_csv("data/train/train_ratings.csv") 
    genre = pd.read_csv("data/train/genres.tsv", sep="\t")
    print("Data loaded.") 
    
    # 시간 가중치 구하기
    time_weights = calculate_time_weights(data)
    
    # Train/Test 분리
    train_data, val_data, test_data = create_train_val_test(data)
    print("Train/Validation/Test split completed.")

    # 장르 유사도 구하기
    genre_matrix = calculate_genre_similarity(genre)
    print("User-item matrix created.")

    # 아이템 유사도 구하기
    user_item_matrix,similarity_matrix = calculate_item_similarity(train_data)
    print("Item similarity matrix calculated.")

    combined_similarity = 0.9 * similarity_matrix + 0.1 * genre_matrix

    # 모델 평가
    scores = evaluate_model(user_item_matrix, combined_similarity, time_weights, val_data, test_data, k=10)

    #  결과 출력
    print("\nEvaluation Results:")
    print(f"Validation Precision@10: {scores['Validation Precision@K']:.4f}")
    print(f"Validation Recall@10: {scores['Validation Recall@K']:.4f}")
    print(f"Test Precision@10: {scores['Test Precision@K']:.4f}")
    print(f"Test Recall@10: {scores['Test Recall@K']:.4f}")

    # 전체 데이터셋
    user_item_matrix,similarity_matrix = calculate_item_similarity(data)
    combined_similarity = 0.9 * similarity_matrix + 0.1 * genre_matrix
    
    recommends =  recommend_all_users(user_item_matrix, combined_similarity, time_weights, k=10)
    print("Generating recommendations for all users")
    
    output_file = "saved/submit/submission.csv"
    result = []

    # 사용자별 추천 결과를 저장
    for user_id, items in recommends.items():  
        for item in items:
            result.append((user_id, item))

    # DataFrame으로 변환하여 저장
    submission_df = pd.DataFrame(result, columns=["user", "item"])
    submission_df.to_csv(output_file, index=False)

    print(f"Submission file saved at {output_file}")
    print("Submission file creation complete. Program finished successfully.")


if __name__ == "__main__":
    main()
