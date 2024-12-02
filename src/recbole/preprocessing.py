import pandas as pd
from pathlib import Path


def preprocess_interactions(input_path, output_path):
    """
    Interaction 데이터를 전처리하여 RecBole 형식으로 저장합니다.
    """
    inter = pd.read_csv(input_path / 'train_ratings.csv')
    inter.columns = ['user_id:token', 'item_id:token', 'timestamp:float']
    inter.loc[:, 'label:float'] = 1

    # Interaction 파일 저장
    inter.to_csv(output_path / 'movierec.inter', sep="\t", index=False, encoding='utf-8')

    # User 파일 생성 및 저장
    inter_unique_users = inter['user_id:token'].drop_duplicates()
    inter_unique_users.to_csv(output_path / 'movierec.user', sep="\t", index=False, encoding='utf-8')

    return inter


def preprocess_item_features(input_path, interaction_data, output_path):
    """
    아이템 관련 데이터와 특징을 전처리하여 RecBole 형식으로 저장합니다.
    """
    # Interaction에 있는 유일한 아이템 추출
    inter_unique_items = interaction_data['item_id:token'].drop_duplicates()

    # 기본 특징 데이터 로드 및 병합
    title = pd.read_csv(input_path / 'titles.tsv', sep='\t')
    item_features = pd.merge(title, inter_unique_items, how='outer', on='item')

    # 장르 병합
    genre = pd.read_csv(input_path / 'genres.tsv', sep='\t')
    genre_item = genre.groupby('item')['genre'].apply(lambda x: " ".join(x)).reset_index()
    item_features = pd.merge(item_features, genre_item, how='outer', on='item')

    # 감독 병합
    directors_df = pd.read_csv(input_path / 'directors.tsv', sep='\t')
    directors_item_df = directors_df.groupby('item')['director'].apply(lambda x: " ".join(x)).reset_index()
    item_features = pd.merge(item_features, directors_item_df, how='outer', on='item')

    # 작가 병합
    writers_df = pd.read_csv(input_path / 'writers.tsv', sep='\t')
    writers_item_df = writers_df.groupby('item')['writer'].apply(lambda x: " ".join(x)).reset_index()
    item_features = pd.merge(item_features, writers_item_df, how='outer', on='item')

    # 연도 병합
    years_df = pd.read_csv(input_path / 'years.tsv', sep='\t')
    item_features = pd.merge(item_features, years_df, how='outer', on='item')

    # 결측치 처리 - 연도
    missing_year_items = item_features[item_features['year'].isna()]
    oldest_timestamps = interaction_data[interaction_data['item_id:token'].isin(missing_year_items['item'])] \
        .groupby('item_id:token')['timestamp:float'].min().reset_index()
    oldest_timestamps['year'] = pd.to_datetime(oldest_timestamps['timestamp:float'], unit='s').dt.year

    item_features = item_features.merge(oldest_timestamps, left_on='item', right_on='item_id:token', how='left')
    item_features['year'] = item_features['year_x'].combine_first(item_features['year_y'])
    item_features = item_features.drop(columns=['year_x', 'item_id:token', 'timestamp:float', 'year_y'])

    # 결측치 처리 - 감독과 작가
    item_features['director'] = item_features['director'].fillna("Unknown")
    item_features['writer'] = item_features['writer'].fillna('Unknown')

    # 컬럼명 재정의 및 저장
    item_features.columns = ['item_id:token', 'movie_title:token_seq', 'movie_genre:token_seq',
                             'movie_director:token_seq', 'movie_writer:token_seq', 'movie_release_year:float']
    item_features.to_csv(output_path / 'movierec.item', sep='\t', index=False)


if __name__ == "__main__":
    # 경로 설정
    input_dir = Path('/data/ephemeral/home/level2-recsys-movierecommendation-recsys-09-lv3/train')
    output_dir = Path('/data/ephemeral/home/level2-recsys-movierecommendation-recsys-09-lv3/RecBole/dataset/movierec')
    output_dir.mkdir(parents=True, exist_ok=True)  # 출력 경로 생성

    # Interaction 데이터 전처리
    interaction_data = preprocess_interactions(input_dir, output_dir)

    # Item 특징 데이터 전처리
    preprocess_item_features(input_dir, interaction_data, output_dir)
