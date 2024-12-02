import pandas as pd
import numpy as np

def calculate_time_weights(train_data):
    """
    시간 가중치 계산
    
    Parameters:
        train_data (pd.DataFrame): train 데이터

    Returns:
        pd.Series: 아이템 별 시간 가중치
    """
    # Step 1: Normalize time using sigmoid-based function
    normalized_time = (train_data['time'] - train_data['time'].min()) / (train_data['time'].max() - train_data['time'].min())
    time_weights = 1 / (1 + np.exp(-normalized_time))

    # Step 2: Calculate item-wise average time weights
    item_time_weights = time_weights.groupby(train_data['item']).mean()

    return item_time_weights


def split_interactions(group, min_interactions=3):
    """
    사용자 그룹을 Train, Validation, Test로 분리하는 함수.
    
    Parameters:
        group (DataFrame): 특정 사용자의 상호작용 데이터 그룹.
        min_interactions (int): 최소 상호작용 개수 (기본값 3). -> 해당 유저 없음
    
    Returns:
        train, val, test (DataFrame, Series, Series): 
        - Train 데이터 (DataFrame)
        - Validation 데이터 (Series)
        - Test 데이터 (Series)
    """
    if len(group) >= min_interactions:
        train = group.iloc[:-2]  # 마지막 두 개를 제외한 나머지 -> Train
        val = group.iloc[-2]     # 마지막에서 두 번째 -> Validation
        test = group.iloc[-1]    # 마지막 -> Test
    elif len(group) == 2:
        train = None
        val = group.iloc[-2]     # 첫 번째 -> Validation
        test = group.iloc[-1]    # 마지막 -> Test
    elif len(group) == 1:
        train = None
        val = None
        test = group.iloc[-1]    # 마지막 -> Test
    else:
        train, val, test = None, None, None

    return train, val, test

def create_train_val_test(train_df):
    """
    사용자 상호작용 데이터를 Train, Validation, Test 데이터셋으로 분리하는 함수.
    
    Parameters:
        train_df (DataFrame): 사용자 상호작용 데이터.
    
    Returns:
        train_data, val_data, test_data (DataFrame): 
        - Train 데이터
        - Validation 데이터
        - Test 데이터
    """
    train_data = []
    val_data = []
    test_data = []

    for user, group in train_df.groupby('user'):
        train, val, test = split_interactions(group)
        if train is not None:
            train_data.append(train)
        if val is not None:
            val_data.append(val)
        if test is not None:
            test_data.append(test)

    # 각각의 데이터프레임으로 병합
    train_data = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
    val_data = pd.DataFrame(val_data)
    test_data = pd.DataFrame(test_data)

    return train_data, val_data, test_data
