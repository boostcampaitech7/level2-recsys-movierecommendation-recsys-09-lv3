import argparse
import torch
import numpy as np
import pandas as pd
from recbole.quick_start import load_data_and_model

from pathlib import Path


def save_predictions_1d(user_list, pred_list, user_id2token, item_id2token, output_path):
    """
    1차원 예측 결과를 저장하는 함수
    """
    result = []
    for user, item in zip(user_list, pred_list):
        user_token = user_id2token[user]
        item_token = item_id2token[item]
        if user_token == '[PAD]' or item_token == '[PAD]':
            continue
        result.append((int(user_token), int(item_token)))

    dataframe = pd.DataFrame(result, columns=["user", "item"])
    dataframe.sort_values(by='user', inplace=True)
    dataframe.to_csv(output_path, index=False)
    print(f"1D Predictions saved to {output_path}")


def save_predictions_2d(user_list, pred_list, user_id2token, item_id2token, output_path):
    """
    2차원 예측 결과를 저장하는 함수
    """
    result = []
    for user, items in zip(user_list, pred_list):
        for item in items:
            user_token = user_id2token[user]
            item_token = item_id2token[item]
            if user_token == '[PAD]' or item_token == '[PAD]':
                continue
            result.append((int(user_token), int(item_token)))

    dataframe = pd.DataFrame(result, columns=["user", "item"])
    dataframe.sort_values(by='user', inplace=True)
    dataframe.to_csv(output_path, index=False)
    print(f"2D Predictions saved to {output_path}")


def predict_full(dataset, model, matrix, user_id2token, item_id2token, device, output_path, is_two_dim=False):
    """
    Full 데이터셋을 사용하여 예측 (1차원 및 2차원 처리)
    """
    pred_list = None
    user_list = None

    for user_id in range(dataset.user_num):
        user_tensor = torch.tensor([user_id], device=device)
        interaction = {"user_id": user_tensor}

        score = model.full_sort_predict(interaction)
        rating_pred = score.cpu().data.numpy().copy()

        # 사용자가 상호작용한 아이템 점수 제거
        rating_pred[matrix[user_id].toarray() > 0] = -np.inf

        if is_two_dim:
            # 2차원 결과 처리
            ind = np.argpartition(rating_pred, -10)[:, -10:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
        else:
            # 1차원 결과 처리
            ind = np.argpartition(rating_pred, -10)[-10:]
            arr_ind = rating_pred[ind]
            arr_ind_argsort = np.argsort(arr_ind)[::-1]
            batch_pred_list = ind[arr_ind_argsort]

        batch_user_index = [user_id]

        # 예측값 저장
        if pred_list is None:
            pred_list = batch_pred_list
            user_list = batch_user_index
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            user_list = np.append(user_list, batch_user_index, axis=0)

    if is_two_dim:
        save_predictions_2d(user_list, pred_list, user_id2token, item_id2token, output_path)
    else:
        save_predictions_1d(user_list, pred_list, user_id2token, item_id2token, output_path)



def predict_test(test_data, model, matrix, user_id2token, item_id2token, device, output_path, is_two_dim=False):
    """
    Test 데이터셋을 사용하여 예측 (1차원 및 2차원 처리)
    """
    pred_list = None
    user_list = None

    for batch in test_data:
        interaction = batch[0].to(device)
        score = model.full_sort_predict(interaction)
        rating_pred = score.cpu().data.numpy().copy()

        user_ids = interaction['user_id'].cpu().numpy()

        if is_two_dim:
            # 2차원 결과 처리
            for i, user_id in enumerate(user_ids):
                interacted_indices = matrix[user_id].indices
                rating_pred[i, interacted_indices] = -np.inf

                ind = np.argpartition(rating_pred[i], -10)[-10:]
                arr_ind = rating_pred[i][ind]
                arr_ind_argsort = np.argsort(arr_ind)[::-1]
                batch_pred_list = ind[arr_ind_argsort]

                if pred_list is None:
                    pred_list = [batch_pred_list]
                    user_list = [user_id]
                else:
                    pred_list.append(batch_pred_list)
                    user_list.append(user_id)
        else:
            # 1차원 결과 처리
            for user_id in user_ids:
                interacted_indices = matrix[user_id].indices
                rating_pred[user_id][interacted_indices] = -np.inf

                ind = np.argpartition(rating_pred[user_id], -10)[-10:]
                arr_ind = rating_pred[user_id][ind]
                arr_ind_argsort = np.argsort(arr_ind)[::-1]
                batch_pred_list = ind[arr_ind_argsort]

                if pred_list is None:
                    pred_list = batch_pred_list
                    user_list = [user_id]
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    user_list = np.append(user_list, [user_id], axis=0)

    # 결과 저장
    if is_two_dim:
        save_predictions_2d(user_list, pred_list, user_id2token, item_id2token, output_path)
    else:
        save_predictions_1d(user_list, pred_list, user_id2token, item_id2token, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path of saved model checkpoint")
    parser.add_argument("--full", action="store_true", help="Use full dataset for prediction")
    parser.add_argument("--two_dim", action="store_true", help="Process 2D prediction output")
    parser.add_argument("--output", type=str, required=True, help="Path to save prediction results")

    args = parser.parse_args()
    
    base_checkpoint_dir = Path("/data/ephemeral/home/level2-recsys-movierecommendation-recsys-09-lv3/RecBole/saved")
    base_output_dir = Path("/data/ephemeral/home/level2-recsys-movierecommendation-recsys-09-lv3/saved/submit")
    
    # Resolve full path of the checkpoint
    checkpoint_path = base_checkpoint_dir / args.path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Resolve full path of the output
    output_path = base_output_dir / args.output
    if not base_output_dir.exists():
        base_output_dir.mkdir(parents=True, exist_ok=True) 
    
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=str(checkpoint_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    user_id2token = dataset.field2id_token["user_id"]
    item_id2token = dataset.field2id_token["item_id"]

    matrix = dataset.inter_matrix(form="csr")

    model.eval()

    if args.full:
        print("Running prediction on full dataset...")
        predict_full(dataset, model, matrix, user_id2token, item_id2token, device, str(output_path), is_two_dim=args.two_dim)
    else:
        print("Running prediction on test dataset...")
        predict_test(test_data, model, matrix, user_id2token, item_id2token, device, str(output_path), is_two_dim=args.two_dim)

    print("Inference completed!")

