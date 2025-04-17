import pandas as pd
import numpy as np

import datasets 
from sklearn.metrics import mean_squared_error

from galaxy_datasets.shared import label_metadata


def regression_metrics(test_preds_loc):

    df = load_test_preds(test_preds_loc)

    question_answer_pairs = label_metadata.gz_evo_v1_public_pairs

    df = blank_uncertain_votes(df, question_answer_pairs)

    results_df = get_overall_metrics(df, question_answer_pairs)
    return results_df

def blank_uncertain_votes(df, question_answer_pairs, min_votes=20):
    for question, answers in question_answer_pairs.items():
        total = df[f'{question}_total-votes']
        for answer in answers:
            fractions = df[f'{question}{answer}_fraction_vol'].values
            fractions = np.where(total < min_votes, np.nan, fractions)
            # print(question, answer, np.isnan(fractions).mean())
            # print(len(fractions), len(df))
            df[f'{question}{answer}_fraction_vol_masked'] = fractions
    return df


def get_overall_metrics(df, question_answer_pairs):

    results = []
    for question, answers in question_answer_pairs.items():
        for answer in answers:
            y_true = df[f'{question}{answer}_fraction_vol_masked']
            y_pred = df[f'{question}{answer}_fraction_ml']
            safe_mask = ~np.isnan(y_true)
            y_true = y_true[safe_mask]
            y_pred = y_pred[safe_mask]
            mse = mean_squared_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            results.append({
                'question_answer': f'{question}_{answer}',
                'mse': mse,
                'rmse': rmse,
                'nan_fraction': 1-safe_mask.mean()
            })

    results_df = pd.DataFrame(results)
    return results_df




def load_test_preds(test_preds_loc):
    df = pd.read_csv(test_preds_loc)
    test_dataset = datasets.load_dataset("mwalmsley/gz_evo", name='default', split='test')
    test_dataset = test_dataset.remove_columns("image")
    test_dataset.set_format('pandas')
    df = pd.merge(test_dataset.data.to_pandas(), df, on='id_str', how='inner', validate='1:1', suffixes=('_vol', '_ml'))
    return df

if __name__ == "__main__":

    regression_metrics()