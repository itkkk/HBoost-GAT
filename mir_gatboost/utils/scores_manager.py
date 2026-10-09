import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
import torch
from torch.nn.functional import softmax


def classification_report_for_disease(y_true, y_pred, logits_matrix, stage_number=1,  output_dict=False):    
    """
    Compute classification metrics for each class.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    logits_matrix : array-like of shape (n_samples, n_classes)
        Raw model outputs (logits) before applying softmax.

    stage_number: int, default=1
        Specifies the model architecture. If set to a value greater than 1, the function expects the input logits_matrix to be softmax-normalized.

    output_dict : bool, default=False
        If True, returns the results as a pandas DataFrame, if False, prints the results instead.

    Returns
    -------
    scores : pandas.DataFrame
        DataFrame containing metrics for each class, rounded to three decimal places.
    """

    if not isinstance(logits_matrix, torch.Tensor):
        logits_matrix = torch.from_numpy(logits_matrix)

    if stage_number == 1:
        logits_matrix = softmax(logits_matrix, dim=1)


    scores_for_disease = pd.DataFrame({
        'Precision':     precision_score(y_true, y_pred, average=None, zero_division=0),
        'Recall':     recall_score(y_true, y_pred, average=None),
        'F1':           f1_score(y_true, y_pred, average=None),
        # 'AUC':          roc_auc_score(y_true, logits_matrix, average=None, multi_class='ovr'),
        # 'PRC':          average_precision_score(y_true, logits_matrix, average= None),
        'Support':      np.unique(y_true, return_counts=True)[1]
        }, index= np.unique(y_true)).round(3)
    
    if output_dict:
        return scores_for_disease
    
    print(scores_for_disease)





def classification_report_average(y_true, y_pred, logits_matrix, stage_number=1, output_dict = False):   
    """
    Compute macro and micro averaged classification metrics for multi-class models.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Predicted class labels.

    logits_matrix : array-like of shape (n_samples, n_classes)
        Raw model outputs (logits) before applying softmax.

    stage_number: int, default=1
        Specifies the model architecture. If set to a value greater than 1, the function expects the input logits_matrix to be softmax-normalized.

    output_dict : bool, default=False
        If True, returns the results as a pandas DataFrame, if False, prints the results instead.

    Returns
    -------
    averages : pandas.DataFrame
        DataFrame containing macro- and micro-averaged metrics rounded to three decimal places.
    """

    if not isinstance(logits_matrix, torch.Tensor):
        logits_matrix = torch.from_numpy(logits_matrix)

    if stage_number == 1:
        logits_matrix = softmax(logits_matrix, dim=1)

    metrics = {}

    # Precision
    metrics['pr_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    # metrics['pr_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)

    # Accuracies
    metrics['acc_macro'] = recall_score(y_true, y_pred, average='macro')
    # metrics['acc_micro'] = accuracy_score(y_true, y_pred)

    # F1 scores
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')

    # ROC AUC score
    # metrics['roc_auc_macro'] = roc_auc_score(y_true, logits_matrix, average='macro', multi_class='ovr')
    # metrics['roc_auc_micro'] = roc_auc_score(y_true, logits_matrix, average='micro', multi_class='ovr')

    # Precision Recall Curve score
    # metrics['prc_macro'] = average_precision_score(y_true, logits_matrix, average='macro')
    # metrics['prc_micro'] = average_precision_score(y_true, logits_matrix, average='micro')

    averages = pd.DataFrame({
        'Precision':     (metrics['pr_macro'], pd.NA),
        'Accuracy':     (metrics['acc_macro'], pd.NA),
        'F1':           (metrics['f1_macro'], metrics['f1_micro']),
        # 'AUC':          (metrics['roc_auc_macro'], metrics['roc_auc_micro']),
        # 'PRC':          (metrics['prc_macro'], metrics['prc_micro']),
        'Support':      (y_true.shape[0], y_true.shape[0])
    }, index=['Macro Avg', 'Micro Avg']).round(3)

    if output_dict:
        return averages
    
    print(averages)

 



def custom_agg(df):
    r"""Define how to aggregate each column of a dataframe.

    Args:
        df (DataFrame): The panda dataframe to be aggregate.
        """
    support_cols = [col for col in df.columns if 'support' in col.lower()]
    mean_cols = [col for col in df.columns if col not in support_cols]
    
    # Mean for metric columns
    mean_part = df[mean_cols].groupby(level=1).mean()
    
    # Sum for support columns
    sum_part = df[support_cols].groupby(level=1).sum()
    
    # Combine the results
    return pd.concat([mean_part, sum_part], axis=1).round(3)