import numpy as np
from sklearn import metrics
from collections import Counter, defaultdict
def compute_AUPRC_and_AUROC_scores(y_true, y_pred, sg_list, sg_dict):
    
    """
    Compute AUROC and AUPRC scores for each sgRNA and aggregate them.

    For each unique sgRNA in the dataset:
    1. Select its subset of true labels and predicted values.
    2. Compute AUROC (ROC AUC score).
    3. Compute AUPRC (area under the precision-recall curve).
    4. Store per-sgRNA scores along with sgRNA identifier.

    Finally, the function returns both per-sgRNA metrics and the overall mean AUROC/AUPRC.

    """
    sg_type = set(sg_list)
    sg_list = np.array(sg_list)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result_dict = {
        'sg_type':[],
        'sg_roc_auc':[],
        'sg_roc_prc':[],
    }

    for sg in sg_type:
        mask = sg_list == sg.item()
        labels = y_true[mask]
        predicted_values = y_pred[mask]
        sg_label_dict = Counter(labels)
        if sg_label_dict[1] == 0 or sg_label_dict[0] == 0:
            continue
        roc_auc = metrics.roc_auc_score(labels, predicted_values)
        prc, rec, _ = metrics.precision_recall_curve(labels, predicted_values)
        prc[rec == 0] = 1.0
        prc_auc = metrics.auc(rec, prc)
        result_dict["sg_type"].append(sg_dict[sg])
        result_dict["sg_roc_auc"].append(roc_auc)
        result_dict["sg_roc_prc"].append(prc_auc)
    
    result_dict['roc_auc'] = np.mean(result_dict['sg_roc_auc'])
    result_dict['roc_prc'] = np.mean(result_dict['sg_roc_prc'])

    return result_dict