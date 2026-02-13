import torch
import numpy as np
from sklearn import metrics

def performance_evaluation(labels: torch.Tensor, predictions: torch.Tensor):
    """
    Compute comprehensive classification performance metrics.
    Handles both binary and multiclass classification.

    Parameters
    ----------
    labels : torch.Tensor
        Ground truth class labels. Shape: (n_samples,)
        - Binary: integer values 0 or 1
        - Multiclass: integer values in range [0, num_classes-1]
    predictions : torch.Tensor
        Predicted values. Shape: (n_samples,)
        - Binary: probabilities in [0, 1], thresholded at 0.5
        - Multiclass: integer class labels (already argmax'ed)

    Returns
    -------
    dict
        Dictionary containing the following metrics:
        - 'Accuracy' : float
            Classification accuracy in percentage (0-100).
        - 'AUC' : float
            AUC-ROC for binary, macro recall for multiclass.
        - 'GM' : float
            Geometric mean of (per-class) recalls.
        - 'Sensitivity' : float
            Binary: TP/(TP+FN). Multiclass: macro recall.
        - 'Specificity' : float
            Binary: TN/(TN+FP). Multiclass: macro specificity.
        - 'CM' : list
            Confusion matrix.
    """
    # Convert tensors to NumPy arrays
    y = labels.detach().cpu().numpy().astype(int)
    pred_raw = predictions.detach().cpu().numpy()
    
    # Detect if binary (float predictions) or multiclass (integer predictions)
    is_binary = pred_raw.dtype in [np.float32, np.float64] and np.all((pred_raw >= 0) & (pred_raw <= 1))
    
    if is_binary:
        # Binary classification: threshold at 0.5
        pred = np.array([1 if x > 0.5 else 0 for x in pred_raw])
    else:
        # Multiclass: predictions are already class labels
        pred = pred_raw.astype(int)
    
    # Compute accuracy
    Accuracy = 100.0 * metrics.accuracy_score(y, pred)

    # Compute confusion matrix
    CM = metrics.confusion_matrix(y, pred)
    
    # Compute metrics based on classification type
    try:
        if is_binary:
            # Binary classification
            try:
                AUC = metrics.roc_auc_score(y, pred_raw)  # Use raw probabilities for AUC
            except ValueError:
                AUC = 0.0
            
            TN, FP, FN, TP = CM.ravel()
            sensitivity = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
            specificity = float(TN / (TN + FP)) if (TN + FP) > 0 else 0.0
            GM = float(np.sqrt(sensitivity * specificity))
            
        else:
            # Multiclass classification
            # Per-class recall (sensitivity for each class)
            per_class_recall = metrics.recall_score(y, pred, average=None, zero_division=0)
            
            # Macro-averaged recall (sensitivity)
            sensitivity = float(np.mean(per_class_recall))
            
            # Compute per-class specificity
            num_classes = CM.shape[0]
            per_class_specificity = []
            
            for i in range(num_classes):
                # True negatives: sum of all cells except row i and column i
                TN = np.sum(CM) - np.sum(CM[i, :]) - np.sum(CM[:, i]) + CM[i, i]
                # False positives: sum of column i except cell (i, i)
                FP = np.sum(CM[:, i]) - CM[i, i]
                
                spec = float(TN / (TN + FP)) if (TN + FP) > 0 else 0.0
                per_class_specificity.append(spec)
            
            # Macro-averaged specificity
            specificity = float(np.mean(per_class_specificity))
            
            # Geometric mean of per-class recalls
            epsilon = 1e-10
            GM = float(np.prod(per_class_recall + epsilon) ** (1.0 / len(per_class_recall)))
            
            # Use macro recall as AUC proxy for multiclass
            AUC = sensitivity
            
    except Exception as e:
        print("[WARNING] " + str(e))
        sensitivity = specificity = GM = AUC = 0.0

    return {
        'Accuracy': Accuracy,
        'AUC': AUC,
        'GM': GM,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'CM': CM.tolist()
    }