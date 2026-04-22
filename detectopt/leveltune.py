import numpy as np
from sklearn.metrics import roc_curve, auc

class Struct:
    """Lightweight class for dot-notation, matching the rest of the framework."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def leveltune(obsval):
    """
    Optimizes the detection threshold by computing the Receiver Operating Characteristic (ROC).
    Calculates False Positive Rate (FPR), Probability of Missed Detection (PMD), and AUC.
    
    Equivalent to leveltune.m, ROCArea.m, and ROCcurveaveraging.m combined.
    """
    allevents = []
    exceedscore = []
    exceed = Struct(exceedscore=[])

    # 1. Aggregate all batches of data
    for obs in obsval:
        # Extract ground truth events (boolean/ints)
        events = np.asarray(obs.event).flatten()
        allevents.extend(events)
        
        # Calculate the anomaly score (absolute value of the residual)
        score = np.abs(np.asarray(obs.data).flatten())
        exceedscore.extend(score)
        exceed.exceedscore.append(score)

    # 2. Handle empty edge cases
    if len(allevents) == 0:
        # Return fallback zeros if no data exists
        dummy_stats = Struct(fprate=np.array([0]), pmd=np.array([0]), thresh=np.array([0]), rocarea=0.0)
        exceed.avg_stats = dummy_stats
        return 0.0, exceed

    allevents = np.array(allevents, dtype=int)
    exceedscore = np.array(exceedscore)

    # 3. Calculate ROC and AUC
    # Check if we have at least one nominal and one anomalous point to draw a curve
    if len(np.unique(allevents)) < 2:
        print("Warning: Only one class present in ground truth for this prediction horizon. Cannot draw ROC.")
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        thresh = np.array([np.max(exceedscore) if len(exceedscore) > 0 else 0.0, 0.0])
        auc_val = 0.5
    else:
        # scikit-learn efficiently calculates the rates for every relevant threshold
        fpr, tpr, thresh = roc_curve(allevents, exceedscore)
        auc_val = auc(fpr, tpr)

    # 4. Calculate PMD (Probability of Missed Detection)
    # PMD is mathematically the False Negative Rate (1 - True Positive Rate)
    pmd = 1.0 - tpr

    # 5. Package results for the orchestrator
    stats = Struct(
        fprate=fpr,
        pmd=pmd,
        thresh=thresh,
        rocarea=auc_val
    )

    # testoptloop expects the data inside rocdata.avg_stats
    exceed.avg_stats = stats

    return auc_val, exceed