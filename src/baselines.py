"""Baseline predictions share exactly the candidate-model origins."""
def predict_baselines(y_fit, baseline_frame):
    out = baseline_frame.copy()
    out['Training mean'] = float(y_fit.mean())
    return out
