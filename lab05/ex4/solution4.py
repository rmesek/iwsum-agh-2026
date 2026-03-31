from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from utils import binary2neg_boolean
import numpy as np

SEED = 1


def detect_cov(data: np.ndarray, outliers_fraction: float) -> list:
    predictions_ee = EllipticEnvelope(
        contamination=outliers_fraction, random_state=SEED
    ).fit_predict(data)
    return binary2neg_boolean(predictions_ee)


def detect_ocsvm(data: np.ndarray, outliers_fraction: float) -> list:
    predictions_ocsvm = svm.OneClassSVM(kernel="rbf", nu=outliers_fraction).fit_predict(
        data
    )
    return binary2neg_boolean(predictions_ocsvm)


def detect_iforest(data: np.ndarray, outliers_fraction: float) -> list:
    predictions_iforest = IsolationForest(
        contamination=outliers_fraction, random_state=SEED
    ).fit_predict(data)
    return binary2neg_boolean(predictions_iforest)


def detect_lof(data: np.ndarray, outliers_fraction: float) -> list:
    predictions_lof = LocalOutlierFactor(
        contamination=outliers_fraction, n_neighbors=400
    ).fit_predict(data)
    return binary2neg_boolean(predictions_lof)
