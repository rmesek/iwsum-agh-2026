import numpy as np


def reconstruction_errors(
    inputs: np.ndarray, reconstructions: np.ndarray
) -> np.ndarray:
    """Calculate reconstruction errors.

    :param inputs: Numpy array of input images
    :param reconstructions: Numpy array of reconstructions
    :return: Numpy array (1D) of reconstruction errors for each pair of input and its reconstruction
    """
    # 1) Błąd średniokwadratowy (MSE)
    return np.mean(np.square(inputs - reconstructions), axis=1)


def calc_threshold(reconstr_err_nominal: np.ndarray) -> float:
    """Calculate threshold for anomaly-detection

    :param reconstr_err_nominal: Numpy array of reconstruction errors for examples drawn from nominal class.
    :return: Anomaly-detection threshold
    """
    # 2) 3 odchylenia standardowe (99.7% rule)
    return np.mean(reconstr_err_nominal) + 3 * np.std(reconstr_err_nominal)


def detect(reconstr_err_all: np.ndarray, threshold: float) -> list:
    """Recognize anomalies using given reconstruction errors and threshold.

    :param reconstr_err_all: Numpy array of reconstruction errors.
    :param threshold: Anomaly-detection threshold
    :return: list of 0/1 values
    """
    # 3) Wyniki detekcji (1 - anomalia, 0 - normalny)
    return (reconstr_err_all > threshold).astype(int).tolist()
