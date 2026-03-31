from sklearn import svm
from utils import binary2neg_boolean
import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> np.ndarray:
    """
    1) Przeprowadzić detekcję anomalii na zbiorze testowym wykorzystując
    odległość Mahalanobisa, jak w zadaniu 2 (użyć funkcji ponownie).
    2) Wyniki porównać z wynikami na zbiorze testowym uzyskiwanymi przez
    algorytm OneClass-SVM: `sklearn.svm.OneClassSVM` Wykorzystać odpowiedni kernel.
    3) Skomentować wyniki - jakie skłonności mają te algorytmy? tj. w jakich
    sytuacjach są odpowiednie?
    4) Jaki wpływ na wyniki ma manipulacja parametrami algorytmu OC-SVM?
    """
    # 2) Detekcja One-Class SVM
    oc_svm = svm.OneClassSVM(kernel="rbf", nu=0.1).fit(train_data)
    predictions_ocsvm = oc_svm.predict(test_data)
    predictions_ocsvm = binary2neg_boolean(predictions_ocsvm)

    return predictions_ocsvm
