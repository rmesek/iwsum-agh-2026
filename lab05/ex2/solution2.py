import numpy as np
from sklearn.covariance import MinCovDet


def detect(train_data: np.ndarray, test_data: np.ndarray) -> np.ndarray:
    """
    1) Wyestymować macierz kowariancji rozkładu statystycznego danych uczących, 
    np. przy pomocy pakietu `sklearn.covariance.MinCovDet`
    2) Obliczyć maksymalną odległość przykładów uczących od wartości oczekiwanej 
    (średniej) wg metryki Mahalanobisa - patrz: `MinCovDet.mahalanobis()`
    3) Obliczyć odległości Mahalanobisa dla przykładów testowych i na tej 
    podstawie określić czy są artefaktami.
    """
    # 1) Estymacja macierzy kowariancji
    mcd = MinCovDet().fit(train_data)

    # 2) Maksymalna odległość Mahalanobisa dla danych uczących
    max_distance = np.max(mcd.mahalanobis(train_data))

    # 3) Odległości Mahalanobisa dla danych testowych
    test_distances = mcd.mahalanobis(test_data)

    # Przykłady, których odległość jest większa niż maksymalna odległość uczących, są artefaktami
    predictions = (test_distances > max_distance).astype(int)
    
    return predictions