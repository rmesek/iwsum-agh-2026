import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> np.ndarray:
    """
    1) Założyć typ rozkładu statystycznego danych uczących
    2) Wyestymować jego parametry na podstawie danych uczących
    3) Określić próg detekcji anomalii w odniesieniu do obliczonych 
    parametrów rozkładu statystycznego
    4) Obliczyć wyniki detekcji dla danych testowych
    """
    # 1) Zakładamy rozkład normalny

    # 2) Estamacja parametrów rozkładu
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)

    # 3) Próg ustawiamy na 3 odchylenia standardowe (99.7% rule)
    threshold = 3 * std

    # 4) Wyniki detekcji (1 - anomalia, 0 - normalny)
    predictions = (np.abs(test_data - mean) > threshold).astype(int)
    
    return predictions