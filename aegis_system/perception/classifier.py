from __future__ import annotations

from typing import Iterable, List, Sequence

from sklearn.ensemble import RandomForestClassifier


class AttackClassifier:
    def __init__(self, n_estimators: int = 20, random_state: int = 42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[str]) -> "AttackClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> List[str]:
        if not hasattr(self.model, "classes_"):
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X).tolist()
