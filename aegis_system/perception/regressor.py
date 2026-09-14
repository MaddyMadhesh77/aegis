from __future__ import annotations

from typing import Sequence

from sklearn.linear_model import LinearRegression


class TimeToCompromiseRegressor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> "TimeToCompromiseRegressor":
        self.model.fit(X, y)
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> Sequence[float]:
        return self.model.predict(X)
