from xgboost import XGBRegressor

class XGBoostModel:
    def __init__(
        self,
        n_estimators=300,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.7,
        colsample_bytree=0.8,
        random_state=57,
        n_jobs=-1,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)