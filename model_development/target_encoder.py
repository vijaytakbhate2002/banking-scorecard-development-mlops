import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold


class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    """Target encoder that supports train-only OOF encoding and inference."""

    def __init__(
        self,
        cols=None,
        stats=None,
        cv=5,
        smooth=10.0,
        is_classification=False,
        random_state=42,
    ):
        self.cols = cols
        self.stats = stats if stats is not None else ["mean", "std", "count", "skew"]
        self.cv = cv
        self.smooth = smooth
        self.is_classification = is_classification
        self.random_state = random_state

        self.global_target_mean_ = None
        self.encoding_maps_ = {}
        self.encoded_feature_names_ = []
        self.cols_ = []

    def _compute_stats(self, df: pd.DataFrame, target: pd.Series, col: str) -> pd.DataFrame:
        temp_df = pd.DataFrame({"cat": df[col], "target": target})
        grouped = temp_df.groupby("cat")["target"]
        stats_df = pd.DataFrame(index=grouped.indices.keys())
        global_mean = target.mean()

        for stat in self.stats:
            if stat == "mean":
                cat_count = grouped.count()
                cat_mean = grouped.mean()
                stats_df["mean"] = (cat_count * cat_mean + self.smooth * global_mean) / (
                    cat_count + self.smooth
                )
            elif stat == "std":
                stats_df["std"] = grouped.std().fillna(0)
            elif stat == "count":
                stats_df["count"] = grouped.count()
            elif stat == "skew":
                stats_df["skew"] = grouped.skew().fillna(0)

        stats_df.columns = [f"{col}_te_{stat}" for stat in stats_df.columns]
        return stats_df

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        y = pd.Series(y).reset_index(drop=True)

        if self.cols is None:
            self.cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            self.cols_ = list(self.cols)

        self.global_target_mean_ = y.mean()
        self.encoding_maps_ = {}
        self.encoded_feature_names_ = []

        for col in self.cols_:
            stats_df = self._compute_stats(X, y, col)
            self.encoding_maps_[col] = stats_df
            self.encoded_feature_names_.extend(stats_df.columns.tolist())

        return self

    def fit_transform_oof(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X_df = X.copy().reset_index(drop=True)
        y_ser = pd.Series(y).reset_index(drop=True)

        if self.cols is None:
            self.cols_ = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            self.cols_ = list(self.cols)

        self.fit(X_df, y_ser)

        if self.is_classification:
            kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        oof_features = pd.DataFrame(index=X_df.index)

        for col in self.cols_:
            col_oof = pd.DataFrame(index=X_df.index)
            for train_idx, val_idx in kf.split(X_df, y_ser):
                X_tr, y_tr = X_df.iloc[train_idx], y_ser.iloc[train_idx]
                X_va = X_df.iloc[val_idx]

                fold_stats = self._compute_stats(X_tr, y_tr, col)
                va_mapped = X_va[[col]].merge(fold_stats, left_on=col, right_index=True, how="left")
                va_mapped = va_mapped.drop(columns=[col])

                for stat in self.stats:
                    stat_col = f"{col}_te_{stat}"
                    if stat == "mean":
                        va_mapped[stat_col] = va_mapped[stat_col].fillna(self.global_target_mean_)
                    else:
                        va_mapped[stat_col] = va_mapped[stat_col].fillna(0)

                col_oof.loc[val_idx, va_mapped.columns] = va_mapped

            oof_features = pd.concat([oof_features, col_oof], axis=1)

        non_cat_cols = [c for c in X_df.columns if c not in self.cols_]
        result = pd.concat([X_df[non_cat_cols], oof_features], axis=1)
        result.index = X.index
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        transformed_dfs = []

        for col in self.cols_:
            stats_df = self.encoding_maps_[col]
            mapped = X_df[[col]].merge(stats_df, left_on=col, right_index=True, how="left")
            mapped = mapped.drop(columns=[col])

            for stat in self.stats:
                stat_col = f"{col}_te_{stat}"
                if stat == "mean":
                    mapped[stat_col] = mapped[stat_col].fillna(self.global_target_mean_)
                else:
                    mapped[stat_col] = mapped[stat_col].fillna(0)

            transformed_dfs.append(mapped)

        non_cat_cols = [c for c in X_df.columns if c not in self.cols_]
        result = pd.concat([X_df[non_cat_cols]] + transformed_dfs, axis=1)
        return result
