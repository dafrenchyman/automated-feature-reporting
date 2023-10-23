from typing import List, Tuple

import pandas


class AutofrData:
    @staticmethod
    def determine_predictor_type(
        x: pandas.Series,
    ) -> str:
        """"""
        # Assume category or object dtypes are categorical
        if x.dtype.name in ["category", "object"]:
            return "categorical"
        # Heuristic from Stack Overflow Q#35826912
        # if float(x.nunique()) / x.count() < categorical_threshold:
        #     return "categorical"
        # Otherwise, we can assume its continuous
        return "continuous"

    @staticmethod
    def get_cont_cant_predictor_names(
        df: pandas.DataFrame, predictor_columns: List[str]
    ) -> Tuple[List[str], List[str]]:
        categorical_predictors = []
        continuous_predictors = []
        for column in predictor_columns:
            predictor_type = AutofrData.determine_predictor_type(x=df[column])
            if predictor_type == "continuous":
                continuous_predictors.append(column)
            else:
                categorical_predictors.append(column)

        return categorical_predictors, continuous_predictors
