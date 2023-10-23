import random
from typing import List, Tuple

import pandas
import seaborn
from sklearn import datasets


def determine_response_type(x: pandas.Series) -> str:
    """"""
    # Assume category or object dtypes are categorical
    if x.dtype.name in ["category", "object"]:
        return "categorical"
    # Heuristic from Stack Overflow Q#35826912
    if x.nunique() == 2:
        return "categorical"
    # Otherwise, we can assume its continuous
    return "continuous"


def determine_predictor_type(
    x: pandas.Series,
    categorical_threshold: float = 0.05,
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


def get_cont_cant_predictor_names(df: pandas.DataFrame, predictor_columns: List[str]):
    categorical_predictors = []
    continuous_predictors = []
    for column in predictor_columns:
        predictor_type = determine_predictor_type(x=df[column])
        if predictor_type == "continuous":
            continuous_predictors.append(column)
        else:
            categorical_predictors.append(column)

    return categorical_predictors, continuous_predictors


class TestDatasets:
    def __init__(self):
        self.seaborn_data_sets = ["mpg", "tips", "titanic"]
        self.sklearn_data_sets = ["diabetes", "breast_cancer"]
        self.csv_data_sets = ["telco"]
        self.other_data_sets = ["iris"]  # , "iris_fixed"]
        self.all_data_sets = (
            self.seaborn_data_sets
            + self.sklearn_data_sets
            + self.csv_data_sets
            + self.other_data_sets
        )

    TITANIC_PREDICTORS = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "embarked",
        "parch",
        "fare",
        "who",
        "adult_male",
        "deck",
        "embark_town",
        "alone",
        "class",
    ]

    def get_all_available_datasets(self) -> List[str]:
        return self.all_data_sets

    def get_test_data_set(
        self, data_set_name: str = None
    ) -> Tuple[pandas.DataFrame, List[str], List[str]]:
        """Function to load a few test data sets

        :param:
        data_set_name : string, optional
            Data set to load

        :return:
        data_set : :class:`pandas.DataFrame`
            Tabular data, possibly with some preprocessing applied.
        predictors :list[str]
            List of predictor variables
        response: :list[str]
            Response variable
        """

        if data_set_name is None:
            data_set_name = random.choice(self.all_data_sets)
        else:
            if data_set_name not in self.all_data_sets:
                raise Exception(f"Data set choice not valid: {data_set_name}")

        predictors = []
        responses = []

        if data_set_name in self.seaborn_data_sets:
            if data_set_name == "mpg":
                data_set: pandas.DataFrame = (
                    seaborn.load_dataset(name="mpg").dropna().reset_index()
                )
                predictors = [
                    "cylinders",
                    "displacement",
                    "horsepower",
                    "weight",
                    "acceleration",
                    "origin",
                ]
                responses = ["mpg"]
            elif data_set_name == "tips":
                data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
                predictors = [
                    "total_bill",
                    "sex",
                    "smoker",
                    "day",
                    "time",
                    "size",
                ]
                responses = ["tip"]
            elif data_set_name in ["titanic", "titanic_2"]:
                data_set = seaborn.load_dataset(name="titanic").dropna()
                data_set["alone"] = data_set["alone"].astype(str)
                data_set["class"] = data_set["class"].astype(str)
                data_set["deck"] = data_set["deck"].astype(str)
                data_set["pclass"] = data_set["pclass"].astype(str)
                predictors = self.TITANIC_PREDICTORS
                if data_set_name == "titanic":
                    responses = ["survived"]
                elif data_set_name == "titanic_2":
                    responses = ["alive"]
            else:
                raise ValueError(f"Invalid Dataset {data_set_name}")
        elif data_set_name in self.sklearn_data_sets:
            if data_set_name == "diabetes":
                data = datasets.load_diabetes()
                data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            elif data_set_name == "breast_cancer":
                data = datasets.load_breast_cancer()
                data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            data_set["target"] = data.target
            predictors = data.feature_names
            responses = ["target"]
        elif data_set_name == "telco":
            data_set = pandas.read_csv(
                "https://teaching.mrsharky.com/data/datasets_13996_18858_WA_Fn-UseC_-Telco-Customer-Churn.csv"
            )

            # Drop useless column
            data_set.drop("customerID", axis=1, inplace=True)

            # Fix total charges to numeric
            data_set["TotalCharges"] = pandas.to_numeric(
                data_set["TotalCharges"], errors="coerce"
            )
            data_set.dropna(inplace=True)
            data_set.reset_index(inplace=True, drop=True)

            # Make a column to see how good cat / cont correlations are
            random.seed(42)
            data_set["SeniorCitizenContinuous"] = [
                random.uniform(0.0, 0.5) if v == 0 else random.uniform(0.5, 1.0)
                for v in data_set["SeniorCitizen"].values
            ]

            data_set["SeniorCitizen"] = data_set["SeniorCitizen"].astype(str)
            data_set["Churn"] = [
                1 if v == "Yes" else 0 for v in data_set["Churn"].values
            ]

            responses = ["Churn"]
            predictors = [i for i in list(data_set.columns) if i not in responses]

            # sort them
            predictors.sort()

        elif data_set_name in self.other_data_sets:
            if data_set_name == "iris":
                data_set = pandas.read_csv(
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                    names=[
                        "sepal length",
                        "sepal width",
                        "petal length",
                        "petal width",
                        "class",
                    ],
                )
                original_response = "class"
                data_set = data_set.dropna()
                extra_responses = list(data_set[original_response].unique())
                for response in extra_responses:
                    data_set[response] = (data_set[original_response] == response) + 0

                # Generate response and predictors
                responses = extra_responses
                predictors = [
                    i
                    for i in data_set.columns
                    if i not in responses + [original_response]
                ]
        else:
            raise ValueError(f"Invalid Dataset {data_set_name}")

        # Change category dtype to string
        for predictor in predictors:
            if data_set[predictor].dtype in ["category"]:
                data_set[predictor] = data_set[predictor].astype(str)

        print(f"Data set selected: {data_set_name}")
        data_set.reset_index(drop=True, inplace=True)
        return data_set, predictors, responses


if __name__ == "__main__":
    test_datasets = TestDatasets()
    test_datasets.get_test_data_set("iris")
    for test in test_datasets.get_all_available_datasets():
        df, predictors, responses = test_datasets.get_test_data_set(data_set_name=test)
