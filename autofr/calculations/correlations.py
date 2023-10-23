from typing import List

import numpy
import pandas
from pycorrcat import pycorrcat

from autofr.utils.autofr_data import AutofrData


class Correlations:
    def __init__(self, df: pandas.DataFrame, predictor_columns: List[str]):
        self.df = df
        self.predictor_columns = predictor_columns
        self.categoricals, self.continuous = AutofrData.get_cont_cant_predictor_names(
            df=df, predictor_columns=predictor_columns
        )

        # categorical vs categorical
        cat_cat_rows_list = []
        cat_cat_corr_matrix = numpy.zeros(
            shape=(len(self.categoricals), len(self.categoricals))
        )
        for cat_1_idx, cat_1_name in enumerate(self.categoricals):
            for cat_2_idx, cat_2_name in enumerate(self.categoricals):
                cat_cat_correlation = Correlations.cat_cat_correlation(
                    df=df,
                    cat_column_1=cat_1_name,
                    cat_column_2=cat_2_name,
                )
                cat_cat_corr_matrix[cat_1_idx][cat_2_idx] = cat_cat_correlation
                result = {
                    "cat_1_name": cat_1_name,
                    "cat_2_name": cat_2_name,
                    "corr": cat_cat_correlation,
                }
                cat_cat_rows_list.append(result)

        self.df_cat_cat_corr = pandas.DataFrame(cat_cat_rows_list)

        # categorical vs continuous
        cat_cont_rows_list = []
        cat_cont_corr_matrix = numpy.zeros(
            shape=(len(self.categoricals), len(self.continuous))
        )
        for idx_1, cat_column_name in enumerate(self.categoricals):
            for idx_2, cont_column_name in enumerate(self.continuous):
                correlation = Correlations.cat_cont_correlation_ratio(
                    categories=df[cat_column_name].to_numpy(),
                    values=df[cont_column_name].to_numpy(),
                )
                cat_cont_corr_matrix[idx_1][idx_2] = correlation

                # Make the DataFrame Table
                result = {
                    "cat": cat_column_name,
                    "cont": cont_column_name,
                    "corr": correlation,
                }
                cat_cont_rows_list.append(result)

        self.df_cat_cont_corr = pandas.DataFrame(cat_cont_rows_list)

    @staticmethod
    def cat_cat_correlation(
        df: pandas.DataFrame,
        cat_column_1: str,
        cat_column_2: str,
        bias_correction: bool = True,
        tschuprow: bool = False,
    ) -> float:
        x = df[cat_column_1].values
        y = df[cat_column_2].values
        correlation = pycorrcat.corr(
            x, y, bias_correction=bias_correction, Tschuprow=tschuprow
        )

        return correlation

    @staticmethod
    def cat_cont_correlation_ratio(
        categories: numpy.ndarray[str], values: numpy.ndarray[float]
    ) -> float:
        """
        Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
        SOURCE:
        1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
        :param categories: Numpy array of categories
        :param values: Numpy array of values
        :return: correlation
        """
        f_cat, _ = pandas.factorize(categories)
        cat_num = numpy.max(f_cat) + 1
        y_avg_array = numpy.zeros(cat_num)
        n_array = numpy.zeros(cat_num)
        for i in range(0, cat_num):
            cat_measures = values[numpy.argwhere(f_cat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = numpy.average(cat_measures)
        y_total_avg = numpy.sum(numpy.multiply(y_avg_array, n_array)) / numpy.sum(
            n_array
        )
        numerator = numpy.sum(
            numpy.multiply(
                n_array, numpy.power(numpy.subtract(y_avg_array, y_total_avg), 2)
            )
        )
        denominator = numpy.sum(numpy.power(numpy.subtract(values, y_total_avg), 2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = numpy.sqrt(numerator / denominator)
        return eta
