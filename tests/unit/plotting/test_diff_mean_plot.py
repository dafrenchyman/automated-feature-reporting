import unittest

from autofr.plotting.diff_mean_plot import DiffMeanPlot
from tests.helpers.data_helpers import TestDatasets


class test_diff_mean_plot(unittest.TestCase):
    def test_diff_mean_plot(self):
        test_datasets = TestDatasets()
        for test in test_datasets.get_all_available_datasets():
            df, predictors, responses = test_datasets.get_test_data_set(
                data_set_name=test
            )

            for response in responses:
                for predictor in predictors:
                    diff_mean_plot = DiffMeanPlot(
                        num_bins=10,
                        tails_force_percentage=0.05,
                    )
                    _html, _unweighted, _weighted = diff_mean_plot.generate_plot(
                        df=df,
                        response=response,
                        predictor=predictor,
                    )
                    _ = diff_mean_plot.generate_data(
                        df=df,
                        response=response,
                        predictor=predictor,
                    )

        return
