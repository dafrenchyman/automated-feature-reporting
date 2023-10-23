from typing import Any, List, Optional, Tuple

import numpy
import pandas
from plotly import graph_objects as go
from sklearn.preprocessing import LabelEncoder

from autofr.utils.autofr_enums import AutofrEnums


class DiffMeanPlot:
    def __init__(
        self, num_bins: int = 10, tails_force_percentage: Optional[float] = None
    ):
        self.num_bins = num_bins
        self.tails_force_percentage = tails_force_percentage

    def _process_inputs(
        self,
        x,
        y,
    ):
        # Handle both continuous and categorical predictors
        data_type = x.dtype
        num_unique_values = len(numpy.unique(x))
        if data_type == "object" or num_unique_values < self.num_bins:
            le = LabelEncoder()
            x = le.fit(x).transform(x)
            num_bins = len(numpy.unique(x))
            hist, bins, bin_means, _ = self._generate_data(
                x, y, num_bins, data_type=AutofrEnums.DataType.CATEGORICAL
            )
            bin_centers = le.classes_.tolist()
        else:
            num_bins = self.num_bins
            hist, bins, bin_means, bin_centers = self._generate_data(
                x, y, num_bins, data_type=AutofrEnums.DataType.CONTINUOUS
            )
        return hist, bins, bin_means, bin_centers

    def _generate_data(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        num_bins: int,
        data_type: AutofrEnums.DataType,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, List[Any], List[Any]]:
        hist, bins = numpy.histogram(x, num_bins)

        if (
            self.tails_force_percentage is not None
            and data_type == AutofrEnums.DataType.CONTINUOUS
        ):
            x_ = x.copy() + 0
            lower_bins = []
            lower_hist = []
            upper_bins = []
            upper_hist = []
            num_replacements = 0
            percentile_values = numpy.percentile(
                x_,
                [
                    self.tails_force_percentage * 100,
                    100 - self.tails_force_percentage * 100,
                ],
            )
            lower_cut_off = percentile_values[0]
            upper_cut_off = percentile_values[1]
            if hist[0] / hist.sum() < self.tails_force_percentage:
                x_loc_in_bins = numpy.where(x_ > lower_cut_off)
                x_ = x_[x_loc_in_bins]
                lower_bins = [x.min()]
                lower_hist = [len(x_loc_in_bins[0])]
                num_replacements += 1
            if hist[len(hist) - 1] / hist.sum() < self.tails_force_percentage / 100:
                x_loc_in_bins = numpy.where(x_ < upper_cut_off)
                x_ = x_[x_loc_in_bins]
                upper_bins = [x.max()]
                upper_hist = [len(x_loc_in_bins[0])]
                num_replacements += 1

            hist, bins = numpy.histogram(x_, num_bins - num_replacements)
            hist = numpy.array(lower_hist + hist.tolist() + upper_hist)
            bins = numpy.array(lower_bins + bins.tolist() + upper_bins)

        bin_means = []
        bin_centers = []
        for idx in range(0, num_bins):
            lower_bin = bins[idx]
            upper_bin = bins[idx + 1]
            bin_center = (lower_bin + upper_bin) / 2
            bin_centers.append(bin_center)
            if idx + 1 < num_bins:
                x_loc_in_bins = numpy.where((x >= lower_bin) & (x < upper_bin))
            else:
                x_loc_in_bins = numpy.where((x >= lower_bin) & (x <= upper_bin))

            y_values_in_bins = y[x_loc_in_bins]
            bin_means.append(numpy.average(y_values_in_bins))
        return hist, bins, bin_means, bin_centers

    def generate_plot(
        self,
        df: pandas.DataFrame,
        response: str,
        predictor: str,
        filename: str = None,
        show_plot: bool = False,
        title: str = None,
    ) -> Tuple[str, float, float]:
        x = df[predictor].values
        y = df[response].values

        if title is None:
            title = (
                f"Binned Difference with Mean of Response ({response}) "
                + f"vs Predictor ({predictor}) Bin"
            )

        hist, bins, bin_means, bin_centers = self._process_inputs(x=x, y=y)

        pop_mean = numpy.average(y)

        data = [
            # The population Bar plot
            go.Bar(x=bin_centers, y=hist, yaxis="y2", name="Population", opacity=0.5),
            # The bin mean plot
            go.Scatter(x=bin_centers, y=bin_means, name="$\mu_{i}$"),  # noqa W605
            # The population average plot
            go.Scatter(
                x=bin_centers,
                y=[pop_mean] * len(hist),
                name="Population Mean ($\mu_{pop}$)",  # noqa W605
                mode="lines",
            ),
        ]

        layout = go.Layout(
            title=title,
            xaxis_title=f"Predictor Bin: {predictor}",
            yaxis_title=f"Response: {response}",
            yaxis2=dict(title="Population", overlaying="y", anchor="y3", side="right"),
        )

        fig = go.Figure(data=data, layout=layout)
        if show_plot:
            fig.show()

        if filename is not None:
            fig.write_html(
                file=filename,
                include_plotlyjs="cdn",
                include_mathjax="cdn",
            )

        html_str = fig.to_html(
            include_plotlyjs="cdn",
            include_mathjax="cdn",
        )

        mean_minus_pop_squared = numpy.power(numpy.array(bin_means) - pop_mean, 2)
        # Calculate unweighted
        unweighted = numpy.nansum(mean_minus_pop_squared) / len(hist)

        # Calculate weighted
        proportions = hist / numpy.sum(hist)
        weighted = numpy.nansum(numpy.multiply(mean_minus_pop_squared, proportions))

        return html_str, unweighted, weighted

    def generate_data(
        self,
        df: pandas.DataFrame,
        response: str,
        predictor: str,
    ) -> pandas.DataFrame:
        x = df[predictor].values
        y = df[response].values

        hist, bins, bin_means, bin_centers = self._process_inputs(x=x, y=y)

        pop_mean = numpy.average(y)
        n = len(y)

        # Extra calculations
        lower_bins = bins[0 : self.num_bins]  # noqa: E203
        upper_bins = bins[1 : (self.num_bins + 1)]  # noqa: E203
        pop_proportion = hist / n
        mean_squared_diff = numpy.power((bin_means - pop_mean), 2)
        mean_squared_diff_weighted = mean_squared_diff * pop_proportion

        # Outputs to help the slides
        df = pandas.DataFrame(
            list(
                zip(
                    lower_bins,
                    upper_bins,
                    bin_centers,
                    hist,
                    bin_means,
                    numpy.repeat(pop_mean, self.num_bins),
                    mean_squared_diff,
                    pop_proportion,
                    mean_squared_diff_weighted,
                )
            ),
            columns=[
                "LowerBin",
                "UpperBin",
                "BinCenters",
                "BinCount",
                "BinMeans",
                "PopulationMean",
                "MeanSquaredDiff",
                "PopulationProportion",
                "MeanSquaredDiffWeighted",
            ],
        )
        return df
