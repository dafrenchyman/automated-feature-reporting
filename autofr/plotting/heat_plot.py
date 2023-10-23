from typing import List

import numpy
from plotly import figure_factory


def generate_heat_map(
    matrix_data: numpy.ndarray,
    dim_1_labels: List[str],
    dim_2_labels: List[str],
    chart_title: str,
    filename: str = None,
    show_fig: bool = False,
) -> str:
    # If for some reason we receive an empty correlation matrix.
    # Process nothing and return an empty string for the HTML
    if matrix_data.shape[0] == 0 or matrix_data.shape[1] == 0:
        return ""

    # Generate the annotated text
    z_text = [[str(round(y, 6)) for y in x] for x in matrix_data]

    # set up figure
    fig = figure_factory.create_annotated_heatmap(
        z=matrix_data,
        x=dim_2_labels,
        y=dim_1_labels,
        annotation_text=z_text,
        colorscale="RdBu",
        font_colors=["black", "black"],
        zmid=0,
    )

    # add title
    fig.update_layout(
        title_text=f"<b>{chart_title}</b>",
        xaxis=dict(title="<b>Category 1</b>"),
        yaxis=dict(title="<b>Category 2</b>"),
    )

    # add custom xaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Correlation",
            xref="paper",
            yref="paper",
        )
    )

    # add custom y-axis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=-0.35,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )

    # adjust margins to make room for y-axis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add color bar
    fig["data"][0]["showscale"] = True
    if show_fig:
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

    return html_str
