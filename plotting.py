import numpy as np
from PathSegment import PathSegment, PrintSegment
import plotly.graph_objects as go
import plotly.colors
import plotly.figure_factory as ff
from typing import List
from binvox_rw import Voxels


def plot_voxelgrid(voxelgrid: Voxels):
    """Plots the complete voxelgrid, colored by occupancy."""
    indices = np.where(voxelgrid.data > 1)
    x_indices = indices[0]
    y_indices = indices[1]
    z_indices = indices[2]
    x, y, z = voxelgrid.get_position_for_index(x_indices, y_indices, z_indices)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    scatter_plot = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        marker=dict(
            color=np.divide(
                voxelgrid.data[x_indices, y_indices, z_indices], np.max(voxelgrid.data)
            ),
            colorscale=[
                (1 / np.max(voxelgrid.data), "red"),
                (0.5, "green"),
                (1, "blue"),
            ],
            opacity=1.0,
            size=5,
        ),
        opacity=0.8,
        mode="markers",
    )
    fig = go.Figure(data=scatter_plot)
    fig["layout"].update(scene=dict(aspectmode="data"))
    fig.show()


def parallel_plot_path_voxels(
    parameters: dict, path_segments: List[PathSegment], voxelgrid: Voxels
) -> None:
    """Same as plot_path_voxels, but parallel."""
    from processing import parallel_determine_segment_voxels

    # This re-determines the path voxels ẃithin more conservative bounds in order to have a nicer plot
    new_params = parameters.copy()
    new_params["bead"]["width"]["maximal"] = parameters["bead"]["width"]["nominal"]
    parallel_determine_segment_voxels(new_params, voxelgrid, path_segments)
    plot_path_voxels(parameters, path_segments, voxelgrid)


def get_continuous_color(colorscale, intermed):
    """
    # Taken from here: https://stackoverflow.com/a/64655638

    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]

    if intermed >= 1:
        return colorscale[-1][1] 

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break
    
    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )


def plot_paths(parameters: dict, path_segments: List[PrintSegment]) -> None:
    """Plot the processed paths using the local cross-sectional area / flowrate for both color and line width.
      This is similar to `plot_voxels` but it does not exhibit the phase shift due to only plotting the width. Note that the width is not to scale, however.

    Args:
        parameters (dict): parameters
        path_segments (List[PathSegment]): The process path segments
    """

    max_value = 0
    min_value = 1
    for segment in path_segments:
        magnitude = (
            segment.calculated_volume / segment.length
        )  # segment.slicer_volume if segment.slicer_volume > 0 else 0
        max_value = np.max((max_value, magnitude))
        min_value = np.min((min_value, magnitude))

    # Same colorscale as plot_path_voxels
    colorscale = [
        [0, f"rgb(0,0,{0xFF})"],  # blue
        [0.0357, f"rgb(0, {0x80}, 00)"],  # green
        [0.1783, f"rgb({0xFF},{0xFF}, 00)"],  # yellow
        [1, f"rgb({0xff}, 0, 0)"],  # red
    ]

    fig = go.Figure()
    for segment in path_segments:
        magnitude = segment.calculated_volume / segment.length
        fig.add_trace(
            go.Scatter(
                x=[segment.start_pose.x, segment.end_pose.x],
                y=[segment.start_pose.y, segment.end_pose.y],
                mode="lines",
                line=dict(
                    width=10 * magnitude,
                    color=get_continuous_color(
                        colorscale, (magnitude - min_value) / (max_value - min_value)
                    ),
                ),
            )
        )

    # add colorbar
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=colorscale,
                showscale=True,
                cmin=min_value,
                cmax=max_value,
                colorbar=dict(
                    thickness=20,
                    tickvals=np.arange(min_value, max_value, 0.5),
                    outlinewidth=0,
                ),
            ),
        )
    )
    

    camera = dict(up=dict(x=0, y=1, z=0), eye=dict(x=0, y=0, z=1))
    fig.update_layout(
        scene_camera=camera,
        showlegend=False,
        #yaxis_range=[0, 45],
        #xaxis_range=[0, 45],
        xaxis_title="X [mm]",
        yaxis_title="Y [mm]",
        autosize=False,
        height=700,
        width=700,
    )
    fig.show()


def plot_path_voxels(
    parameters: dict, path_segments: List[PrintSegment], voxelgrid: Voxels
) -> None:
    """This method determines voxels belonging to path segments and assigns colors according to the cross-sectional area.
    As voxels are colored in the order of the printing sequence and voxels often belong to multiple path segments,
    this results in a phase shift in the resulting image (as overlapping voxels are overpainted). The `plot_paths` method does not suffer from this drawback,
    but doesn't accurately reflect coverage.

    Args:
        parameters (dict):
        path_segments (List[PathSegment]): List of the processed path segments.
        voxelgrid (Voxels): Processed voxelgrid.
    """

    colorgrid = voxelgrid.clone()
    colorgrid.data = np.zeros(shape=voxelgrid.data.shape, dtype=np.float32)

    for segment in path_segments:
        i, j, k = segment.occupying_voxels_indices
        gridindices = voxelgrid.data[i, j, k]
        own_voxels = np.where(gridindices >= 2)

        magnitude = (
            segment.calculated_volume / segment.length
        )  # segment.slicer_volume if segment.slicer_volume > 0 else 0
        colorgrid.data[i[own_voxels], j[own_voxels], k[own_voxels]] = magnitude
        # colorgrid.data[i[own_voxels], j[own_voxels], k[own_voxels]] += magnitude / (gridindices[own_voxels] - 1)

    indices = np.where(colorgrid.data > 0)
    x_indices = indices[0][::10]
    y_indices = indices[1][::10]
    z_indices = indices[2][::10]
    x, y, z = voxelgrid.get_position_for_index(x_indices, y_indices, z_indices)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    # Colorscale that roughly matches height map
    # Colorscale must be [0, 1]
    colorscale = [
        (0, "blue"),
        (0.0357, "green"),
        (0.1783, "yellow"),
        (1, "red"),
    ]

    scatter_plot = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            color=colorgrid.data[x_indices, y_indices, z_indices],
            colorscale=colorscale,
            colorbar=dict(thickness=20),
            opacity=0.5,
            size=5,
        ),
    )
    fig = go.Figure(data=scatter_plot)
    scene = dict(
        xaxis_title="X [mm]", yaxis_title="Y [mm]", zaxis_title="", aspectmode="data"
    )
    camera = dict(up=dict(x=0, y=1, z=0), eye=dict(x=0, y=0, z=1))
    fig.update_layout(scene=scene, scene_camera=camera)
    fig.show()


def plot_histogram(path_segments: List[PrintSegment]) -> None:
    original_volume = []
    calculated_volume = []
    adjusted_volume = []
    length = []
    for seg in path_segments:
        original_volume.append(seg.slicer_volume)
        calculated_volume.append(seg.calculated_volume)
        adjusted_volume.append(seg.adjusted_volume)
        length.append(seg.length)

    go.Histogram(x=original_volume, name="Original (interpolated)")
    go.Histogram(x=calculated_volume, name="Calculated")

    ff_plot = ff.create_distplot(
        [original_volume, calculated_volume],
        group_labels=["Original/Slicer (interpolated)", "Calculated"],
    )
    ff_plot.show()
