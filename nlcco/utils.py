import json
from matplotlib import pyplot as plt
import numpy as np


def transpose_list(l):
    if isinstance(l[0], list):
        return list(map(list, zip(*l)))
    return l


def add_data_to_json(x, f, problem_name, filename):
    """
    Append \"xopt\" and \"fmin\" to json data for problem name
    """
    with open(filename, "r") as json_file:
        data = json.load(json_file)
    while True:
        try:
            tmp = data[problem_name]
        except KeyError:
            data[problem_name] = {}
        break
    fmin = tmp.get("fmin")
    if fmin is not None:
        if fmin < f:
            print("Already better fmin")
            return False
    tmp["xopt"] = list(x)
    tmp["fmin"] = f
    with open(filename, "w") as json_file:
        json.dump(data, json_file)
    return True


def bar_plot(ax, data, top_text=None, colors=None,
             total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if top_text is None:
        top_text = {}

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y,
                         width=bar_width * single_width,
                         color=colors[i % len(colors)],
                         zorder=3)

        bars.append(bar[0])

    for i, ((name, values), (_, n_nans)) in enumerate(zip(data.items(), top_text.items())):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        for x, (y, n_nan) in enumerate(zip(values, n_nans)):
            if n_nan > 0:
                print(x)
                ax.text(x + x_offset - .1, (1 + .05) * y, f"{100*n_nan:.0f}%",
                        color="red", bbox=dict(facecolor='white', alpha=1))

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(), fancybox=True, shadow=True)


def single_target_ecdfs(ax, data,
                        legend=True, colortable=None,
                        styles=None, markers=None, **kwargs):
    """
    Args:
        ax: axe to plot
        data: dictionary of algo, runtimes pairs
        colortable: a dictionary mapping algo to a color in the CSS palette
        https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    """

    for i, (algo, runtimes) in enumerate(data.items()):
        rt = np.array(runtimes, copy=True, dtype=float)
        rt.sort()
        nt = len(rt)
        if colortable is not None:
            kwargs['color'] = colortable[algo]
        if styles is not None:
            kwargs['linestyle'] = styles.get(algo, "-")
        if markers is not None:
            kwargs['marker'] = markers.get(algo, None)
            if not kwargs.get('markevery'):
                kwargs['markevery'] = list(range(1, nt, 10))

        # kwargs.setdefault('linewidth', 3)
        if nt == 0:
            # shadow plot if default colors
            raise ValueError("No data for algorithm %s", algo)
            ax.step([], [], label=algo, **kwargs)
        else:
            ax.step(rt, np.linspace(1 / len(rt), 1, len(rt)),
                    where='post', label=algo, **kwargs)

    for i, (algo, runtimes) in enumerate(data.items()):
        rt = np.array(runtimes, copy=True, dtype=float)
        rt.sort()
        if colortable is not None:
            kwargs['color'] = colortable[algo]
        if styles is not None:
            kwargs['linestyle'] = styles.get(algo, "-")
        # kwargs.setdefault('linewidth', 3)
        if len(rt) == 0:
            # shadow plot if default colors
            ax.step([], [], label=algo, **kwargs)
        else:
            ax.plot(2 * [rt[0]], [0, 1 / len(rt)], **kwargs)
            ax.plot([np.nanmax(rt), ax.get_xlim()[1]],
                    2 * [sum(np.isfinite(rt)) / len(rt)], **kwargs)

            # Draw legend if we need
        if legend:
            ax.legend(fancybox=True, shadow=True)


if __name__ == "__main__":
    # Usage example:
    data = {
        "a": [1, 2, 3, 2, 1],
        "b": [2, 3, 4, 3, 1],
        "c": [3, 2, 1, 4, 2],
        "d": [5, 9, 2, 1, 8],
        "e": [1, 3, 2, 2, 3],
        "f": [4, 3, 1, 1, 4],
    }

    fig, axes = plt.subplots(1, 2)
    bar_plot(axes[0], data, total_width=.8, single_width=.9)
    single_target_ecdfs(axes[1], data,)
    plt.show()
