import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from . import suntimes

COLOR_AXES = "#999999"
COLOR_FACE = "#999999"

DEFAULT_FIGRATIO = 1.618
DEFAULT_FIGWIDTH = 8
DEFAULT_FIGHEIGHT = DEFAULT_FIGWIDTH / DEFAULT_FIGRATIO
DEFAULT_FIGSIZE = (DEFAULT_FIGWIDTH, DEFAULT_FIGHEIGHT)

plt.rcParams["figure.figsize"] = DEFAULT_FIGSIZE
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 9
# plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.02


def _datetime_from_time(time):
    # Time in local timezone
    # reult is not localized
    date_yaxis = dt.datetime(year=1970, month=1, day=1).date()
    mydt = dt.datetime.combine(date_yaxis, time)
    return mydt


def _annotate_suntimes(ax, daterange):
    timezone = daterange.tz
    times = daterange.to_frame(index=False, name="date").merge(
        pd.DataFrame(columns=["date", "sunrise", "sunset"]), how="left", on="date"
    )
    sun = suntimes.Sun(lat=47.492, long=8.555)

    times[["sunrise", "sunset"]] = times.apply(
        lambda d: (
            _datetime_from_time(sun.sunrise(d["date"])),
            _datetime_from_time(sun.sunset(d["date"])),
        ),
        axis="columns",
        result_type="expand",
    )

    # Suntimes are calcualted in the local timezone and need to be localized
    times["sunset"] = times["sunset"].dt.tz_localize(timezone)
    times["sunrise"] = times["sunrise"].dt.tz_localize(timezone)
    # Alternatively do this in the apply call
    # import pytz
    # mydt = pytz.timezone(config.tz_local).localize(mydt)

    ax.plot(daterange, times["sunset"], color="#ffffff", lw=0.5)
    ax.plot(daterange, times["sunrise"], color="#ffffff", lw=0.5)
    return times


def HeatmapFigure(
    df,
    column,
    interval_minutes=None,
    timezone=None,
    title=None,
    ylabel=None,
    figsize=None,
    histy_label="Mean Demand\nProfile (kW)",
    histx_label="Daily Energy\n(kWh)",
    cbar_label="Power (kW)",
    **kwargs,
):
    """
    Make a figure with a full heatmap, daily overview and annotations

    Parameter
    ---------
    df : pandas.Dataframe
        Pandas Dataframe that holds **power** measuements in `column`.
        Needs to have a timezone aware DateTimeIndex.
            Localized to local timezone unless parameter `timezone` is passed.
            Requires frequency, unless parameters `interval_minutes` is passed.

    column : str
        Name of the column with the power measurements.

    Returns
    -------
    matplotlib.figure
        Figure with the heatmap
    """

    # Norms and cmaps to try out:
    # norm = colors.TwoSlopeNorm(vcenter=0.)
    # norm = colors.CenteredNorm()
    # cmap = 'gist_rainbow_r'
    # cmap = 'guppy'
    # cmap = cmr.fusion_r,

    if interval_minutes is None:
        # TODO complain if not set
        interval_minutes = df.index.freq.nanos / 60e9
        # interval_minutes = min(diff(df.index))

    if timezone is None:
        # TODO complain if not present
        # TODO localize if not localized
        timezone = df.index.tz

    # Generate the pivoted heatmap and corresponding time and date range
    data, daterange, timerange = _heatmap_data_from_pandas(df, column, interval_minutes)

    # Set up the figure and axes
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(8, 1),
        height_ratios=(2, 7),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.01,
        hspace=0.01 * DEFAULT_FIGRATIO,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_histy = fig.add_subplot(gs[1, 1])
    ax_cbar = ax_histx.inset_axes([1.07, 0, 0.035, 1])

    # TODO for some reasons setting this for the ax only does not work, so we
    # modify the global defualt for now
    # -> probably have to add it explicitpy to lacators
    # ax.xaxis_date(tz=timezone)
    # ax.yaxis_date(tz=timezone)
    # ax_histx.xaxis_date(tz=timezone)
    # ax_histy.yaxis_date(tz=timezone)
    plt.rcParams["timezone"] = str(timezone)

    _plot_hists(
        daterange,
        timerange,
        data,
        ax_histx,
        ax_histy,
        interval_minutes,
        histx_label=histx_label,
        histy_label=histy_label,
    )

    mesh = plot_pcolormesh(ax, daterange, timerange, data, **kwargs)

    cbar = fig.colorbar(
        mesh,
        cax=ax_cbar,
        label=cbar_label,
    )
    cbar.outline.set_color(COLOR_AXES)
    cbar.outline.set_linewidth(0)
    ax_cbar.tick_params(color=COLOR_AXES, rotation=90)
    # TODO fix alignment here: when doing it as below, the colorbar is not adjusted

    _annotate_suntimes(ax, daterange)

    return fig


def _heatmap_data_from_pandas(df, column, interval_minutes):
    """
    Get day/hour matrix from DataFrame
    """

    data_df = df.copy()
    timezone = data_df.index.tz
    # TODO why does this not make it work with pivot without aggfunc?
    # df.drop_duplicates(subset='Timestamp', keep='first', inplace=True)
    data_df["date"] = df.index.date
    data_df["time"] = df.index.time
    # data = df.pivot(index="to_time", columns="to_date", values='A+')
    # mysum = lambda x: x.sum(skipna=False)
    mysum = lambda x: x.iloc[0]
    data = data_df.pivot_table(
        index="time", columns="date", values=column, aggfunc=mysum, dropna=False
    )

    daterange = data.columns.astype("datetime64[ns]").tz_localize(timezone)
    # # daterange.freq = daterange.inferred_freq
    # daterange = daterange.union(pd.date_range(daterange[-1] + daterange.freq, periods=1, freq=daterange.freq))
    # daterange = pd.date_range(start=daterange.min(), end=daterange.max() + dt.timedelta(days=1), tz=config.tz_local)

    # timerange = data.index
    # timerange = timerange.union(pd.Index([dt.datetime.time(23, 59, 59)]))
    # timerange = pd.to_datetime(timerange)  # .astype('datetime64[ns]')
    # timerange = pd.Series(timerange)
    # timerange = timerange.apply(lambda t: dt.datetime.combine(dt.datetime(year=1970, month=1, day=1), t))
    timerange = pd.date_range(
        start="1970-01-01T00:00:00",
        end="1970-01-02T00:00:00",
        freq=f"{interval_minutes}min",
        tz=timezone,
    )

    data = data.to_numpy()
    # Discard the last day, since daterange needs to extend one day later
    data = data[:, :-1]
    return data, daterange, timerange


def plot_pcolormesh(ax, daterange, timerange, data, **kwargs):
    """
    Plot the 2D demand profile
    Take a numpy matrix and indices and make a figure with heatmap and sum/avg
    """

    mesh = ax.pcolormesh(daterange, timerange, data, **kwargs)
    ax.set_xlim(daterange[0], daterange[-1])
    ax.invert_yaxis()

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    # Use '%-H' for linux, '%#H' for windows to remove leading zero
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    ax.yaxis.set_minor_locator(mdates.HourLocator())

    # Remove last x tick label, to avoid overlapp with histogram
    # TODO does not work in older matplotlib
    # ax_xticks = ax.get_xticks()
    # ax_xticklabels = ax.get_xticklabels()
    # ax_xticklabels[-1] = ""
    # ax.set_xticks(ax_xticks)
    # ax.set_xticklabels(ax_xticklabels)

    ax.set_xlabel("Date")
    ax.set_ylabel("Hour")

    ax.spines["left"].set_color(COLOR_AXES)
    ax.spines["bottom"].set_color(COLOR_AXES)
    ax.tick_params(axis="both", which="both", color=COLOR_AXES)
    for pos in ["top", "right"]:
        ax.spines[pos].set_visible(False)

    return mesh


def _plot_hists(
    daterange,
    timerange,
    data,
    ax_histx,
    ax_histy,
    interval_minutes,
    histx_label=None,
    histy_label=None,
    minimal=False,
):

    # Daily sum
    daily_demand = np.nansum(data, axis=0) * interval_minutes / 60
    daily_max_draw = np.nanmax(data, axis=0)
    twinx = ax_histx.twinx()
    twinx.set_ylabel("Daily Peak (kW)", labelpad=0)
    twinx.scatter(
        daterange[:-1] + dt.timedelta(hours=12),
        daily_max_draw,
        color="black",
        s=1,
        linewidths=0,
    )
    twinx.set_ylim(0, None)

    for pos in ["top", "left", "bottom"]:
        twinx.spines[pos].set_visible(False)
    twinx.spines["right"].set_color(COLOR_AXES)
    # Rotate in case they are long
    twinx.tick_params(color=COLOR_AXES, rotation=90)
    # TODO this is deprecated but currently the only way to set alignment
    # TODO does not work in older mpl
    # twinx.set_yticks(twinx.get_yticks())
    # twinx.set_yticklabels(twinx.get_yticklabels(), rotation=90, va='center')

    ax_histx.fill_between(
        daterange[:-1] + dt.timedelta(hours=12),
        daily_demand,
        facecolor=COLOR_FACE,
        alpha=0.5,
    )
    ax_histx.set_xlim(daterange[0], daterange[-1])
    # Need to set the max here as welll, else when removing the lower tick (below) the limits get extended, since the ticks have not been renderd yet.
    ax_histx.set_ylim(min(0, daily_demand.min()), daily_demand.max())

    ax_histx.set_ylabel(histx_label)
    ax_histx.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    # yticks = ax_histy.yaxis.get_major_ticks()
    # yticks[0].set_visible(False)
    # Remove fist label because it may overlapp with heat map
    # TODO does not work in older mpl
    # ax_yticks = ax_histx.get_yticks()
    # ax_yticklabels = ax_histx.get_yticklabels()
    # ax_yticklabels[0] = ""
    # ax_histx.set_yticks(ax_yticks)
    # ax_histx.set_yticklabels(ax_yticklabels)

    # Mean profile
    ax_histy.fill_betweenx(
        timerange[:-1] + dt.timedelta(minutes=interval_minutes) / 2,
        np.nanmean(data, axis=1),
        facecolor=COLOR_FACE,
        alpha=0.5,
    )
    ax_histy.axes.yaxis.set_ticklabels([])
    # If demand is larger than zero, alsways show from zero, else show from negative demand on
    ax_histy.set_xlim(min(0, np.nanmean(data, axis=1).min()), None)
    ax_histy.set_ylim(timerange[0], timerange[-1])
    ax_histy.set_xlabel(histy_label)
    ax_histy.invert_yaxis()  # This has to be after setting lims

    # Hide the ticks and labels
    ax_histx.get_xaxis().set_visible(False)
    ax_histy.get_yaxis().set_visible(False)
    # Hide axes frame lines
    for pos in ["top", "right", "bottom"]:
        ax_histx.spines[pos].set_visible(False)
    for pos in ["top", "right", "left"]:
        ax_histy.spines[pos].set_visible(False)
    ax_histx.spines["left"].set_color(COLOR_AXES)
    ax_histx.tick_params(color=COLOR_AXES)
    ax_histy.spines["bottom"].set_color(COLOR_AXES)
    ax_histy.tick_params(color=COLOR_AXES)
