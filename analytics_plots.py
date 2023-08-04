import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def bland_altman_plots(m1, m2, sd_limit=1.96, ax=None, annotate=True,
                       scatter_kwds=None, mean_line_kwds=None, limit_lines_kwds=None):

    if ax is None:
        try:
            import matplotlib.pyplot as plt
        except:
            raise ImportError("Matplotlib is not found.")
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    scatter_kwds = scatter_kwds or {}
    if 's' not in scatter_kwds:
        scatter_kwds['s'] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'gray'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 1
    if 'linestyle' not in mean_line_kwds:
        kwds['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kwds:
        kwds['linestyle'] = ':'

    ax.scatter(means, diffs, **scatter_kwds) # Plot the means against the diffs.
    ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

    # Annotate mean line with mean difference.
    if annotate:
        ax.annotate('mean diff:\n{}'.format(np.round(mean_diff, 2)),
                    xy=(0.99, 0.5),
                    horizontalalignment='right',
                    verticalalignment='center',
                    fontsize=14,
                    xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)
        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kwds)
        if annotate:
            range_lim = (0.20 - (-0.20))
            low_lim = (lower - (-0.20))/(0.20 - (-0.20))
            high_lim = (upper - (-0.20))/(0.20 - (-0.20))
            ax.annotate(f'\N{MINUS SIGN}{sd_limit} SD: {lower:0.2g}',
                        xy=(0.99, low_lim-0.02*range_lim), # (0.99, 0.07),
                        horizontalalignment='right',
                        verticalalignment='top',
                        fontsize=14,
                        xycoords='axes fraction')
            ax.annotate(f'+{sd_limit} SD: {upper:0.2g}',
                        xy=(0.99, high_lim+0.02*range_lim), # (0.99, 0.92),
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        fontsize=14,
                        xycoords='axes fraction')
    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

    ax.set_ylabel('Difference', fontsize=15)
    ax.set_xlabel('Means', fontsize=15)
    ax.tick_params(labelsize=13)
    fig.tight_layout()
    return fig


def box_plots(df_hs_clean, df_to_clean):
    cm = 1 / 2.54

    hs_cols = ['LF_HS', 'RF_HS']
    to_cols = ['LF_TO', 'RF_TO']

    fig, axs = plt.subplots(1, 2, figsize=(18 * cm, 8 * cm), sharey=True)
    sns.boxplot(ax=axs[0], data=df_hs_clean, y="diff_msec", x="event", order=hs_cols, showfliers=True,
                flierprops=dict(marker='o', ms=2, mec=(0, 0, 0), alpha=0.4, mfc='none'))
    sns.despine(bottom=True)
    axs[0].set_xlabel("")  # clear xlabel
    axs[0].set_xticklabels([s.replace("_HS", '') for s in hs_cols])
    axs[0].set_ylabel('time error (in s)')  # set ylabel
    axs[0].set_yticks(np.arange(-200, 250, 100))
    # axs[0].set_yticklabels(np.arange(-200, 250, 100)/1000)
    axs[0].set_yticklabels([f'{x}'.replace('-', '\N{MINUS SIGN}') for x in np.arange(-200, 250, 100) / 1000])
    axs[0].yaxis.set_minor_locator(plt.MultipleLocator(10))
    axs[0].grid(visible=True, which="major", axis="y", c=(0, 0, 0), alpha=0.1, ls=":")
    axs[0].set_title("Heel Strike")
    axs[0].tick_params(axis="x", which="both", length=0)
    sns.boxplot(ax=axs[1], data=df_to_clean, y="diff_msec", x="event", order=to_cols, showfliers=True,
                flierprops=dict(marker='o', ms=2, mec=(0, 0, 0), alpha=0.4, mfc='none'))
    sns.despine(bottom=True)
    axs[1].set_xlabel("")  # clear xlabel
    axs[1].set_ylabel("")
    axs[1].set_xticklabels([s.replace("_TO", '') for s in to_cols])
    # axs[1].set_ylabel('time error / ms')  # set ylabel
    axs[1].set_yticks(np.arange(-200, 250, 100))
    axs[1].yaxis.set_minor_locator(plt.MultipleLocator(10))
    axs[1].grid(visible=True, which="major", axis="y", c=(0, 0, 0), alpha=0.1, ls=":")
    axs[1].set_title("Toe Off")
    axs[1].tick_params(axis="x", which="both", length=0)
    plt.tight_layout()
    plt.show()


def scatter_plots(x):

    cm = 1 / 2.54
    fig, axs = plt.subplots(1, 3, figsize=(43 * cm, 14 * cm), sharey=True)

    bland_altman_plots(x.stride_time_true, x.stride_time_pred, ax=axs[0], annotate=True, scatter_kwds=dict(c="r"))
    bland_altman_plots(x.stance_time_true, x.stance_time_pred, ax=axs[1], annotate=True,  scatter_kwds=dict(c="g"))
    bland_altman_plots(x.swing_time_true, x.swing_time_pred, ax=axs[2], annotate=True,  scatter_kwds=dict(c="b"))

    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].set_title("Stride time", size=18)
    axs[0].set_xlabel("Mean (in s)")
    axs[0].set_ylabel("Difference (in s)")
    axs[0].set_xticks(np.arange(0, 3.5, 0.5))
    axs[0].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].set_title("Stance time", size=18)
    axs[1].set_xlabel("Mean (in s)")
    axs[1].set_ylabel("")
    axs[1].set_xticks(np.arange(0, 3.5, 0.5))
    axs[1].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)
    axs[2].set_title("Swing time", size=18)
    axs[2].set_xlabel("Mean (in s)")
    axs[2].set_ylabel("")
    axs[2].set_xticks(np.arange(0, 3.5, 0.5))
    axs[2].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    axs[0].set_ylim((-0.20, 0.20))
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=16)

    plt.show()
