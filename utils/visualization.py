import matplotlib.pyplot as plt
import seaborn as sns


# Estimated density with central tendecies
def plot_density_with_central_tendencies(series, set_title=True):
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [6, 1]}
    )
    sns.histplot(
        data=series,
        element='step',
        kde=True,
        stat='probability',
        ax=ax_hist
    )
    ax_hist.vlines(
        series.mean(),
        *ax_hist.get_ylim(),
        color='red',
        ls='--',
        lw=1.5,
        label='mean'
    )
    ax_hist.vlines(
        series.median(),
        *ax_hist.get_ylim(),
        color='green',
        ls='-.',
        lw=1.5,
        label='median'
    )
    ax_hist.vlines(
        series.mode()[0],
        *ax_hist.get_ylim(),
        color='purple',
        ls='-',
        lw=1.5,
        label='mode'
    )
    sns.boxplot(
        data=series.values,
        showfliers=True,
        showmeans=True,
        meanline=True,
        meanprops={'linewidth': 2, 'color': 'red'},
        orient='horizontal',
        ax=ax_box
    )

    if set_title:
        ax_hist.set_title(series.name)
    ax_hist.set(xlabel=None, ylabel=None)
    ax_hist.legend()
    sns.despine(ax=ax_hist)

    plt.tight_layout()


# Estimated density with central tendencies with respect to the target feature
def plot_train_density_with_central_tendencies_by_target(data_train, feat):
    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    sns.kdeplot(
        data=data_train,
        x=feat,
        hue='TARGET',
        shade=False,
        common_norm=False,
        palette=['tab:green', 'tab:red'],
        ax=ax_hist
    )
    sns.boxplot(
        data=data_train,
        x=feat,
        y='TARGET',
        showfliers=True,
        showmeans=True,
        meanline=True,
        meanprops={'linewidth': 2, 'color': 'tab:blue'},
        boxprops=dict(facecolor='None'),
        orient='horizontal',
        ax=ax_box
    )

    ax_hist.legend(labels=['defaulter', 'non-defaulter'])
    ax_box.set_yticklabels(['non-defaulter', 'defaulter'])
    ax_box.set(ylabel=None)
    sns.despine(ax=ax_hist)
    plt.tight_layout()


# Bar plot with respect to the target feature
def plot_train_bars_by_target(data_train, feat):
    df = data_train.groupby(feat).TARGET.agg(defaulter=sum)
    df['non-defaulter'] = data_train.groupby(
        feat).TARGET.agg(lambda x: x.eq(0).sum())
    df.sort_values(
        by='defaulter',
        ascending=False,
        inplace=True
    )
    # figsize = (10, 4) if df.shape[0] >= 20 else None # One-liner
    if df.shape[0] >= 100:
        figsize = (15, 4)
    elif df.shape[0] >= 20:
        figsize = (10, 4)
    else:
        figsize = None

    df.plot.bar(
        stacked=True,
        color=['tab:red', 'tab:green'],
        figsize=figsize
    )
    plt.tight_layout()
    plt.show()

    df['total'] = df.sum(1)
    print(df)
