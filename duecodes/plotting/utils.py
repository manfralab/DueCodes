from matplotlib.offsetbox import AnchoredText

def add_plot_text(axis, text, loc="best"):
    # fontsize should == legend fontsize from rc params
    # figure it out later
    anchored_text = AnchoredText(text, loc=loc, prop={'fontsize': 10})
    anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    anchored_text.patch.set(edgecolor='0.8')
    axis.add_artist(anchored_text)
    return anchored_text

def legend_without_duplicate_labels(ax, loc='best'):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), loc=loc)
