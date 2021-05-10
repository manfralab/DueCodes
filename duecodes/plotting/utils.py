from matplotlib.offsetbox import AnchoredText

def add_plot_text(axis, text, loc="best"):
    # fontsize should == legend fontsize from rc params
    # figure it out later
    anchored_text = AnchoredText(text, loc=loc, prop={'fontsize': 10})
    anchored_text.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    anchored_text.patch.set(edgecolor='0.8')
    axis.add_artist(anchored_text)
    return anchored_text
