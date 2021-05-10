import matplotlib.pyplot as plt
from pathlib import Path

STYLE_DIR = Path(__file__).parent / 'styles'

NOTEBOOK_STYLE = "duecodes-notebook.mplstyle"
DARK_STYLE = "qb-dark.mplstyle"
LIGHT_STYLE = "qb-light.mplstyle"


def notebook_style():
    # useful to load in your analysis notebook for nice copy and pasting of plots
    # use along with %matplotlib inline

    plt.style.use(str(STYLE_DIR / NOTEBOOK_STYLE))
