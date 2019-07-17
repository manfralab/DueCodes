# Loading data for analysis

```python
%matplotlib notebook
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import qcodes as qc

from shockley import list_measured_params, get_data_from_ds
```

Connect to database with experiment data

```python
db_loc = '/path/to/experiment/database.db'
qc.config['core']['db_location'] = db_loc
```

Now use `list_measured_params()` to see what is in a dataset. And use `get_data_from_ds()` to load that data as a pandas dataframe or as a dictionary of numpy arrays.
