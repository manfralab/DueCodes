# Starting a new experiment

### Imports and qcodes setup

Import packages/modules/variables

```python
%load_ext autoreload
%autoreload 2

# typical python and qcodes stuff
%matplotlib notebook
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import itertools
import qcodes as qc
from qcodes.dataset.plotting import plot_by_id

# custom packages for Manfra lab
# not including MDAC or device classes
from shockley import sweeps as swp
from shockley import get_station_config_path, clear_station_instruments
from shockley.drivers.parameters import CurrentParam1211
```

Open/repoen database

```python
from qcodes import initialise_database
from qcodes import experiments, load_or_create_experiment

qc.config['core']['db_location'] = '/path/to/experiment/database.db'
initialise_database()

exp = load_or_create_experiment('4K_first_cooldown', sample_name='M0621191E')
```

Start station and load configuration file (here we are loading the config file for the Oxford 4K magnet dewar)

```python
station_config_file = get_station_config_path('4K magnet')
station = qc.Station(config_file=station_config_file)
```

### Load instruments

Now we can load some instruments by name from the station configuration file

```python
clear_station_instruments(station)

k2612 = station.load_instrument('k2612')
dmm24 = station.load_instrument('dmm24')
srs7 = station.load_instrument('srs7')
ithaco = station.load_instrument('ithaco')
```
