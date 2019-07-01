import pandas as pd
from MDAC import MDAC

def check_pairs(smu_param, pairs, df = None, cutoff = 1000.0):
    # smu_param is some qcodes parameter that returns a resistance

    # pairs is a list of tuples that refer to pairs of MDAC microD pins to test
    # pin1 goes to bus, pin2 goes to ground
    # the SMU is connected to bus and measures the resistance from pin1 to pin2 to ground
    # df is an existing dataframe to append data
    # cutoff is a threshold resistance to determine whether or not the pair is good
    #
    # example of pairs...
    # a = np.arange(1, 33, 2) # evens
    # b = np.arange(2, 33, 2) # odds
    # pairs = list(zip(a,b))

    if df is None:
        df = pd.DataFrame(columns=['pin1', 'pin2', 'resistance', 'ok'])

    for i, p1, p2 in enumerate(pairs):
