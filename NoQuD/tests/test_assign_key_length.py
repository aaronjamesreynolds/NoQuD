#!/usr/bin/env python
import NoQuD.read_input_data.read_csv_input_file
import pandas as pd
import numpy as np



def test_assign_key_length():

    filename = 'AI_test.csv'
    data = pd.read_csv(filename, header = None)
    assembly_types = int(data.iloc[0, 1])

    obv_key_length = NoQuD.read_input_data.read_csv_input_file.assign_key_length(data, assembly_types)
    exp_key_length = np.array([8, 8, 16])
    assert obv_key_length == exp_key_length



