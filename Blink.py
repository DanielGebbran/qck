#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import time
from random import *

while True:
    raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'last_name': ['Miller', 'Jacobson', ".", 'Milner', 'Cooze'], 
        'age': [42, 52, 36, 24, 73], 
        'preTestScore': [4, 24, 31, ".", "."],
        'postTestScore': ["25,000", "94,000", 57, 62, 70]}
    df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])
    df
    df.to_csv('/Py/qck/my.csv')
    #df.to_csv('./my.csv')
    time.sleep(uniform(0.1,0.6))
