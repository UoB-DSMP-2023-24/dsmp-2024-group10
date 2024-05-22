import pandas as pd
import numpy as np

df_copy= pd.read_csv('vdjdb.txt',sep='\t')
print("len(df_copy): ", len(df_copy))