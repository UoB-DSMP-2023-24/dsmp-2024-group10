import os
import pandas as pd

path = os.path.abspath(".")
df = pd.read_csv(path + "/Data/vdjdb.txt", sep="\t")
num = df["mhc.a"].nunique()
print(num)