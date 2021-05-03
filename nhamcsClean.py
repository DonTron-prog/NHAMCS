import unidecode
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



file = open("features.txt")
features = file.read().splitlines()
file.close()
print(features)

dataset = pd.DataFrame(pd.read_csv(
    '/home/donald/github/NHAMCS/nhamcs2018.csv'))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Database with required attributes
nhamcs = dataset[features]

print (nhamcs.head(5))



'''
with open("features.txt", "r") as features:
    for line in features:

	lines = features.readlines()
    lines.append(as)
	print(lines)
'''
