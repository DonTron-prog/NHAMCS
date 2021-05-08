import unidecode
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


#import list of relevent features
file = open("features.txt")
features = file.read().splitlines()
file.close()

dataset = pd.DataFrame(pd.read_csv(
    '/home/donald/github/NHAMCS/nhamcs2018.csv'))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Database with required attributes
nhamcs = dataset[features]
'''
diagnostics = ['J101', 'J111'] #'J101', 'J111'- 143 occurances, J101 - 67 occurances
#J09X - 32 occurances
nhamcs = nhamcs[nhamcs['DIAG1'].isin(diagnostics)]
'''
#convert Diagonstic to 1 for influenze otherwise zero
print (nhamcs['RFV1'].unique())

nhamcs['DIAG1'] = nhamcs['DIAG1'].map({'J101': 1,
                                       'J111': 1})
nhamcs['DIAG1'] = pd.to_numeric(nhamcs['DIAG1'], errors='coerce').fillna(0, downcast='infer')#docast infer makes it integer



print(nhamcs['DIAG1'].value_counts())

'''
import rfv #reason for visit list
nhamcs = nhamcs[nhamcs["RFV1"].isin(rfv.rfv1)]
print (nhamcs.head(5))
'''
'''
#Kernel density estimation (kde) plots display the values of a smoothed density curve of the histogam data values
def plot_density_hist(nhamcs, cols, bins = 10, hist = False):
    for col in cols:
        sns.set_style("whitegrid")
        sns.distplot(nhamcs[col], bins = bins, rug=True, hist = hist)
        plt.title('histogram of ' + col)
        plt.xlabel(col)
        plt.ylabel('Number of visits')
        plt.show()

num_cols = ['VMONTH']
plot_density_hist(nhamcs, num_cols, bins = 20, hist = True)
'''

'''
#series with the avearge of each day of the week
month = nhamcs.groupby(["VMONTH", "VDAYR"], sort = True).mean()
print(month)

month = list(month['AGE'])
y = month
fig, ax = plt.subplots(figsize=(20, 6))

ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Week of Month')
#ax.plot(y.resample('W').mean(),marker='v', markersize=8, linestyle='-', label='Weekly Mean Resample')
#ax.plot(y.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('Average')
ax.legend();
plt.show()
'''
