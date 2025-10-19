
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

font = {'size' : 20}
plt.rc('font', **font)



N_dfs = 124

dfs = []

for i in range(1, N_dfs+1):

    df = pd.read_csv(f'bvm/bvm_{i}.csv')

    dfs.append(df)

merged = pd.concat(dfs)

plt.plot(merged['Time'] * 1e-06, merged['BvM'])
plt.ylabel('TV dist')
plt.xlabel('Steps (millions)')
plt.tight_layout()
plt.savefig('zozo')

print(merged['SE'].max())
