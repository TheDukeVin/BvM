
import pandas as pd
import numpy as np

N_dfs = 124

n_arms = 80

counts = np.zeros(n_arms)
rewardSums = np.zeros(n_arms)

for i in range(1, N_dfs+1):

    df = pd.read_csv(f'partition/part_{i}.csv')

    out_df = pd.DataFrame(columns=[f'count{i}' for i in range(n_arms)] +
                                  [f'sum{i}' for i in range(n_arms)], index=np.arange(len(df)))

    for j, row in df.iterrows():
        counts[row['item_id']] += 1
        rewardSums[row['item_id']] += row['click']

        out_df.iloc[j] = np.concatenate([counts, rewardSums])

    out_df.to_csv(f'states/state{i}.csv')