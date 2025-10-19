# BvM

## Synthetic data

Generating data (Total runtime ~5hr):

```
python vanilla.py && python batched.py && python contextual.py && python lqr.py
```

Generating Plots:

```
python plot_vanilla.py && python plot_batched.py && python plot_contextual.py && python plot_lqr.py
```

## Zozo

Instructions for running the Zozo data:

1. Navigate to the ```zozo``` directory.
1. Unzip the data partition files by running: ```python unzip_parts.py```. Alternatively, redownload the data from the original source ```https://research.zozo.com/data.html#Shift15mDataset``` under Open Bandit Dataset. Unzip the downloaded zip file and obtain the file ```bts/all/all.csv```. Copy that into the ```zozo``` directory and run ```python partition_data.py```.
1. Run ```python get_states.py && get_bvm.py && plot_bvm.py```.
