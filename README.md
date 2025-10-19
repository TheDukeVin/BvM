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

First, unzip the file all.csv.zip. Then, run:

```
python partition_data.py && get_states.py && get_bvm.py && plot_bvm.py
```