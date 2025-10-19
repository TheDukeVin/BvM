import pandas as pd

input_file = "all.csv"
chunk_size = 100000  # number of rows per smaller file

# iterate through file in chunks
for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
    output_file = f"partition/part_{i+1}.csv"
    chunk.to_csv(output_file, index=False)
    print(f"Wrote {output_file} ({len(chunk)} rows)")
