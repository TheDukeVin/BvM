import os
import zipfile

N_dfs = 124

def delete_csv():

    for i in range(1, N_dfs+1):

        file_path = f"partition/part_{i}.csv"

        if os.path.exists(file_path):
            os.remove(file_path)

def unzip():

    for i in range(1, N_dfs+1):

        zip_path = f"partition/part_{i}.zip"
        extract_to = ""

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

unzip()
