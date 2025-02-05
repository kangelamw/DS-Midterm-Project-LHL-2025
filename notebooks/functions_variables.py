# Imports
import os
import json
import random
import string
import pandas as pd

# Included with the project
def encode_tags(df):

    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low 
    counts of tags to keep cardinality to a minimum.
       
    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    tags = df["tags"].tolist()
    # create a unique list of tags and then create a new column for each tag
        
    return df

# ----- We start here... ----- ###
def json_files_summary(path):
    """
    Summarizes files in a directory and stores file details in a df.

    Returns:
    - A DataFrame containing the summary of the JSON files in the specified directory path.
    """
    empty_list = []
    files = os.listdir(path)

    for i in files:
        file_path = os.path.join(path, i)
        try:
            with open(file_path, 'r') as f:
                data_json = json.load(f)
                
            df = pd.json_normalize(data_json['data']['results'])

            # File details
            file_name = os.path.basename(file_path)  # Get filename

            # File overview
            rows_count = df.shape[0]  # Get rows count
            cols_count = df.shape[1]  # Get columns count
            cols_name = df.columns.tolist()  # Get column names

            # Add to dictionary
            file_dict = {
                'file_name': file_name,
                'file_path': file_path,
                'rows_count': rows_count,
                'cols_count': cols_count,
                'cols_name': cols_name
            }

            # Append to list
            empty_list.append(file_dict)
            print(f"{file_name} processed. {len(empty_list)}/{len(files)} files processed.")

        except Exception as e:
            print(f"Error processing file {i}: {e}")
            continue

    # Convert the list to a DataFrame
    new_df = pd.DataFrame(empty_list)
    print("\nProcess complete.")
    
    return new_df

def read_json(file_path):
    """
    Reads our JSON files and returns its contents.

    Returns:
    - The contents of the JSON file as a DataFrame.
    """
    with open(file_path, 'r') as f:
        data_json = json.load(f)
    
    file_df = pd.json_normalize(data_json['data']['results'])
    return file_df

def cols_overview(df):
    """
    It takes a DataFrame as input and returns a sorted DataFrame with the following columns:
    - nulls_count: number of missing values in the column.
    - col_name: name of the column.
    - col_dtype: data type of the column.
    - nunique: number of unique values in the column.
    - unique: unique values in the column.
    - col_data_1: first 5 elements of the column.
    - col_data_2: last 5 elements of the column.
    
    Returns:
    - A DataFrame containing the overview of the columns in the input DataFrame.
    """
    cols = []
    for i in df:
        col = {'nulls_count': df[i].isnull().sum(),
            'col_name': i,
            'col_dtype': df[i].dtype,
            'nunique': df[i].nunique(),
            'unique': df[i].unique(),
            'col_data_1': df[i].head(5).tolist(),
            'col_data_2': df[i].tail(5).tolist()}
        cols.append(col)
    to_df = pd.DataFrame(cols)
    sorted = to_df.sort_values(by='nulls_count', ascending=False)
    return sorted

def generate_id():
    """
    Generates a random ID with the following format: LLNNNN
    where L is an uppercase letter and N is a digit.
    """
    # Generate 2 random uppercase letters
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    # Generate 4 random digits
    numbers = ''.join(random.choices(string.digits, k=4))
    # Combine letters and numbers
    return letters + numbers