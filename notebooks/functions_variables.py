# Imports
import os
import json
import random
import string
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

# Modified function with Diba's code from the notebook.
def encode_tags(df, min_frequency):

    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low 
    counts of tags to keep cardinality to a minimum.
       
    Args:
        pandas.DataFrame
        min_frequency (int): Minimum number of times a tag must appear to be encoded.

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    # create a unique list of tags and then create a new column for each tag
    
    # Flatten the list of all tags and count occurrences
    tag_counts = Counter(df.explode("tags")["tags"])
    
    # Keep only tags that meet the minimum frequency requirement
    common_tags = {tag for tag, count in tag_counts.items() if count >= min_frequency}
    
    # Encode tags as binary values
    def filter_common_tags(tag_list):
        if isinstance(tag_list, list):
            return [tag for tag in tag_list if tag in common_tags]
        return []
    
    df["tags"] = df["tags"].apply(filter_common_tags)
    
    # Removing rows that tags is empty after filtering
    df = df[df["tags"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    
    # Apply MultiLabelBinarizer for one-hot encoding
    mlb = MultiLabelBinarizer()
    tags_encoded = pd.DataFrame(mlb.fit_transform(df["tags"]), columns=mlb.classes_)
    
    # Reset index before merging
    df = df.reset_index(drop=True)
    tags_encoded = tags_encoded.reset_index(drop=True)
    
    # Merge encoded tags with the original dataset
    df = pd.concat([df.drop(columns=["tags"]), tags_encoded], axis=1)

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


def calculate_min_frequency(data, column_name):
    """
    Calculates the minimum frequency threshold for filtering low-frequency categorical values (tags).

    Parameters:
    - data (pd.DataFrame): The dataset containing categorical values in list format.
    - column_name (str): The column containing lists of categorical values.
    - percentile (float): The percentile threshold for filtering (default: 10%).
    - min_threshold (int): The minimum threshold to prevent excessive filtering (default: 5).

    Returns:
    - int: Suggested threshold for filtering low-frequency tags.
    """

    # Flatten the column and count occurrences
    tag_counts = Counter(data.explode(column_name)[column_name])

    # Convert to DataFrame
    tag_counts_df = pd.DataFrame(tag_counts.items(), columns=["Tag", "Count"])
    tag_counts_df = tag_counts_df.sort_values(by="Count", ascending=False)

    # Determine threshold based on percentile
    threshold = tag_counts_df["Count"].quantile(0.10)
    threshold = max(5, int(threshold))  # Ensure minimum threshold is applied

    return threshold