# Data Science Midterm Project

## Project/Goals
This project aims to predict U.S. housing prices using real estate transaction data. By cleaning and analyzing raw data, we develop machine learning models to estimate home values based on key features like location, size, and market trends.

Goals:
- Load, preprocess, and explore housing sales data.
- Enhance predictions with external data sources.
- Train, evaluate, and optimize supervised learning models.
- Fine-tune the best model and implement a prediction pipeline.

Following a structured approach, this project applies best practices in data science to build an accurate and scalable housing price prediction model.

<br>

## Process
### Data Importing
1. Wrote the following functions to automate the import process.
  - `json_files_summary(path)` : Takes a directory path and stores file details.
  - `read_json(file_path)` : Takes a specific file path, reads all the JSON files and returns its contents as a DataFrame.
2. Excluded files with no data in them.

### Data Cleaning and Wrangling
1. Wrote the following functions to automate the cleaning process.
  - `cols_overview(df)` : Takes a DataFrame as and returns another sorted DataFrame (according to nulls_count) with details of concern such as the number of null_values, the unique values, column name, it's first 5 values, last 5 values, and dtype of the column.
2. Reviewed and replaced null values as necessary.
3. Dropped columns that are completely empty.

### (your step 3)

<br>

## Results
(fill in how your model performed)

<br>

## Challenges 
(discuss challenges you faced in the project)

<br>

## Future Goals
(what would you do if you had more time?)
