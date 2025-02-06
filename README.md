# Data Science Midterm Project

## Project/Goals
This project aims to predict U.S. housing prices using real estate transaction data. By cleaning and analyzing raw data, we develop machine learning models to estimate home values based on key features like location, size, and market trends.

Goals:
- Load, preprocess, and explore housing sales data.
- Train, evaluate, and optimize supervised learning models.
- Fine-tune the best model and implement a prediction pipeline.

Following a structured approach, this project applies best practices in data science to build an accurate and scalable housing price prediction model.

<br>

## Process

### [Data Cleaning, Exploration and Visualization]('./notebooks/1 - EDA.ipynb')
#### Data Importing
1. Wrote the following functions to automate the import process.
  - `json_files_summary(path)` : Takes a directory path and stores file details.
  - `read_json(file_path)` : Takes a specific file path, reads all the JSON files and returns its contents as a DataFrame.
2. Excluded files with no data in them.

#### Data Cleaning and Wrangling
1. Wrote `cols_overview(df)` function which takes a DataFrame as and returns another sorted DataFrame (according to nulls_count) with details of concern such as the number of null_values, the unique values, column name, it's first 5 values, last 5 values, and dtype of the column.
2. Reviewed, replaced and imputed null values as necessary.
3. Dropped columns that are completely empty.
4. Dropped redundant columns.

#### Dealing with Tags
1. `'tags'` column is an object of `'str'` dtype and needed converted into a `'list'` type for `.explode()` to work.
2. Modified  `'encode_tags(df, min_frequency)'` function to manually encode tags from each sale. This function utilized sklearn's `MultiLabelBinarizer` to perform One-Hot Encoding on tags.
3. Wrote function `'calculate_min_frequency(data, column_name)'` to calculate the minimum frequency threshold for filtering low-frequency categorical values (tags)

#### Dealing with Cities
1. Split the data and used the training data to encode `'city'` with mean sale price.
2. Used binary encoder on property `'description.type'` column (only 9 unique values), so we can keep the categorical components without having to add more columns.

#### EDA/Visualization
1. Analyzed numerical variable distributions.
  ![histogram](./images/numerical_columns_hist.png)

2. Checked for skewness and applied log transformation for highly skewed columns.
  ![histogram 2](./images/numerical_columns_hist_log.png)

3. Checked for potentially redundant features and found none.

### 2 - [Model Selection]('./notebooks/2 - model_selection.ipynb')

<br>

## Results
Models we tried:
1. Linear Regression, RidgeCV, LassoCV and Elastic Net
2. Support Vector Machine for Regression
3. Random Forest
4. XGBoost

(fill in how your model performed)

<br>

## Challenges 
(discuss challenges you faced in the project)

...to not finetune as you test models, because WHY NOT

<br>

## Future Goals
(what would you do if you had more time?)
