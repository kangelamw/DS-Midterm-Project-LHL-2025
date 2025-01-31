## Part 1: EDA

Welcome to the new midterm project, it's been put together just for you!  You'll find a data folder with quite a few JSON files.  In each of these files you'll find data for housing sales in the US.  To wrangle the data, try iterating over each file, loading it, and storing it in memory.

Your tasks are as follows:
1. Load the data from the provided files (in the `data/` directory) into a Pandas dataframe
2. Explore, clean and preprocess your data to make it ready for ML Modelling - hints and guidance can be found in the `1 - EDA.ipynb` notebook
3. (Stretch) Explore some outside data sources - is there any other information you could join to your data that might be helpful to predict housing prices?
4. Perform EDA on your data to help understand the distributions and relationships between your variables
5. Save your finalized dataframes (`X_train`, `y_train`, `X_test` and `y_test`) as .csv's in your `data/` directory. You may want to make a `processed/` subfolder.

Complete the **1 - EDA.ipynb** notebook to demonstrate how you executed the tasks above. 

## Part 2: Model Selection

1. Try a variety of supervized learning models on your preprocessed data
2. Decide on your criteria for model selection - what metrics are most important in this context? Describe your reasoning 
3. (Stretch) Even after preprocessing, you may have a lot of features, but they not all be needed to make an accurate prediction. Explore Feature Selection. How does this change model performance? Remember that a simpler model is generally preferred to a complex one if performance is similar

Complete the **2 - model_selection.ipynb** notebook to demonstrate how you executed the tasks above.

## Part 3: Tuning and Pipelining 

1. Perform hyperparameter tuning on the best performing models from Part 2. But be careful! Depending on how you preprocessed your data, you may not be able to use the default Scikit-Learn functions without leaking information. You'll find some helpful starter docstrings in the `3 - tuning_pipeline.ipynb` notebook.
2. Save your tuned model - you may want to create a new `models/` directory in your repo
3. Build a pipeline that performs your preprocessing steps and makes predictions using your tuned model for new data - assume the new data is in the same JSON format as your original data.
4. Save your final pipeline 

Complete the **3 - tuning_pipeline.ipynb** notebook to demonstrate how you executed the tasks above.

Congratulations, your project is complete!
