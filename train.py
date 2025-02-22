import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# Create TabularDataset using TabularDatasetFactory
# Data is located at:
url_path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
ds = TabularDatasetFactory.from_delimited_files(url_path)

def get_X_y(ds, encode_cat='onehot', context='hpd'):
  """Prepare features and a target in line with exploratory data analysis.
  For `encode_cat` parameter use either 'onehot' or 'label'.
  For `context` parameter use either 'hpd' or 'aml'."""
  
  if context == 'hpd':
    df = ds.to_pandas_dataframe()
  else:
    df = ds.copy() # in 'aml' context type(ds) == pd.DataFrame()

  # Separate and encode the target
  y = df.pop('y').apply(lambda s: 1 if s == 'yes' else 0)

  # Binarize 'pdays' feature, as it doesn't matter how many days passed
  # due to prevalence of 'no previous contact' cases (31728)
  df['pdays'] = df['pdays'].apply(lambda i: 0 if i == 999 else 1)

  # Drop a potential data leakage columns including
  # high correlated 'duration (of a call)' (coef 0.41).
  # Features related with the last contact of the current
  # campaign are not known when planning a new campaign!
  for col in ['contact', 'month', 'day_of_week', 'duration', 'campaign']:
    df.drop(col, axis=1, inplace=True)

  # Drop an uninformative column 'default'
  # which has only 3 'yes'.
  df.drop('default', axis=1, inplace=True)

  # Encode the non-numeric columns in the 'hpd' context
  if context == 'hpd':
    for col in df.select_dtypes('object').columns:
      if encode_cat == 'onehot':
        df = df.join(pd.get_dummies(df[col], prefix=col))
        df.drop(col, axis=1, inplace=True)
      else:  # Label encoding
        df[col], _ = df[col].factorize()

    # Return features and the target
    return df, y
  
  else: # for 'aml' context
    return pd.concat([df, y], axis=1)
     
    

x, y = get_X_y(ds)

# Split data into train and test sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=1000, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver='lbfgs').fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(model, './outputs/model.joblib')

if __name__ == '__main__':
    main()
