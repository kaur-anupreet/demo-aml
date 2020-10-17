
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get_data():
    df = pd.read_csv('/tmp/azureml_runs/boots/data-latest.csv')

    le = LabelEncoder()
    le.fit(df['BOUGHT_CATEGORY_FNN'].values)
    y = le.transform(df['BOUGHT_CATEGORY_FNN'].values)

    df = df.drop(['BOUGHT_CATEGORY_FNN'], axis=1)

    return { "X" : df, "y" : y }
