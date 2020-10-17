import pickle
import json
import numpy
import azureml.train.automl
import pandas as pd

from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from sklearn.externals import joblib

# full set of columns
columns = ['ACCOUNT_NUMBER', 'AGE', 'APP_DOWNLOADED', 'APP_OFFER_USER',
    'BOUGHT_CATEGORY_FNN', 'BOUGHT_CATEGORY_10', 'BOUGHT_CATEGORY_WLN', 'BOUGHT_CATEGORY_3', 'BOUGHT_CATEGORY_4',
    'BOUGHT_CATEGORY_5', 'BOUGHT_CATEGORY_6', 'BOUGHT_CATEGORY_7', 'BOUGHT_CATEGORY_8', 'BOUGHT_CATEGORY_9',
    'CATEGORY_10_F', 'CATEGORY_10_R', 'CATEGORY_10_S', 'CATEGORY_1_F', 'CATEGORY_1_R', 'CATEGORY_1_S', 'CATEGORY_2_F', 'CATEGORY_2_R',
    'CATEGORY_2_S', 'CATEGORY_3_F', 'CATEGORY_3_R', 'CATEGORY_3_S', 'CATEGORY_4_F', 'CATEGORY_4_R', 'CATEGORY_4_S', 'CATEGORY_5_F', 'CATEGORY_5_R',
    'CATEGORY_5_S', 'CATEGORY_6_F', 'CATEGORY_6_R', 'CATEGORY_6_S', 'CATEGORY_7_F', 'CATEGORY_7_R', 'CATEGORY_7_S', 'CATEGORY_8_F', 'CATEGORY_8_R',
    'CATEGORY_8_S', 'CATEGORY_9_F', 'CATEGORY_9_R', 'CATEGORY_9_S', 'COUNTRY_CODE', 'EMAILABLE', 'GENDER',
    'HEALTH_CLUB', 'LATITUDE', 'LONGITUDE', 'MAILABLE', 'ONLINE_SHOPPER',
    'PARENT_CLUB', 'TENURE']

input_sample = pd.DataFrame([[48522366, 42.0, "N", "N", "N", "N", "N", "N", "N", "N", "N", "Y", "N", 0, 0, 0.0, 1, 3, 15.79, 2, 2, 7.44, 1, 3, 16.37, 1, 2, 9.21, 0, 0, 0.0, 8, 8, 750.38, 5, 6, 11.56, 1, 2, 49.55, 1, 1, 19.94, "ENG", "N", "M", "N", 53.79252613207973, 1.550589503639435, "Y", "Y", "N", 13]], columns = columns)
output_sample = pd.DataFrame([0])

def init():
    global model
    # this name is model.id of model that we want to deploy
    model_path = Model.get_model_path(model_name = '<<modelid>>')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

@input_schema('data', PandasParameterType(input_sample))
@output_schema(PandasParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
    except Exception as e:
        result = str(e)
        return {"error": result}
    return {"result":result.tolist()}