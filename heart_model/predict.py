import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from heart_model import __version__ as _version
from heart_model.config.core import config
from heart_model.pipeline import heart_pipe
from heart_model.processing.data_manager import load_pipeline
from heart_model.processing.data_manager import pre_pipeline_preparation
from heart_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
heart_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config.features)
    #print(validated_data)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = heart_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = heart_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in={
        'Patient_ID':[101],
'Age':[37],
'Sex':['Female'],
'Cholesterol':[208],
'Blood_Pressure':[137/40],
'Heart_Rate':[70],
'Diabetes':[1],
'Alcohol_Consumption':[0],
'Diet':['Average']

    }
    
    make_prediction(input_data=data_in)
