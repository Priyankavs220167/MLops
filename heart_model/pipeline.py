import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from heart_model.config.core import config

from heart_model.processing.features import Mapper,age_col_tfr

print("variable- ",config.model_config.target)

heart_pipe = Pipeline([

    
    ##==========Mapper======##
    ('map_sex',Mapper(config.model_config.gender_var, config.model_config.gender_mappings)
     ),
    ('map_diet',Mapper(config.model_config.diet_var, config.model_config.diet_mappings)
     ),

     # Transformation of age column
    ("age_transform", age_col_tfr(config.model_config.age_var)
    ),
    
    # scale
    ('scaler', StandardScaler()),

    # Model fit
    ('model_rf', RandomForestClassifier(n_estimators=150, max_depth=5,random_state=42))
    
])

