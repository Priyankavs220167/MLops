
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from heart_model.config.core import config
from heart_model.processing.features import age_col_tfr


def test_age_variable_transformer(sample_input_data):
    # Given
    transformer = age_col_tfr(
        variables=config.model_config.age_var,  # cabin
    )
    assert np.isnan(sample_input_data[0].loc[711,'Age'])

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[711,'Age'] == 69
    