current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.get_data import Load_data
from eda.eda import EDA

data_loader = Load_data()
dataframe = data_loader.load()

eda = EDA(data=dataframe)
skewed_cols = eda.get_skewed_cols() # Skewed Columns 
outlier_cols = eda.get_outlier_cols() # Outlier Columns

result = eda.build_preprocessing_plan()

log_transform_cols = result['Log Transform']
robust_scale_cols = result['Robust Scale']
standard_scale_cols = result['Standard Scale']
ohe_encode_cols = result['OHE (One Hot Encoding)']
high_card_encode = result["High Card Encode"]
corr_feat_to_drop = result["Drop Corr"]
drop_id_cols = result['Drop ID']
vif_selected_cols = result['VIF Selected']

import sys
import os

# columns to drop = corr_feat_to_drop + drop_id_cols

    #    X_train = X_train[vif_selected_cols]
        # X_test = X_test[vif_selected_cols]
        
        # log_cols = [c for c in log_transform_cols if c in X_train.columns]
        # robust_cols = [c for c in robust_scale_cols if c in X_train.columns]
        # standard_cols = [c for c in standard_scale_cols if c in X_train.columns]
        # ohe_cols = [c for c in ohe_encode_cols if c in X_train.columns]