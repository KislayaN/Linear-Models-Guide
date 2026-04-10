from sklearn.linear_model import LinearRegression

import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# from 