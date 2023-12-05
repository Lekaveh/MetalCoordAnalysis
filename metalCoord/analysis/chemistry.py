import os
import sys
import pandas as pd

d = os.path.dirname(sys.modules["metalCoord"].__file__)
data = pd.read_csv(os.path.join(d, "data/ideal_cova_rad_for_all_elememt.list"), delimiter='         ', engine='python').reset_index()
data.columns = ['Element', 'Radius'] 
radiuses = dict(data.values)

