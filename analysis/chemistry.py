import pandas as pd

data = pd.read_csv("./data/ideal_cova_rad_for_all_elememt.list", delimiter='         ', engine='python').reset_index()
data.columns = ['Element', 'Radius'] 
radiuses = dict(data.values)

