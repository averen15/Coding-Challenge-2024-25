import pandas as pandas

# loading the data set, getting summary statistics
data = pd.read_cvs('steels_data.cv')
print(data.isnull().sum())




