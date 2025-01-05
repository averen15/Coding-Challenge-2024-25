# IMPORTING NECESSARY LIBRARIES

import numpy as np # numerical operations
import pandas as pd # data manipulation

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split # splitting the data into training and testing
from sklearn.preprocessing import StandardScaler, LabelEncoder # scaling features
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier # modelling
from sklearn.metrics import mean_squared_error, accuracy_score # evaluation metrics

import re
import matplotlib.pyplot as plt # graphs


# LOADING THE DATA

# uploading the cvs file, pf meaning data file
data = pd.read_csv('steels_data.csv')
data.head() # prints out the csv sheet


# PARSE AND EXTRACTING ELEMENTAL COMPOSITION

# loading csv into a pandas DataFrame
df = pd.DataFrame(data)

# parsing (making data more readable and processable)
# data_parse takes the strings and extracts weight%, applied to each column, result stored in new columns
def data_parse(formula):
    composition = {}
    for elements in ['c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']:
        match = re.search(rf'({elements})(\d*\.?\d+)', formula)
        if match:
            composition[elements] = float(match.group(2))
        else:
            composition[elements] = 0.0
        return composition

data.head()


# KNN IMPUTATION FOR MISSING VALUES

#converting columns to numeric (one-hot encoding)
df_encoded = pd.get_dummies(df)

# handling missing values using K-nearest neighbors algorithm
imputer = KNNImputer(n_neighbors=5)
df_imputed_encoded = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)

#* verifying if imputation was successful
print("Missing values after imputation: ")
print(df_imputed_encoded.isnull().sum())

# saves imputed DataFrame to a new csv
df_imputed_encoded.to_csv('imputed_data.csv', index=False) # false argument ensures row indicies are not saved as seperate columns in the csv file

print("\nImputed Data: ")
df_imputed_csv = pd.read_csv('imputed_data.csv') # loads the NEW csv file into DataFrame
print(df_imputed_csv) # prints out NEW the csv sheet

data.head()


# SPLITTING DATA (FEATURES & TARGETS) AND FEATURE SCALING

print(df_imputed_encoded.columns)

# features: compositional data
X = df_imputed_encoded.drop(columns=['yield_strength', 'tensile_strength', 'elongation']) # X contains chemical composition
print(df_imputed_encoded.columns)

# targets: yield strength and tensile strength will have regressions, elongation will be classified
y_strength = df_imputed_encoded[['yield_strength', 'tensile_strength']] # y_strength contain both yeild and tensiles strength 
y_elongation = df_imputed_encoded['elongation'].apply(lambda x: 'fragile' if x<5 else 'medium' if x<10 else 'strong') # y_elongation contains elongation

# scaling (removing mean and scaling to unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# SPLITTING DATA (TRAINING & TESTING), to evaluate performance of model

# train_test_split is splitting both features and targets into 80% training and 20% targets
X_train, X_test, y_train_strength, y_test_strength = train_test_split(X_scaled, y_strength, test_size=0.2, random_state=42)
X_train, X_test, y_train_elongation, y_test_elongation = train_test_split(X_scaled, y_elongation, test_size=0.2, random_state=42)

# displaying shape of training and testing data to ensure correctness
print(f"X_train shape: {X_train.shape}, X_test shape:{X_test.shape}")
print(f"y_train_strength shape: {y_train_strength.shape}, y_test_strenght shape: {y_test_strength.shape}")
print(f"y_train_elongation shape: {y_train_elongation.shape}, y_test_elongation shape: {y_test_elongation.shape}")

