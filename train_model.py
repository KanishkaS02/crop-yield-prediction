import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# load dataset
data = pd.read_csv("dataset/crop_yield.csv")

# input features
X = data[['Rainfall','Temperature','Humidity','Fertilizer']]

# output
y = data['Yield']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create model
model = RandomForestRegressor()

# train model
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("model/yield_model.pkl","wb"))

print("Model trained successfully and saved!")