import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
pd.set_option('display.max_columns', None)
dataset = pd.read_csv("e-shop clothing 2008.csv", sep=";")

#clean column names
dataset.columns = [c.lower().replace(" ", "_").replace("(", "").replace(")", "") for c in dataset.columns]

clickfilter = dataset.groupby(["session_id"])["order"].count()
sessions = dataset.groupby(["session_id", "month"])["order"].count().reset_index()
sessions["flag"] = np.where((sessions["order"]>=10), 1,0)

# Training Data
#1st 3 clicks of sessions with >10 clicks from Apr-Jul
training_data = dataset[(dataset["order"]<=3) & (dataset["month"].isin([4,5,6,7]))]

#characteristics of the 1st 3 clicks of the training data
features = training_data.groupby("session_id").agg(
#pricing
    price_avg = ("price", "mean"),
    price_dev = ("price", "std"),

#entropy count
    en_categories = ("page_1_main_category", "nunique"),
    en_colors = ("colour", "nunique"),
    en_model = ("page_2_clothing_model", "nunique"),

#landing page metrics
    land_category = ("page_1_main_category", "first"),
    land_colors = ("colour", "first"),
    land_location = ("location", "first")
).reset_index()

# Validation Data
validation_data = sessions[sessions["month"].isin([4,5,6,7])][["session_id","flag"]]

# Testing Data
testing_data_august = dataset[(dataset["month"]==8) & (dataset["order"]<=3)]

testing_data = sessions[sessions["month"]==8][["session_id","flag"]]

features_august = testing_data_august.groupby("session_id").agg(
#pricing
    price_avg = ("price", "mean"),
    price_dev = ("price", "std"),

#entropy count
    en_categories = ("page_1_main_category", "nunique"),
    en_colors = ("colour", "nunique"),
    en_model = ("page_2_clothing_model", "nunique"),

#landing page metrics
    land_category = ("page_1_main_category", "first"),
    land_colors = ("colour", "first"),
    land_location = ("location", "first")
).reset_index()


categorical_data = ["land_category", "land_colors", "land_location"]
for i in categorical_data:
    features[i] = features[i].astype(str)
    features_august[i] = features_august[i].astype(str)

#one-hot encoding
encoding_features = pd.get_dummies(features, columns=categorical_data)
encoding_features_august = pd.get_dummies(features_august, columns=categorical_data)

#master files
training_master = pd.merge(encoding_features, validation_data, on="session_id", how="inner")
testing_master = pd.merge(encoding_features_august, testing_data, on="session_id", how="inner")

X_train = training_master.drop(columns=["session_id", "flag"])
y_train = training_master["flag"]

X_test = testing_master.drop(columns=["session_id", "flag"])
y_test = testing_master["flag"]

#random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=67, class_weight="balanced")
model.fit(X_train, y_train)

# Test the system on unseen August data
august_predictions = model.predict(X_test)

print(classification_report(y_test, august_predictions))

