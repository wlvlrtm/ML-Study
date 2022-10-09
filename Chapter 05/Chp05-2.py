import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV



wine = pd.read_csv("https://bit.ly/wine_csv_data")
data = wine[["alcohol", "sugar", "pH"]]
target = wine["class"].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
print(sub_input.shape, val_input.shape)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))

scores = cross_validate(dt, train_input, train_target)
print(scores)

print(np.mean(scores["test_score"]))

scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores["test_score"]))

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
score = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores["test_score"]))

params = {"min_impurity_decrease" : [0.001, 0.002, 0.003, 0.004, 0.005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))
print(gs.best_params_)
print(gs.cv_results_["mean_test_score"])

best_index = np.argmax(gs.cv_results_["mean_test_score"])
print(gs.cv_results_["params"][best_index])

params = { "min_impurity_decrease" : np.arange(0.0001, 0.001, 0.0001),
            "max_depth" : range(5, 20, 1), 
            "min_samples_split" : range(2, 100, 10)}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(gs.best_params_)
print(np.max(gs.cv_results_["mean_test_score"]))

params = {"min_impurity_decrease" : uniform(0.0001, 0.001),
          "max_depth" : randint(20, 50),
          "min_samples_split" : randint(2, 25), 
          "min_samples_leaf" : randint(1, 25)}
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)

print(gs.best_params_)
print(np.max(gs.cv_results_["mean_test_score"]))
dt = gs.best_estimator_
print(dt.score(test_input, test_target))