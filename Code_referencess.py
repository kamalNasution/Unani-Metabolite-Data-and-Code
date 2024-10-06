# code Refrencess


# AdaBoost

# define the model with default hyperparameters
model = AdaBoostClassifier()

# define the grid of values to search
grid = dict()
grid['n_estimators'] = [10, 50, 100, 500]
grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(x, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# Bagging

from sklearn.ensemble import BaggingClassifier
# define the model with default hyperparameters
model = model = BaggingClassifier(bootstrap=False)

# define the grid of values to search
grid = dict()
grid['n_estimators'] = [10, 50, 100, 500]
grid['base_estimator'] = [ SVC() , DecisionTreeClassifier()]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(x, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# bernauli
from sklearn.naive_bayes import BernoulliNB
# define the model with default hyperparameters
model = model = model = BernoulliNB()

# define the grid of values to search
grid = dict()
grid['alpha'] = [1, 0.1, 0.01]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(x, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Decission Tree
from sklearn.naive_bayes import BernoulliNB
# define the model with default hyperparameters
model = DecisionTreeClassifier()

# define the grid of values to search
grid = dict()
grid['max_depth'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
grid['min_samples_split'] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(x, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Extra trees

from sklearn.ensemble import ExtraTreesClassifier
# define the model with default hyperparameters
model = ExtraTreesClassifier()

# define the grid of values to search
grid = dict()
grid['criterion'] = ['gini', 'entropy']
grid['max_features'] = ['auto', 'sqrt', 'log2' , None]
grid['n_estimators'] = [10,20,50,100,200]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(x, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
# define the model with default hyperparameters
model = GradientBoostingClassifier()

# define the grid of values to search
grid = dict()
grid['loss'] = ['deviance', 'exponential']
grid['n_estimators'] = [10,50,100,200,500]
grid['criterion'] = ['friedman_mse', 'mse']
grid['max_features'] = ['auto', 'sqrt', 'log2']


# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(x, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# KNN

from sklearn.neighbors import KNeighborsClassifier
# define the model with default hyperparameters
model = KNeighborsClassifier()

# define the grid of values to search
grid = dict()
grid['n_neighbors'] = [2,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,37,29]
grid['metric'] = ["euclidean", "manhattan", "cityblock" , "minkowski"]


# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(x, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# logistics regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# define the grid of values to search
grid = dict()
grid['C'] = [0.1, 0.5, 1, 5, 10, 50, 100]



# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(x, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



## Multi layer perceptron
from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
# define the grid of values to search
grid = dict()
grid['solver'] = ['lbfgs', 'sgd', 'adam']
grid['activation'] = ['identity', 'logistic', 'tanh', 'relu']
grid['alpha'] = [0.00001 ,0.0001 , 0.001, 0.01 , 0.5 , 1]
grid['learning_rate'] = ['constant', 'invscaling', 'adaptive']
grid['momentum'] = [0.5 ,0.9 , 0.95 , 0.99]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(x, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
# define the grid of values to search
grid = dict()
grid['max_depth'] = [5,10,20,25,30,40,51]
grid['min_samples_split'] = [2,3,4,5,6,7,8,9,10,11]
grid['n_estimators'] = [10 ,50 , 100, 150 , 200 , 500]
grid['criterion'] = ['gini', 'entropy']
grid['max_features'] = ['auto', 'sqrt', 'log2']

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(x, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# SVM 

model = SVC()
# define the grid of values to search
grid = dict()
grid['C'] = [0.1, 0.5, 1, 5]
grid['kernel'] = ['Linear', 'rbf', 'poly']

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(x, y)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# feature selection 
from sklearn.feature_selection import SelectKBest
# we Will select the top 500 important features
sel_five_cols = SelectKBest(mutual_info_classif, k=500)
sel_five_cols.fit(x, y)
index = x.columns[sel_five_cols.get_support()]


# PCA 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
pca = PCA(n_components = 20)
x = pca.fit_transform(x)


# SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(x, y)



