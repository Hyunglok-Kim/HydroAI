import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ML packages
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error
from bayes_opt import BayesianOptimization
import xgboost as xgb
import cuml
from cuml.ensemble import RandomForestRegressor as cuRF
from cuml.svm import SVR
import lightgbm as lgb
from cuml.linear_model import ElasticNet as cuML_ElasticNet



def feature_engineering_XbyVIF(X_train):
    vif = pd.DataFrame()
    vif['VIF_Factor'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif['Feature'] = X_train.columns
    vif = vif.sort_values(by='VIF_Factor', ascending=False)
    vif = vif.reset_index(drop=True)

    return vif


def create_feature_importance_df(X, y, n_estimators=100, seed=0, n_jobs=-1):
    """
    Args:
        X (array)         : Predictor variables
        y (array)         : Outcome variables
        n_estimator (int) : Number of estimator of RFR method
        seed (int)        : Random seed for reproducibility
        n_jobs (int)      : Number of cpu core (-1 means available all cores)
    """
    rfr = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)
    rfr.fit(X, y)
    feature_importances = rfr.feature_importances_

    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    feature_importance_df = feature_importance_df.reset_index(drop=True)
 
    return feature_importance_df


def select_important_feature_list(feature_importance_df, threshold):
    """
    Args:
        feature_importance_df (dataframe) : Result of "create_feature_importance_df" function 
        threshold (float)                 : Threshold value of importance column
    """
    return list(feature_importance_df[feature_importance_df['importance'] > threshold]['feature'].values)


def Bayesian_optimization_RF(X_train, y_train, pbounds, n_iter=20, cv=10, seed=0, n_jobs=-1, scoring="neg_mean_squared_error", library_model='RF_sklearn'):
    """
    Args:
        X_train, y_train (array) : Train dataset for supervised learning
        pbounds (dict)           : Bounds of n_estimators, max_features, max_depth, min_samples_split
        n_iter (int)             : Number of iteration to optimize (>2)
        cv (int)                 : Number of cross validation
        seed (int)               : Random seed for reproducibility
        n_jobs (int)             : Number of cpu core (-1 means available all cores)
        scoring (str)            : Scoring method
    """
    def RF_sklearn(n_estimators, max_features, max_depth, min_samples_split):
        model = RandomForestRegressor(n_estimators=round(n_estimators),
                                      max_features=min(max_features, 0.999),  # Fraction, must be <= 1.0
                                      max_depth=round(max_depth),
                                      min_samples_split=round(min_samples_split),
                                      random_state=seed,
                                      n_jobs=n_jobs)
        model.fit(X_train, y_train)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
        return score

    def RF_cuml(n_estimators, max_features, max_depth, min_samples_split):
        model = cuRF(n_estimators=round(n_estimators),
                     max_features=min(max_features, 0.999),  # Fraction, must be <= 1.0
                     max_depth=round(max_depth),
                     min_samples_split=round(min_samples_split),
                     n_streams=1, random_state=seed) # n_stream & random_state: For reproducible
        model.fit(X_train, y_train)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
        return score

    # Bayesian optimization
    if library_model == 'RF_sklearn':
        func = RF_sklearn
    elif library_model == 'RF_cuml':
        func = RF_cuml
    else:
        print(f'No supported library ML model: {library_model}')
        exit(1)
    bo=BayesianOptimization(f=func, pbounds=pbounds, verbose=2, random_state=seed)
    bo.maximize(2, n_iter-2) # (init_points n_iter, additional n_iter)

    # Read best model's score & parameter
    target_score = bo.max['target']
    best_params = bo.max['params']
    best_params_formatted = {
                        'n_estimators': round(best_params['n_estimators']),
                        'max_features': best_params['max_features'],
                        'max_depth': round(best_params['max_depth']),
                        'min_samples_split': round(best_params['min_samples_split']),
                        'random_state': seed
                        }
    print("Best model's target score:", target_score)
    print("Best model's parameter:", best_params_formatted)

    return target_score, best_params_formatted


def Bayesian_optimization_XGB(X_train, y_train, pbounds, n_iter=20, cv=10, seed=0):
    def XGB(n_estimators, max_features, max_depth, min_samples_split):
        model = xgb.XGBRegressor(n_estimators=round(n_estimators),
                                 max_depth=round(max_depth),
                                 min_child_weight=round(min_samples_split),
                                 colsample_bytree=max_features,
                                 tree_method='hist', # specifies the histogram-based algorithm for faster training
                                 objective='reg:squarederror', # regression objective function
                                 device='cuda', # specifies that the model should use a GPU for training
                                 random_state=seed)

        # Using a custom progress bar for cross-validation
        cv_results = cross_val_score(model, X_train, y_train, cv=cv, scoring=make_scorer(mean_squared_error, greater_is_better=False))

        return sum(cv_results) / len(cv_results)

    bo = BayesianOptimization(f=XGB, pbounds=pbounds, verbose=2, random_state=seed)
    bo.maximize(2, n_iter-2) # (init_points n_iter, additional n_iter)

    # Read best model's score & parameter
    target_score = bo.max['target']
    best_params = bo.max['params']
    best_params_formatted = {
                        'n_estimators': round(best_params['n_estimators']),
                        'max_depth': round(best_params['max_depth']),
                        'min_child_weight': round(best_params['min_samples_split']),
                        'colsample_bytree': best_params['max_features'],
                        'tree_method': 'hist',
                        'objective': 'reg:squarederror',
                        'device': 'cuda',
                        'random_state': seed
                    }

    return target_score, best_params_formatted


def Bayesian_optimization_SVM(X_train, y_train, pbounds, n_iter=20, cv=10, seed=0, scoring="neg_mean_squared_error", library_model="SVM_sklearn"):
    """
    Args:
        X_train, y_train (array) : Train dataset for supervised learning
        pbounds (dict)           : Bounds of C, epsilon, gamma, degree (for polynomial kernel)
        n_iter (int)             : Number of iteration to optimize (>2)
        cv (int)                 : Number of cross validation
        seed (int)               : Random seed for reproducibility
        scoring (str)            : Scoring method
    """
    def SVM_sklearn(C, epsilon, gamma, degree):
        print("Not yet supported SVM_sklearn function")
        exit(1)

    def SVM_cuml(C, epsilon, gamma, degree):
        model = SVR(C=C,
                    epsilon=epsilon,
                    gamma=gamma,
                    degree=round(degree),
                    kernel='rbf')  # You can change this to 'linear', 'poly', 'sigmoid' as needed.
        model.fit(X_train, y_train)

        # Perform cross-validation using cuML's cross_val_score
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
        return score

    # Bayesian optimization
    if library_model == 'SVM_sklearn':
        func = SVM_sklearn
    elif library_model == 'SVM_cuml':
        func = SVM_cuml
    else:
        print(f'No supported library ML model: {library_model}')
        exit(1)
    bo = BayesianOptimization(f=func, pbounds=pbounds, verbose=2, random_state=seed)
    bo.maximize(init_points=2, n_iter=n_iter-2)  # (init_points n_iter, additional n_iter)

    # Read best model's score & parameter
    target_score = bo.max['target']
    best_params = bo.max['params']
    best_params_formatted = {
        'C': best_params['C'],
        'epsilon': best_params['epsilon'],
        'gamma': best_params['gamma'],
        'degree': round(best_params['degree']),
        'kernel': 'rbf'  # Change as needed
    }

    print("Best model's target score:", target_score)
    print("Best model's parameter:", best_params_formatted)

    return target_score, best_params_formatted


def Bayesian_optimization_LGBM(X_train, y_train, pbounds, n_iter=20, cv=10, seed=0, scoring="neg_mean_squared_error"):
    """
    Args:
        X_train, y_train (array) : Train dataset for supervised learning
        pbounds (dict)           : Bounds of num_leaves, max_depth, learning_rate, n_estimators
        n_iter (int)             : Number of iteration to optimize (>2)
        cv (int)                 : Number of cross validation
        seed (int)               : Random seed for reproducibility
        scoring (str)            : Scoring method
    """
    def LGBM(num_leaves, max_depth, learning_rate, n_estimators):
        model = lgb.LGBMRegressor(num_leaves=round(num_leaves),
                                  max_depth=round(max_depth),
                                  learning_rate=learning_rate,
                                  n_estimators=round(n_estimators),
                                  random_state=seed,
                                  device='cuda',  # Enable GPU acceleration
                                  n_jobs=-1,
                                  verbose=-1)  # Suppress warnings and messages

        model.fit(X_train, y_train)

        # Perform cross-validation using sklearn's cross_val_score
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
        return score

    # Bayesian optimization
    bo = BayesianOptimization(f=LGBM, pbounds=pbounds, verbose=2, random_state=seed)
    bo.maximize(init_points=2, n_iter=n_iter-2)  # (init_points n_iter, additional n_iter)

    # Read best model's score & parameter
    target_score = bo.max['target']
    best_params = bo.max['params']
    best_params_formatted = {
        'num_leaves': round(best_params['num_leaves']),
        'max_depth': round(best_params['max_depth']),
        'learning_rate': best_params['learning_rate'],
        'n_estimators': round(best_params['n_estimators']),
        'random_state': seed,
        'device': 'cuda'  # Ensure GPU acceleration is used
    }

    print("Best model's target score:", target_score)
    print("Best model's parameter:", best_params_formatted)

    return target_score, best_params_formatted


def Bayesian_optimization_ElasticNet(X_train, y_train, pbounds, n_iter=20, cv=10, seed=0, scoring="neg_mean_squared_error", library_model="ElasticNet_cuml"):
    """
    Args:
        X_train, y_train (array) : Train dataset for supervised learning
        pbounds (dict)           : Bounds of alpha and l1_ratio
        n_iter (int)             : Number of iteration to optimize (>2)
        cv (int)                 : Number of cross validation
        seed (int)               : Random seed for reproducibility
        scoring (str)            : Scoring method
    """
    def ElasticNet_sklearn(alpha, l1_ratio):
        print("Not yet supported ElasticNet_sklearn function")
        exit(1)

    def ElasticNet_cuml(alpha, l1_ratio):
        model = cuML_ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_train, y_train)

        # Perform cross-validation using cuML's cross_val_score
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
        return score

    # Bayesian optimization
    if library_model == 'ElasticNet_sklearn':
        func = ElasticNet_sklearn
    elif library_model == 'ElasticNet_cuml':
        func = ElasticNet_cuml
    else:
        print(f'No supported library ML model: {library_model}')
        exit(1)
    bo = BayesianOptimization(f=func, pbounds=pbounds, verbose=2, random_state=seed)
    bo.maximize(init_points=2, n_iter=n_iter-2)  # (init_points n_iter, additional n_iter)

    # Read best model's score & parameter
    target_score = bo.max['target']
    best_params = bo.max['params']
    best_params_formatted = {
        'alpha': best_params['alpha'],
        'l1_ratio': best_params['l1_ratio']
    }

    print("Best model's target score:", target_score)
    print("Best model's parameter:", best_params_formatted)

    return target_score, best_params_formatted
