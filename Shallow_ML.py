import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

# ML packages
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error
from bayes_opt import BayesianOptimization
import xgboost as xgb


def create_feature_importance_df(X, y, n_estimators=100, seed=0, method='RFR'):
    """
    Args:
        X (array)         : Predictor variables
        y (array)         : Outcome variables
        n_estimator (int) : Number of estimator of RFR method
        seed (int)        : Random seed for reproducibility
        method (str)      : Method for feature selection
    """
    if method == 'RFR': # Random Forest Regressor
        rfr = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
        rfr.fit(X, y)
        feature_importances = rfr.feature_importances_
    else:
        print(f'No supported method: {method}')
        exit(1)

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


def Bayesian_optimization_RF(X_train, y_train, pbounds, n_iter=20, cv=10, seed=0, n_jobs=-1, scoring="neg_mean_squared_error"):
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
    def RF(n_estimators, max_features, max_depth, min_samples_split):
        model = RandomForestRegressor(n_estimators=round(n_estimators),
                                      max_features=min(max_features, 0.999),  # Fraction, must be <= 1.0
                                      max_depth=round(max_depth),
                                      min_samples_split=round(min_samples_split),
                                      random_state=seed,
                                      n_jobs=n_jobs)
        model.fit(X_train, y_train)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
        return score

    # Bayesian optimization
    bo=BayesianOptimization(f=RF, pbounds=pbounds, verbose=2, random_state=seed)
    bo.maximize(2, n_iter-2) # (init_points n_iter, additional n_iter)
    
    # Read best model's score & parameter
    target_score = bo.max['target']
    best_params = bo.max['params']
    best_params_formatted = {
                        'n_estimators': round(best_params['n_estimators']),
                        'max_features': best_params['max_features'],
                        'max_depth': round(best_params['max_depth']),
                        'min_samples_split': round(best_params['min_samples_split']),
                        'random_state': seed,
                        'n_jobs': n_jobs 
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



