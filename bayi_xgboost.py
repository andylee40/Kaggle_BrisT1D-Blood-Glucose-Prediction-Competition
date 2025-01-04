from xgboost import DMatrix, train
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from bayes_opt import BayesianOptimization
from xgboost import XGBRegressor

# 載入資料
train_cleaned = pd.read_csv('train_cleaned.csv')
X = train_cleaned.drop('bg+1:00', axis=1)
# X = X.loc[:, ['p_num', 'bg-0:15', 'bg-0:05', 'bg-0:00', 'hour_cos']]
y = train_cleaned['bg+1:00']

test_cleaned = pd.read_csv('test_cleaned.csv')
# test_cleaned = test_cleaned.loc[:, ['p_num', 'bg-0:15', 'bg-0:05', 'bg-0:00', 'hour_cos']]
test=pd.read_csv('test.csv')


# 超參數範圍
pbounds = {
    'learning_rate': (0.01, 0.1),
    'max_depth': (3, 10),
    'min_child_weight': (0, 5),
    'subsample': (0.5, 0.7),
    'colsample_bytree': (0.01, 1),
    'n_estimators': (100, 500),
    'gamma': (0.01, 1),
    'reg_alpha': (0.01, 1),
    'reg_lambda': (0.5, 5),
    'base_score': (0.2, 1)
}

# 目標函數
def xgboost_hyper_param(learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, 
                        n_estimators, gamma, reg_alpha, reg_lambda, base_score):
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)

    reg = XGBRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        base_score=base_score,
        early_stopping_rounds=10,
        eval_metric="rmse",
        tree_method="hist",
        device="cuda"
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        reg.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        y_pred = reg.predict(X_valid)
        scores.append(np.sqrt(np.mean((y_valid - y_pred) ** 2)))
    return -np.mean(scores) ##Bayesian Optimization 最大化，需返回負值

# 開始優化找最佳參數
optimizer = BayesianOptimization(
    f=xgboost_hyper_param,
    pbounds=pbounds,
    random_state=1,
)

# 執行優化
# init_points：初始化點數量、n_iter：迭代次數
optimizer.maximize(init_points=5, n_iter=30)

# 印出最佳結果、參數
print(optimizer.max)

# 儲存最佳參數
best_param = optimizer.max['params']
max_depth_p = round(best_param['max_depth'])
learning_rate_p = best_param['learning_rate']
n_estimators_p = round(best_param['n_estimators'])
gamma_p = best_param['gamma']
reg_alpha_p = best_param['reg_alpha']
reg_lambda_p = best_param['reg_lambda']
min_child_weight_p = round(best_param['min_child_weight'])
subsample_p = best_param['subsample']
colsample_bytree_p = best_param['colsample_bytree']
base_score_p = best_param['base_score']

# 預測test資料
def submitpredict(X_train, y_train, X_test):
    # 初始化測試集預測結果
    test_predictions = np.zeros(X_test.shape[0])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, valid_idx in kf.split(X_train):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        best_reg = XGBRegressor(
                max_depth=max_depth_p,
                learning_rate=learning_rate_p,
                n_estimators=n_estimators_p,
                gamma=gamma_p,
                reg_alpha=reg_alpha_p,
                reg_lambda=reg_lambda_p,
                min_child_weight=min_child_weight_p,
                subsample=subsample_p,
                colsample_bytree=colsample_bytree_p,
                base_score=base_score_p,
                early_stopping_rounds=10,
                eval_metric="rmse",
                tree_method="hist",
                device="cuda"
        )     
        best_reg.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_valid, y_fold_valid)],
            verbose=False
        )
        test_predictions += best_reg.predict(X_test) / kf.n_splits
    return test_predictions


# def submitpredict(X_train, y_train, X_test):
#     best_reg = XGBRegressor(
#                 max_depth=max_depth_p,
#                 learning_rate=learning_rate_p,
#                 n_estimators=n_estimators_p,
#                 gamma=gamma_p,
#                 reg_alpha=reg_alpha_p,
#                 reg_lambda=reg_lambda_p,
#                 min_child_weight=min_child_weight_p,
#                 subsample=subsample_p,
#                 colsample_bytree=colsample_bytree_p,
#                 base_score=base_score_p,
#                 tree_method="hist",
#                 device="cuda"
#         )     
#     best_reg.fit(
#         X_train, y_train,
#         verbose=False
#     )
#     test_predictions=best_reg.predict(X_test)
#     return test_predictions

X_test=test_cleaned
test_predictions=submitpredict(X, y, X_test)
submission = pd.DataFrame({'id': test['id'], 'bg+1:00': test_predictions})
submission.to_csv('submission_1122_4.csv', index=False)

# X_test=test_cleaned
# test_predictions=best_reg.predict(X_test)
# submission = pd.DataFrame({'id': test['id'], 'bg+1:00': test_predictions})
# submission.to_csv('submission_1121_3.csv', index=False)