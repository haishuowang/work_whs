import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
plt.plotting()
import sys

sys.path.append('/mnt/mfs')
from work_dmgr_fut.loc_lib.pre_load.plt import savfig_send

# mean_squared_error(y_true, y_pred)

data_train = pd.read_csv('./data/zhengqi_train.txt', sep='\t')
data_test = pd.read_csv('./data/zhengqi_test.txt', sep='\t')

def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff ** 2)
    n = len(y_pred)

    return np.sqrt(sum_sq / n)


def mse(y_ture, y_pred):
    return mean_squared_error(y_ture, y_pred)


def find_outliers(model, X, y, sigma=3):
    # predict y values using model
    try:
        y_pred = pd.Series(model.predict(X), index=y.index)
    # if predicting fails, try fitting the model first
    except:
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)

    # calculate residuals between the model prediction and true y values
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()

    # calculate z statistic, define outliers to be where |z|>sigma
    z = (resid - mean_resid) / std_resid
    outliers = z[abs(z) > sigma].index

    # print and plot the results
    print('R2=', model.score(X, y))
    print('rmse=', rmse(y, y_pred))
    print("mse=", mean_squared_error(y, y_pred))
    print('---------------------------------------')

    print('mean of residuals:', mean_resid)
    print('std of residuals:', std_resid)
    print('---------------------------------------')

    print(len(outliers), 'outliers:')
    print(outliers.tolist())

    # plt.figure(figsize=(15, 5))
    # ax_131 = plt.subplot(1, 3, 1)
    # plt.plot(y, y_pred, '.')
    # plt.plot(y.loc[outliers], y_pred.loc[outliers], 'ro')
    # plt.legend(['Accepted', 'Outlier'])
    # plt.xlabel('y')
    # plt.ylabel('y_pred');
    #
    # ax_132 = plt.subplot(1, 3, 2)
    # plt.plot(y, y - y_pred, '.')
    # plt.plot(y.loc[outliers], y.loc[outliers] - y_pred.loc[outliers], 'ro')
    # plt.legend(['Accepted', 'Outlier'])
    # plt.xlabel('y')
    # plt.ylabel('y - y_pred');
    #
    # ax_133 = plt.subplot(1, 3, 3)
    # z.plot.hist(bins=50, ax=ax_133)
    # z.loc[outliers].plot.hist(color='r', bins=50, ax=ax_133)
    # plt.legend(['Accepted', 'Outlier'])
    # plt.xlabel('z')
    #
    # plt.savefig('outliers.png')

    return outliers


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def train_model(model, param_grid, X, y, splits=5, repeats=5):
    # get unmodified training data, unless data to use already specified
    # if len(y) == 0:
    #     X, y = get_trainning_data_omitoutliers()
    #     # poly_trans=PolynomialFeatures(degree=2)
    #     # X=poly_trans.fit_transform(X)
    #     # X=MinMaxScaler().fit_transform(X)

    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)

    # perform a grid search if param_grid given
    # if len(param_grid) > 0:
    # setup grid search parameters
    gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                           scoring="neg_mean_squared_error",
                           verbose=1, return_train_score=True, n_jobs=3)

    # search the grid
    gsearch.fit(X, y)

    # extract best model from the grid
    model = gsearch.best_estimator_
    best_idx = gsearch.best_index_

    # get cv-scores for best model
    grid_results = pd.DataFrame(gsearch.cv_results_)
    cv_mean = abs(grid_results.loc[best_idx, 'mean_test_score'])
    cv_std = grid_results.loc[best_idx, 'std_test_score']

    # no grid search, just cross-val score for given model
    # else:
    #     grid_results = []
    #     cv_results = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=rkfold)
    #     cv_mean = abs(np.mean(cv_results))
    #     cv_std = np.std(cv_results)

    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean': cv_mean, 'std': cv_std})

    # predict y using the fitted model
    y_pred = model.predict(X)

    # print stats on model performance
    print('----------------------')
    print(model)
    print('----------------------')
    print('score=', model.score(X, y))
    print('rmse=', rmse(y, y_pred))
    print('mse=', mse(y, y_pred))
    print('cross_val: mean=', cv_mean, ', std=', cv_std)

    # residual plots
    y_pred = pd.Series(y_pred, index=y.index)
    resid = y - y_pred
    mean_resid = resid.mean()
    std_resid = resid.std()
    z = (resid - mean_resid) / std_resid
    # n_outliers = sum(abs(z) > 3)
    #
    # plt.figure(figsize=(15, 5))
    # ax_131 = plt.subplot(1, 3, 1)
    # plt.plot(y, y_pred, '.')
    # plt.xlabel('y')
    # plt.ylabel('y_pred')
    # plt.title('corr = {:.3f}'.format(np.corrcoef(y, y_pred)[0][1]))
    # ax_132 = plt.subplot(1, 3, 2)
    # plt.plot(y, y - y_pred, '.')
    # plt.xlabel('y')
    # plt.ylabel('y - y_pred')
    # plt.title('std resid = {:.3f}'.format(std_resid))
    #
    # ax_133 = plt.subplot(1, 3, 3)
    # z.plot.hist(bins=50, ax=ax_133)
    # plt.xlabel('z')
    # plt.title('{:.0f} samples with z>3'.format(n_outliers))

    return model, cv_score, grid_results


data_train = reduce_mem_usage(data_train)
# scatter_matrix(data_train, figsize=(20, 16))

num_pipeline = Pipeline([
    # ('Imputer', Imputer("median")),
    ('StandardScaler', StandardScaler()),
])

data_train_std = num_pipeline.fit_transform(data_train)
data_train_std = pd.DataFrame(data_train_std, index=data_train.index,
                              columns=data_train.columns)
# savfig_send()

# data_train.iloc[:, :10].hist(bins=50, figsize=[20, 15])
linear_regression = LinearRegression()
X = data_train_std.iloc[:, :-1]
y = data_train_std['target']
# linear_model = linear_regression.fit(X, y)

# outliers = find_outliers(Ridge(), X, y)

model, cv_score, grid_results = train_model(LinearRegression(), {}, X=X, y=y, splits=5, repeats=5)


model = 'XGB'
opt_models[model] = XGBRegressor()

param_grid = {'n_estimators':[100,200,300,400,500],
              'max_depth':[1,2,3],
             }

opt_models[model], cv_score,grid_results = train_model(opt_models[model], param_grid=param_grid, X=X, y=y,
                                              splits=splits, repeats=1)

cv_score.name = model
score_models = score_models.append(cv_score)

y_test = opt_models['XGB'].predict(data_test)
y_test = pd.Series(y_test).to_csv('./y_test.txt', index=False, header=False)