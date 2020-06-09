import sys
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Lasso

# def load_data():

# data = pd.read_csv('./data/used_car_sample_submit.csv')
# data = pd.read_csv('./data/used_car_testB_20200421.csv', sep=' ')
data = pd.read_csv('./data/used_car_train_20200313.csv', sep=' ')
# data = pd.read_csv('./data/used_car_testA_20200313.csv', sep=' ')
data['use_day'] = (pd.to_datetime(data['creatDate'].astype(str), errors='coerce') -
                   pd.to_datetime(data['regDate'].astype(str), errors='coerce')).dt.days
b = data.corr()['price'].sort_values()
# high_corr_keys = {1: ['regDate', 'v_0', 'v_8', 'v_12'],
#                   -1: ['v_3', 'kilometer']}


# bodyType
# b = data.corr()['price'].sort_values()
# b_0 = .corr()['price'].sort_values()
# b_1 = .corr()['price'].sort_values()


# # 这里我包装了一个异常值处理的代码，可以随便调用。
# def outliers_proc(data, col_name, scale=3):
#     """
#     用于清洗异常值，默认用 box_plot（scale=3）进行清洗
#     :param data: 接收 pandas 数据格式
#     :param col_name: pandas 列名
#     :param scale: 尺度
#     :return:
#     """
#
#     def box_plot_outliers(data_ser, box_scale):
#         """
#         利用箱线图去除异常值
#         :param data_ser: 接收 pandas.Series 数据格式
#         :param box_scale: 箱线图尺度，
#         :return:
#         """
#         iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
#         val_low = data_ser.quantile(0.25) - iqr
#         val_up = data_ser.quantile(0.75) + iqr
#         rule_low = (data_ser < val_low)
#         rule_up = (data_ser > val_up)
#         return (rule_low, rule_up), (val_low, val_up)
#
#     data_n = data.copy()
#     data_series = data_n[col_name]
#     rule, value = box_plot_outliers(data_series, box_scale=scale)
#     index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
#     print("Delete number is: {}".format(len(index)))
#     data_n = data_n.drop(index)
#     data_n.reset_index(drop=True, inplace=True)
#     print("Now column number is: {}".format(data_n.shape[0]))
#     index_low = np.arange(data_series.shape[0])[rule[0]]
#     outliers = data_series.iloc[index_low]
#     print("Description of data less than the lower bound is:")
#     print(pd.Series(outliers).describe())
#     index_up = np.arange(data_series.shape[0])[rule[1]]
#     outliers = data_series.iloc[index_up]
#     print("Description of data larger than the upper bound is:")
#     print(pd.Series(outliers).describe())
#
#     fig, ax = plt.subplots(1, 2, figsize=(10, 7))
#     sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
#     sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
#     return data_n

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


data = reduce_mem_usage(data)
