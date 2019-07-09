import os
import pandas as pd
from WindPy import *
import datetime


class WindData:
    def __init__(self, begin_date='20180101', end_date = datetime.datetime.now().strftime('%Y%m%d')):
        self.begin_date = begin_date
        self.end_date = end_date

    def get_spot_data(self, spot_id):
        try:
            w.start()
            add_data = w.edb(spot_id, self.begin_date, self.end_date)
            w.close()
            if add_data.Data:
                add_df = pd.DataFrame(add_data.Data).T
                add_df.index = pd.to_datetime([x.strftime('%Y%m%d') for x in add_data.Times])
                return add_df
            else:
                return pd.DataFrame()
        except Exception as error:
            print(error)
        finally:
            return pd.DataFrame()

    def get_spots_data(self, spot_id_df):
        for x, y in spot_id_df.iterrows():
            self.get_spot_data(y.iloc[0])




def spot_data_update(ttt):
    change = []
    root_file = r'\\win-g12\ResearchWorks\Fulltime\haishuo.wang\Spot\spot_data\{}'.format(ttt)
    filenames = os.listdir(root_file)
    for i in range(len(filenames)):
        path = root_file + '\\' + filenames[i]
        spot_data = pd.read_excel(path, index_col=0)
        spot_data.index.name = None
        df = pd.DataFrame()
        spot_data_info = spot_data.ix[:8]
        spot_data = spot_data.ix[8:]
        spot_data.index = pd.to_datetime(spot_data.index)
        for j in range(len(spot_data.columns)):
            column = spot_data.columns[j]
            print(j)
            print(column)
            column_df = spot_data[column].dropna()
            begin_date = column_df.index[-1].strftime('%Y%m%d')
            end_date = datetime.date.today().strftime('%Y%m%d')
            print(end_date)
            id = spot_data_info.ix[u'指标ID'][column].encode('utf-8')
            name = spot_data_info.ix[u'指标名称'][column].encode('utf-8')
            add_data = w.edb(id, begin_date, end_date)
            if add_data.Data:
                add_df = pd.DataFrame(add_data.Data).T
                add_df.index = pd.to_datetime([x.strftime('%Y%m%d') for x in add_data.Times])
                add_df = add_df.ix[add_df.index > begin_date]
                for i_error in range(len(add_df)):
                    if type(add_df.ix[-(i_error + 1)]) == unicode:
                        add_df = add_df.ix[:-(i_error + 1)]
                if list(add_df.values):
                    if type(add_df.values.ravel()[0]) != float:
                        column_df = column_df.append(add_df)
                        column_df.columns = [column]
                        change += [[name, filenames[i][:-4], ttt, add_df.index[-1], add_df.values.ravel()[-1]]]
                        print('changed')
                else:
                    print('no change')
            else:
                print('no change')
            df = pd.concat([df, column_df], axis=1)

        # new_spot_data = pd.concat([spot_data_info, spot_data], axis=0)
        new_spot_data = pd.concat([spot_data_info, df], axis=0)
        print(new_spot_data)
        condition = False
        for k in range(len(new_spot_data.columns)):
            if type(new_spot_data.ix[-1, k]) == unicode:
                condition = True
                break
        if condition:
            new_spot_data.ix[:-1].to_excel(path)
        else:
            new_spot_data.to_excel(path)
    return change


if __name__ == '__main__':
    # List = ['煤焦钢矿_焦煤焦炭', '煤焦钢矿_螺纹线材','煤焦钢矿_硅铁锰硅', '煤焦钢矿_铁矿石', '农副产品_鸡蛋', '油脂油料_菜油', '油脂油料_大豆', '油脂油料_豆油']
    # List = ['油脂油料_豆油', '化工_橡胶']
    List = ['软产品_白糖', '能源_动力煤']
    # List = ['煤焦钢矿_螺纹线材']
    change_list = []
    w.start()
    for i in range(len(List)):
        change = spot_data_update(List[i])
        change_list += change
    w.close()
    if change_list:
        change_df = pd.DataFrame(change_list)
        change_df.to_csv('change_df.csv', index=False, encoding='gbk')
        sm.mail(['change_df.csv'], ['spot_change.csv'], 'data update time')
    else:
        sm.mail1(u'\u4ef7\u683c', 'spot data daily update')
    print('_____________________________________________________________')
    print('end~')
