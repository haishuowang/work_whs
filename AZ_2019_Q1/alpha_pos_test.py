import sys

# 247MXRF7A4
# sys.path.append('/mnf/mfs')
# from work_whs.loc_lib.pre_load import *
# from work_whs.loc_lib.pre_load.plt import savfig_send
import pandas as pd
import numpy as np
import os
from datetime import datetime


class bt:
    @staticmethod
    def AZ_Load_csv(target_path, parse_dates=True, sep='|', **kwargs):
        target_df = pd.read_table(target_path, sep=sep, index_col=0, low_memory=False, parse_dates=parse_dates,
                                  **kwargs)
        return target_df


class ReturnData:
    def __init__(self, root_path, begin_date, end_date):
        self.root_path = root_path
        self.begin_date = begin_date
        self.end_date = end_date
        return_df_raw = self._load_return_data()
        self.return_df_raw = return_df_raw

        self.xinx = return_df_raw.index
        return_df = return_df_raw.copy()
        return_df['IF01'] = self.load_index_data_('000300')
        return_df['IC01'] = self.load_index_data_('000905')
        self.return_df = return_df

    def load_index_data(self, index_name):
        data = bt.AZ_Load_csv(f'{self.root_path}/EM_Funda/DERIVED_WHS/CHG_{index_name}.csv', header=None)
        target_df = data.iloc[:, 0].reindex(index=self.xinx)
        return target_df

    def load_index_data_(self, index_name):
        data = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/INDEX_TD_DAILYSYS/CHG.csv'))
        target_df = data[index_name].reindex(index=self.xinx)
        return target_df * 0.01

    def _load_return_data(self):
        return_df = bt.AZ_Load_csv(os.path.join(self.root_path, 'EM_Funda/DERIVED_14/aadj_r.csv'))
        return_df = return_df[(return_df.index >= self.begin_date) & (return_df.index < self.end_date)]
        return return_df

    def load_hedge_return(self, if_weight, ic_weight):
        hedge_df = self.return_df['IF01'] * if_weight + self.return_df['IC01'] * ic_weight
        return_df_hedge = self.return_df.sub(hedge_df, axis=0)
        return return_df_hedge

    def cut_low_pos(self, pos_df):
        pos_df.rank(pct=True, axis=1)

    def annual_return(self, pos_df, window=600):
        if_weight, ic_weight = 1, 0
        pos_df_cut = pos_df.iloc[-window:]
        return_df_hedge = self.load_hedge_return(if_weight, ic_weight) \
            .reindex(index=pos_df_cut.index, columns=pos_df_cut.columns)

        pnl_df_cut = (pos_df_cut * return_df_hedge).sum(1)
        pnl_df_cut_l = (pos_df_cut[pos_df_cut > 0] * return_df_hedge).sum(1)
        pnl_df_cut_s = (pos_df_cut[pos_df_cut < 0] * return_df_hedge).sum(1)

        # plt.plot(pnl_df_cut.cumsum(), color='b')
        # plt.plot(pnl_df_cut_l.cumsum(), color='r')
        # plt.plot(pnl_df_cut_s.cumsum(), color='g')
        # plt.grid()
        # savfig_send(subject=file_name)

        annual_r = pnl_df_cut.sum() / pos_df_cut.abs().sum(1).sum() * 250
        margin_l = pnl_df_cut_l.sum() / pos_df_cut[pos_df_cut > 0].abs().sum(1).sum() * 250
        margin_s = pnl_df_cut_s.sum() / pos_df_cut[pos_df_cut < 0].abs().sum(1).sum() * 250
        return annual_r, margin_l, margin_s


# def temp_fun():
#     root_path = '/mnt/mfs/DAT_EQT'
#     begin_date, end_date = pd.to_datetime('20130101'), datetime.now()
#     return_data = ReturnData(root_path, begin_date, end_date)
#     pos_path = '/mnt/mfs/AAPOS'
#     whs_list = sorted([x for x in os.listdir('/mnt/mfs/AAPOS') if x.startswith('WHS')])
#
#     stock_id = '000963.SZ'
#     whs_list = ['WHSAMEAME18.pos']
#     for file_name in whs_list:
#         if file_name.split('.')[0][3:-2] in ['AMEAME']:
#             print(file_name)
#             pos_df = bt.AZ_Load_csv(f'{pos_path}/{file_name}')
#             plt.figure(figsize=[16, 10])
#             pnl_df_1 = (pos_df.shift(2) * return_data.return_df).sum(1)
#             pnl_df_2 = (pos_df.shift(0) * return_data.return_df).sum(1)
#             plt.plot(pnl_df_1.cumsum(), color='b')
#             plt.plot(pnl_df_2.cumsum(), color='r')
#             plt.grid()
#             savfig_send(subject=file_name)


if __name__ == '__main__':
    root_path = '/media/hdd1/DAT_EQT'
    # root_path = '/mnt/mfs/DAT_EQT'
    begin_date, end_date = pd.to_datetime('20130101'), datetime.now()
    return_data = ReturnData(root_path, begin_date, end_date)
    pos_path = '/mnt/mfs/AAPOS'
    whs_list = sorted([x for x in os.listdir('/mnt/mfs/AAPOS') if x.startswith('WHS')])

    for file_name in whs_list[:1]:
        file_name = 'WHSQWERTY13.pos'
        print(file_name)
        pos_df = bt.AZ_Load_csv(f'{pos_path}/{file_name}').shift(2)
        annual_r, margin_l, margin_s = return_data.annual_return(pos_df, window=620)
        print('raw return')
        print(annual_r, margin_l, margin_s)
