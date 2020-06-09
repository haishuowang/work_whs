import sys

sys.path.append('/mnt/mfs')
from work_dmgr_fut.loc_lib.pre_load import *
from work_dmgr_fut.loc_lib.pre_load.plt import savfig_send
from work_dmgr_fut.fut_script.FutDataLoad import FutData, FutClass
from work_dmgr_fut.fut_script.signal_fut_fun import FutIndex, Signal, Position
from work_dmgr_fut.loc_lib.pre_load.senior_tools import SignalAnalysis

fut_data = FutData()

tick_path = '/media/hdd1/CTP_DATA_HUB/TICk_DATA'

date_list = sorted(os.listdir(tick_path))


class BaseTools:
    def __init__(self):
        self.root_path = '/mnt/mfs/DAT_FUT'
        self.trade_time = pd.read_csv(f'{self.root_path}/DailyPX/TradeTime', sep='|', index_col=0)['TradeTime']
        self.trade_date = bt.AZ_Load_csv(f'{self.root_path}/DailyPX/TradeDates').index
        self.trade_intra_date = self.trade_date + timedelta(hours=16)

    def get_start_time(self, fut_name):
        trade_time = pd.read_csv(f'{self.root_path}/DailyPX/TradeTime', sep='|', index_col=0)['TradeTime']
        start_time_list = []
        for trade_range_str in trade_time.loc[fut_name].split(','):
            begin_str, end_str = trade_range_str.split('-')
            start_time_list.append(begin_str)
        return start_time_list

    def get_start_time_dict(self, fut_name_list):
        target_dict = {}
        for fut_name in fut_name_list:
            target_dict[fut_name] = self.get_start_time(fut_name)
        return target_dict

    def get_end_time(self, fut_name):
        trade_time = pd.read_csv(f'{self.root_path}/DailyPX/TradeTime', sep='|', index_col=0)['TradeTime']
        end_time_list = []
        for trade_range_str in trade_time.loc[fut_name].split(','):
            begin_str, end_str = trade_range_str.split('-')
            end_time_list.append(end_str)
        return end_time_list

    def get_end_time_dict(self, fut_name_list):
        target_dict = {}
        for fut_name in fut_name_list:
            target_dict[fut_name] = self.get_end_time(fut_name)
        return target_dict

    def get_start_end_time(self, fut_name):
        trade_time = pd.read_csv(f'{self.root_path}/DailyPX/TradeTime', sep='|', index_col=0)['TradeTime']
        start_end_time_list = []
        for trade_range_str in trade_time.loc[fut_name].split(','):
            begin_str, end_str = trade_range_str.split('-')
            start_end_time_list.append([begin_str, end_str])
        return start_end_time_list

    def judge_trade_date(self, date_now=datetime.now()):
        today_str = self.trade_date[date_now < self.trade_intra_date][0].strftime('%Y%m%d')
        next_trd_str = self.trade_date[date_now < self.trade_intra_date][1].strftime('%Y%m%d')
        last_trd_str = self.trade_date[date_now > self.trade_intra_date][-1].strftime('%Y%m%d')
        return today_str, last_trd_str, next_trd_str


def date_fun(date_now, fut_name, bar_list):
    day_begin = '08:00:00.0'
    day_end = '16:00:00.0'
    base_tools = BaseTools()
    today_str, last_trd_str, _ = base_tools.judge_trade_date(pd.to_datetime(date_now.replace('_', '')))
    last_trd_str_next = (pd.to_datetime(last_trd_str) + timedelta(days=1)).strftime('%Y%m%d')
    start_end_time_list = base_tools.get_start_end_time(fut_name.upper())  # '09:00', '10:30', '13:30', '21:00'

    def judge_time(start_end_time_list, data_time):
        res_bool = None
        for start_time, end_time in start_end_time_list:
            start_time = start_time + ':00.0'
            end_time = end_time + ':00.0'
            # print(start_time, end_time)
            if day_begin > end_time:
                tmp_con = (end_time >= data_time) | (data_time >= start_time)
            else:
                tmp_con = (end_time >= data_time) & (data_time >= start_time)
            # print(tmp_con[:5])
            if res_bool is not None:
                res_bool = res_bool | tmp_con
            else:
                res_bool = tmp_con
        return res_bool

    def get_date_range(start_end_time_list, bar):
        res_list = []
        for start_time, end_time in start_end_time_list:
            start_time = start_time + ':00.0'
            end_time = end_time + ':00.0'
            # print(start_time, end_time)
            if start_time > day_end:
                if end_time < day_begin:
                    start_date = pd.to_datetime(f'{last_trd_str} {start_time}')
                    end_date = pd.to_datetime(f'{last_trd_str_next} {end_time}')

                else:
                    start_date = pd.to_datetime(f'{last_trd_str} {start_time}')
                    end_date = pd.to_datetime(f'{last_trd_str} {end_time}')
            else:
                start_date = pd.to_datetime(f'{today_str} {start_time}')
                end_date = pd.to_datetime(f'{today_str} {end_time}')
            # print(start_date, end_date)
            time_list = list(pd.date_range(start_date, end_date, freq=f'{bar}S', closed='right'))
            res_list += time_list
        return sorted(res_list)

    def old_id_to_new(con_id_old):
        fut_id = re.sub('\d+', '', con_id_old)
        num_id = re.sub('\D+', '', con_id_old)
        if len(num_id) < 4:
            num_id = '2' + num_id
        con_id = fut_id.upper() + num_id
        return con_id


    def get_bar_info():
        bar_time_list_dict = {}
        for bar in bar_list:
            save_root_path = f'/mnt/mfs/DAT_FUT/intraday/fut_sbar/fut_{bar}sbar/{today_str}'
            bt.AZ_Path_create(save_root_path)
            bar_time_list = get_date_range(start_end_time_list, bar)
            bar_time_list_dict[bar] = [save_root_path, bar_time_list]
        return bar_time_list_dict

    bar_time_list_dict = get_bar_info()
    con_list = [x for x in sorted(os.listdir(date_path)) if re.sub('\d+', '', x.split('_')[0]) == fut_name]



    for con_file in con_list:
        con_path = f'{date_path}/{con_file}'
        con_data = pd.read_csv(con_path, sep='|', header=None)

        col_list = ['Con_id',
                    'Date',
                    'Time',
                    'Now_px',
                    'Trade_num',
                    'OI',
                    'Turnover_sum',
                    'bid_px',
                    'bid_num',
                    'ask_px',
                    'ask_num',
                    'Trade_date'
                    ]
        if len(con_data.columns) == 11:
            con_data.columns = col_list[:-1]
            a = False
        else:
            con_data.columns = col_list
            a = True

        con_data.index = pd.to_datetime(con_data['Time'].apply(lambda x:
                                                               pd.to_datetime(f'{today_str} {x}')
                                                               if day_begin < x < day_end else
                                                               (pd.to_datetime(f'{last_trd_str} {x}') if x > day_end
                                                                else
                                                                pd.to_datetime(f'{last_trd_str_next} {x}'))))

        con_data = con_data[~con_data.index.duplicated('last')]
        # 剔除非交易时间段数据
        con_data = con_data[judge_time(start_end_time_list, con_data['Time'])]


        # 根据bar_time_list获得re bar data
        if len(con_data) > 0:
            for bar in bar_list:
                save_root_path, bar_time_list = bar_time_list_dict[bar]
                rebar_data = con_data.reindex(index=con_data.index | bar_time_list).ffill().loc[bar_time_list]
                rebar_data['Volume'] = rebar_data['Trade_num'] - rebar_data['Trade_num'].shift(1).fillna(0)
                rebar_data['Turnover'] = rebar_data['Turnover_sum'] - rebar_data['Turnover_sum'].shift(1).fillna(0)
                if a:
                    rebar_data.drop(columns=['Trade_num', 'Turnover_sum', 'Trade_date'], inplace=True)
                else:
                    rebar_data.drop(columns=['Trade_num', 'Turnover_sum'], inplace=True)
                con_id = old_id_to_new(con_file.split('_')[0])
                rebar_data.to_csv(f'{save_root_path}/{con_id}', sep='|', index_label='Trade_time')


if __name__ == '__main__':

    bar_list = [5, 10, 20, 30]
    for date_now in date_list:
        date_path = f'{tick_path}/{date_now}'
        fut_name_list = list(set([re.sub('\d+', '', x.split('_')[0]) for x in sorted(os.listdir(date_path))]))
        # print(fut_name_list)
        for fut_name in fut_name_list:
            try:
                date_fun(date_now, fut_name, bar_list)
                # print(date_now, fut_name, "Deal")
            except Exception as e:
                print('_____________________________')
                print(date_now, fut_name, "Error")
                print(e)
