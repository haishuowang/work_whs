import sys

sys.path.append('/mnt/mfs')
from work_whs.loc_lib.pre_load import *
from work_whs.loc_lib.pre_load import log
from work_whs.loc_lib.pre_load.plt import savfig_send
from work_whs.test_future.FutDataLoad import FutData, FutClass
from work_whs.test_future.signal_fut_fun import FutIndex, Signal, Position

fut_data = FutData()


def plot_send(con_id, begin_time, end_time, n=15):
    data_df = fut_data.load_intra_data(con_id, ['Close'])['Close']
    part_df = data_df.truncate(begin_time, end_time)
    plt.figure(figsize=[80, 10])
    plt.plot(range(len(part_df)), part_df.values)
    plt.xticks([int(len(part_df.index.strftime('%Y-%m-%d %H:%M')) / (n-1)) * i for i in range(n)],
               part_df.index.strftime('%Y-%m-%d %H:%M')[[[int(len(part_df.index) / (n-1)) * i for
                                                          i in range(n)]]], rotation=45)
    plt.grid()
    savfig_send()


plot_send('I2001.DCE', datetime(2019, 8, 10), datetime(2019, 11, 28), 80)
