from loc_lib.pre_load import *

alpha_pos_path = '/mnt/mfs/AAPOS'
pos_file_list = [x for x in os.listdir(alpha_pos_path) if x.startswith('WHS')]
return_df = pd.read_csv('/media/hdd1/DAT_EQT/EM_Funda/DERIVED_14/aadj_r.csv', index_col=0, sep='|', parse_dates=True)
target_date = return_df.index[-1]
info_list = []
for x in pos_file_list:
    pos_df = pd.read_csv(os.path.join(alpha_pos_path, x), index_col=0, sep='|', parse_dates=True)
    if target_date != pos_df.index[-1]:
        info_list += [f'{x[:-4]} today position have not update!']

info_str = '<br /><br />'.join(info_list)
send_email.send_email(info_str, ['whs@yingpei.com'], [], f'[ALPHA_POS_INFO]{datetime.now()}')

