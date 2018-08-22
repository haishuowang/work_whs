ticker = '002272.SZ'
begin_date = pd.to_datetime('20050101')
a = (daily_vwap_adj_return[ticker][daily_vwap_adj_return.index > begin_date] + 1)
b = (aadj_r[ticker][aadj_r.index > begin_date] + 1)
a_prod = a.cumprod()
b_prod = b.cumprod()
dif = (a_prod - b_prod) / b_prod
print(dif)

begin_date = pd.to_datetime('20050101')
a = (daily_vwap_adj_return[daily_vwap_adj_return.index > begin_date] + 1)
b = (aadj_r[aadj_r.index > begin_date] + 1)
a_prod = a.cumprod()
b_prod = b.cumprod()
dif = (a_prod - b_prod) / b_prod

data = pd.read_table('/mnt/mfs/DAT_EQT/EM_Tab14/DERIVED_14/aadj_r_vwap.csv', sep='|', index_col=0, parse_dates=True)