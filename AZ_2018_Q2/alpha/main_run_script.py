import os
import pandas as pd

data = pd.read_csv('/mnt/mfs/alpha_whs/alpha.config', sep='|')

alpha_list = data['alpha_id']
calc_list = data['alpha_calc']

python_path = '/opt/anaconda3_beta/bin/python'
for calc_script in set(calc_list):
    print(calc_script)
    bashCommand = f"{python_path} /mnt/mfs/alpha_whs/{calc_script}"
    os.system(bashCommand)

for alpha_script in alpha_list:
    print(alpha_script)
    bashCommand = f"{python_path} /mnt/mfs/alpha_whs/{alpha_script}"
    os.system(bashCommand)
