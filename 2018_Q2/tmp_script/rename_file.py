import os


def rename_fun(root_path):
    file_list = os.listdir(root_path)
    for file_name in file_list:
        new_name = file_name[:-3] + '.pkl'
        os.rename(os.path.join(root_path, file_name), os.path.join(root_path, new_name))


if __name__ == '__main__':
    root_path = '/mnt/mfs/DAT_EQT/EM_Tab14/raw_data/TRAD_SK_REVALUATION/split_data'
    rename_fun(root_path)
