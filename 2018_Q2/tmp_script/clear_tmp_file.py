import os


def delete_all_file(root_path, except_list):
    file_list = os.listdir(root_path)
    file_list = list(set(file_list) - set(except_list))
    for file_name in sorted(file_list):
        os.remove(os.path.join(root_path, file_name))


if __name__ == '__main__':
    log_root_path = '/media/hdd0/whs/result/log'
    result_root_path = '/media/hdd0/whs/result/result'
    para_root_path = '/media/hdd0/whs/result/para'
    except_list = ['20180614_1847.txt']

    delete_all_file(log_root_path, except_list)
    delete_all_file(result_root_path, except_list)
    delete_all_file(para_root_path, except_list)


