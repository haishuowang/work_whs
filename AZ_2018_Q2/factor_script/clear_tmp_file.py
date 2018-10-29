import os


def delete_all_file(root_path, except_list=None):
    if except_list is None:
        except_list = []
    file_list = os.listdir(root_path)
    file_list = list(set(file_list) - set(except_list))
    for file_name in sorted(file_list):
        os.remove(os.path.join(root_path, file_name))


if __name__ == '__main__':
    log_root_path = '/mnt/mfs/dat_whs/result/log'
    result_root_path = '/mnt/mfs/dat_whs/result/result'
    para_root_path = '/mnt/mfs/dat_whs/result/para'
    # except_list = ['20180626_1047.txt', '20180626_1106.txt', '20180626_1148.txt', '20180626_1224.txt',
    #                '20180626_1555.txt']
    except_list = ['market_top_100_False_20180706_1514.txt',
                   'market_top_100_True_20180706_1515.txt',
                   'market_top_500_False_20180706_1201.txt',
                   'market_top_500_True_20180706_1142.txt']

    # delete_all_file(log_root_path, except_list)
    # delete_all_file(result_root_path, except_list)
    # delete_all_file(para_root_path, except_list)
    delete_all_file('/mnt/mfs/dat_whs/tmp_figure', except_list=None)
