from datetime import datetime
import requests
import urllib
# import urllib2
from bs4 import BeautifulSoup
import time
import os
import sys

sys.path.append('/mnt/mfs')
from work_dmgr_fut.loc_lib.pre_load import *

# import cookielib


def AZ_Path_create(target_path):
    """
    添加新路径
    :param target_path:
    :return:
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)


class SpiderData:
    pass


def deal_link(title, link_path):
    try:
        wb_info = requests.get(link_path)
        soup = BeautifulSoup(wb_info.text, 'lxml')

        time_source = soup.select('body div["class"="main_left"] div["class"="left-content"] '
                                  'div["class"="Info"] div["class"="time-source"] ')[0]

        news_time = time_source.select('div["class"="time"]')[0].text.replace('年', '-') \
            .replace('月', '-').replace('日', '')
        w_time = datetime.now().strftime('%Y-%m-%d %H:%M')

        save_day_path = f'{save_root_path}/{news_time[:10]}'
        AZ_Path_create(save_day_path)

        data_source = time_source.select('div["class"="source data-source"]')[0]['data-source']
        print(news_time, data_source)

        save_str_path = f'{save_day_path}/{data_source}'
        save_ma_str_path = f'/mnt/mfs/dat_whs/甲醛/temp/{data_source}'
        # AZ_Path_create(save_str_path)

        main_info = soup.select('body div["class"="main_left"] div["class"="left-content"] div["class"="Body"]')[0]
        main_str = main_info.text.replace('\n', '').replace('\u3000', '').replace('\r', '').replace(' ', '')
        wr_str = '|'.join([title, w_time, news_time, link_path, main_str]) + '\n'
        with open(save_str_path, 'a') as file:
            file.write(wr_str)
        if '甲醇' in title:
            with open(save_ma_str_path, 'a') as file:
                file.write(wr_str)

    except Exception as error:
        print(error)


def update_fun(last_date):
    with open(save_info_path, 'a') as file:
        for j in range(1, 26):
            print('___________________________')

            if j == 1:
                download_page = f'http://futures.eastmoney.com/news/cnpbb.html'
            else:
                download_page = f'http://futures.eastmoney.com/news/cnpbb_{j}.html'
            print(f'now in page {j}')
            print(download_page)
            wb_data = requests.get(download_page)
            soup = BeautifulSoup(wb_data.text, 'lxml')
            for i in range(20):
                try:
                    a = soup.select(f'body div["class"="repeatList"] ul["id"="newsListContent"] '
                                    f'li["id"="newsTr{i}"] div["class"="text text-no-img"]')[0]

                    link_path = a.select('a["href"]')[0]['href']
                except Exception as error:
                    print(error)
                    a = soup.select(f'body div["class"="repeatList"] ul["id"="newsListContent"] '
                                    f'li["id"="newsTr{i}"] div["class"="text"]')[0]

                    link_path = a.select('a["href"]')[0]['href']

                news_time = a.select('p["class"="time"]')[0].text.replace(' ', '').replace('\r', '').replace('\n', '')

                news_time_list = news_time.replace('月', ' ').replace('日', ' ').replace(':', ' ').split(' ')
                news_date = datetime(datetime.now().year, *[int(x) for x in news_time_list])
                try:
                    if j == 1 and i == 0:
                        if last_date >= news_date:
                            return_time = last_date
                        else:
                            return_time = news_date

                    if last_date > news_date:
                        return return_time
                except Exception as error:
                    print(error)
                    continue

                news_name = a.select('a["href"]')[0].text.replace(' ', '').replace('\r', '').replace('\n', '')
                wr_str = f'{news_name}|{link_path}\n'
                file.write(wr_str)
                deal_link(news_name, link_path)
        return return_time


def main():
    wait_m = 1
    last_date = datetime(2019, 9, 4, 1, 7)

    while True:
        print('__________________________________________')
        print(f'update {last_date}')
        t1 = time.time()
        last_date = update_fun(last_date)
        t2 = time.time()
        print(t2 - t1)
        print(last_date)
        print(f'wait {wait_m} minute next update')
        time.sleep(wait_m * 60)


if __name__ == '__main__':
    save_root_path = '/mnt/mfs/temp/spider_data'
    save_info_path = f'{save_root_path}/choice_info'
    bt.AZ_Path_create(save_root_path)
    main()

    # title = '2019年07月11日工业品期货主约榜'
    # link_path = 'http://futures.eastmoney.com/a/201907121176811821.html'
