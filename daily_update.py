#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import logging

import pandas as pd

from calculate_percentile import calculate_percentile
from lixinger import *

logging.basicConfig(level=logging.INFO)

pd.set_option('display.width', 1000)
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def split_list(original_list, chunk_size=90):
    # 使用列表推导式和切片将列表分割
    return [original_list[i:i + chunk_size] for i in range(0, len(original_list), chunk_size)]


def fetch_all_data(date, stock_codes):
    result = []
    for stock_code_list in split_list(stock_codes):
        fetch = fetch_data(date, stock_code_list)
        result.extend(fetch)
    return result


def fetch_data(date, stock_codes):
    fetch = query_json(url_suffix="cn/index/fundamental",
                       query_params={
                           "date": date,
                           "stockCodes": stock_codes,
                           "metricsList": [
                               "pe_ttm.mcw",
                               "ps_ttm.mcw",
                               "pb.mcw",
                               "dyr.mcw",
                               "mc",
                               "cpc",
                               "to_r",
                               "sb",
                               "fb"
                           ]
                       }
                       )
    if fetch['message'] != "success":
        raise Exception
    return fetch["data"]


def update_data(data, stock_code):
    new_df = pd.json_normalize(data)
    new_df['date'] = pd.to_datetime(new_df['date']).dt.tz_convert(None)
    with open(AMENTAL_DIR.joinpath(f"{stock_code}.json"), "r", encoding="utf-8") as f:
        old_df = pd.read_json(f, orient="records")

    # if new_df['date'].iloc[-1] == old_df['date'].iloc[-1]:
    #     return None
    df_concat = pd.concat([old_df, new_df], ignore_index=True)
    # print(df_concat[-5:])

    df_unique = df_concat.drop_duplicates(subset=['date'], keep='last')
    df_sorted = df_unique.sort_values(by=['date'], ascending=True)

    df_calculate = calculate_percentile(df_sorted)

    with open(AMENTAL_DIR.joinpath(f"{stock_code}.json"), 'w', encoding="utf-8") as f:
        df_calculate.to_json(f, orient="records", indent=4)


def main():
    stockCodes = [file.stem for file in AMENTAL_DIR.glob('*.json')]
    d = datetime.datetime.now().strftime("%Y-%m-%d")
    data_all = fetch_all_data(d, stockCodes)
    total = len(data_all)
    i = 1
    for data in data_all:
        stock_code = data["stockCode"]
        update_data(data, stock_code)
        logging.info(f"[{i}/{total}] {stock_code} 更新成功")
        i += 1
    from html_generator import main as html_generator
    html_generator()
    logging.info("更新完成")


if __name__ == '__main__':
    main()
