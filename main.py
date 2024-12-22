#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from functools import wraps
import datetime
import logging
import json
import pickle
import pytz
import pathlib
import math

import pandas as pd
from datetime import datetime, timedelta, timezone
from jinja2 import Environment, FileSystemLoader

from lixinger import *

BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR.joinpath("data")
DAILY_DIR = BASE_DIR.joinpath("daily")
OUTPUT_DIR = BASE_DIR.joinpath("output")

UPDATE_FILE = BASE_DIR.joinpath("update.json")
UPDATE_DAYS = 7

# 屏蔽的指数
BLOCK_INDEX = [
    "000917",  # 300公用，交易日期太少
    "1000002",  # A股全指,无基金
    "1000003",  # 深圳A股,无基金
    "1000004",  # 上海A股,无基金
    "1000005",  # 上海非金融A股,无基金
    "1000007",  # 创业板全指,无基金
    "1000008",  # 沪深B股,无基金
    "1000009",  # 深圳B股,无基金
    "1000010",  # 上海B股,无基金
    "1000011",  # A股非金融,无基金
    "1000011",  # A股非金融,无基金
    "1000012",  # 中小板全指,无基金
    "1000014",  # 科创板全指,无基金
    "H50040",  # 上红低波,无基金
    "931845",  # 生猪产业,无基金,不看好
    "399983",  # 地产等权,不看好
    "399393",  # 国证地产,不看好
    "931775",  # 房地产,不看好
    "399965",  # 800地产,不看好
    "931802",  # 中证龙头,无基金
    "399314",  # 巨潮大盘,无基金
    "399315",  # 巨潮中盘,无基金
    "399316",  # 巨潮小盘,无基金
    "000842",  # 800等权,范围太大不明确
    "399311",  # 国证1000,范围太大不明确
    "931247",  # 中证信创 交易日小于2500
    "931574",  # 港股科技 交易日小于2500
    "931008",  # 汽车指数 交易日小于2500
    "931009",  # 建筑材料 交易日小于2500
    "000857",  # 500医药 交易日小于2500
    "931573",  # HKC科技 交易日小于2500
    "399806",  # 环境治理 交易日小于2500
    "399814",  # 大农业 交易日小于2500
    "399996",  # 智能家居 交易日小于2500
    "399417",  # 新能源车 交易日小于2500
    "000852",  # 中证1000 交易日小于2500
    "399989",  # 中证医疗 交易日小于2500
    "399976",  # CS新能车 交易日小于2500
    "399987",  # 中证酒 交易日小于2500

]

logging.basicConfig(level=logging.INFO)
pd.set_option('display.width', 1000)
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

# 布林线时间
BOLL_PERIOD = 20


def write_current_time_to_json():
    """将当前时间写入 JSON 文件"""
    current_time = datetime.now().isoformat()
    with open(UPDATE_FILE, 'w') as f:
        json.dump({'last_update': current_time}, f)


def read_last_update_time():
    """读取 JSON 文件中的最后更新时间"""
    if not UPDATE_FILE.exists():
        return None
    with open(UPDATE_FILE, 'r') as f:
        data = json.load(f)
        return datetime.fromisoformat(data.get('last_update'))


def should_skip_execution():
    """检查是否应该跳过执行"""
    last_update_time = read_last_update_time()
    if last_update_time is None:
        return False  # 如果没有更新记录，则不跳过
    current_time = datetime.now()
    time_difference = current_time - last_update_time
    return time_difference.days < UPDATE_DAYS


def retry(max_attempts=3, delay=1):
    """
    装饰器：在函数执行失败时自动重试。

    :param max_attempts: 最大重试次数
    :param delay: 每次重试之间的延迟（秒）
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logging.info(f"第 {attempts} 次尝试失败: {e}")
                    if attempts < max_attempts:
                        time.sleep(delay)
            logging.info("所有尝试均失败，抛出最后的异常。")
            raise e  # 抛出最后一次异常

        return wrapper

    return decorator


@retry(max_attempts=5, delay=5)
def fetch_index():
    """
    获取所有中证指数信息，并保存到index.json
    :return:
    """
    fetch = query_json("cn/index", {})
    if fetch['message'] != "success":
        raise Exception
    index_data = fetch["data"]
    with open(DATA_DIR.joinpath("index_origin.pickle"), 'wb') as f:
        pickle.dump(index_data, f)


def filter_index():
    """
    读取index.json
    过滤掉不需要的指数
    过滤标准如下：
        -- 未满10年的指数
    最后保存到index.pickle
    :return:
    """
    with open(DATA_DIR.joinpath("index_origin.pickle"), 'rb') as f:
        index_data = pickle.load(f)

    logging.info(f'未过滤指数总数为:{len(index_data)}')

    result = []
    #
    today = datetime.now(pytz.timezone("Asia/Shanghai"))
    #
    for index in index_data:
        if isinstance(index["launchDate"], str):
            index["launchDate"] = datetime.fromisoformat(index["launchDate"])

        # 对未满十年的进行过滤
        if (today - index["launchDate"]).days < 365 * 10:
            continue

        if index["stockCode"] in BLOCK_INDEX:
            continue
        result.append(index)
    #
    logging.info(f'过滤指数总数为:{len(result)}')

    # 排序
    result.sort(key=lambda x: x['launchDate'])
    #
    with open(DATA_DIR.joinpath("index.pickle"), 'wb') as f:
        pickle.dump(result, f)


def get_dates_ranges(launch_datetime, end_datetime):
    """
    根据开始日期和发射日期计算日期范围列表。

    参数:
    start_datetime: datetime对象，表示开始日期。
    launch_datetime: datetime对象，表示发射日期。

    返回:
    一个包含日期范围元组的列表，每个元组包含两个格式为"%Y-%m-%d"的字符串，分别表示范围的开始和结束日期。

    抛出:
    ValueError: 如果发射日期晚于开始日期，则抛出此异常。
    """
    # 检查发射日期是否早于开始日期，如果不是，则抛出ValueError
    if launch_datetime > end_datetime:
        raise ValueError("Launch date must be before start date")

    date_ranges = []
    # 当发射日期小于开始日期时，循环继续
    while launch_datetime < end_datetime:
        # 计算结束日期，确保不超过开始日期，且最多增加10年
        _end_datetime = min(launch_datetime + timedelta(days=365 * 9), end_datetime)
        # 将开始日期和结束日期格式化为字符串，并添加到日期范围列表中
        date_ranges.append((launch_datetime.strftime("%Y-%m-%d"), _end_datetime.strftime("%Y-%m-%d")))
        # 将启示期更新为结束日期下一天，为下一次循环做准备
        launch_datetime = _end_datetime + timedelta(days=1)

    return date_ranges


@retry(max_attempts=5, delay=5)
def fetch_index_fundamental(index: dict):
    """
    获取一个指数的基本面数据 ，并保存到对应指数代码的pickle文件

    :param index:一个指数对象
    :return:
    """

    stock_code = index["stockCode"]
    launch_datetime = index["launchDate"]
    end_datetime = datetime.now(pytz.timezone("Asia/Shanghai"))

    result = []

    # 将日期分组
    date_ranges = get_dates_ranges(launch_datetime, end_datetime)

    for start, end in date_ranges:
        fetch = query_json(url_suffix="cn/index/fundamental",
                           query_params={
                               "startDate": start,
                               "endDate": end,
                               "stockCodes": [stock_code, ],
                               "metricsList": [
                                   "pe_ttm.mcw",  # 滚动市盈率(市值加权)
                                   "ps_ttm.mcw",  # 滚动市销率(市值加权)
                                   "pb.mcw",  # 市净率(市值加权)
                                   "dyr.mcw",  # 股息率(市值加权)
                                   "mc",  # 市值
                                   "cp",  # 收盘点位
                                   "to_r",  # 换手率
                                   "sb",  # 融券余额
                                   "fb"  # 融资余额
                               ]
                           }
                           )
        if fetch['message'] != "success":
            raise Exception
        result.extend(fetch["data"])

    for record in result:
        record["date"] = datetime.fromisoformat(record["date"])

    result.sort(key=lambda x: x['date'])
    df = pd.DataFrame(result)
    # df.set_index(["date"], inplace=True)
    df = df.sort_values(by='date')
    with open(DAILY_DIR.joinpath(f"{stock_code}.pickle"), 'wb') as f:
        pickle.dump(df, f)


def fetch_index_fundamental_all():
    """
    获取所有指数的基本面数据
    :return:

    """

    with open(DATA_DIR.joinpath("index.pickle"), 'rb') as f:
        index_data = pickle.load(f)
    for index in index_data:
        logging.info(f"fetch {index['stockCode']} data")
        fetch_index_fundamental(index)


@retry(max_attempts=5, delay=5)
def fetch_index_tracking_fund(index):
    """获取一个指数的场内基金"""
    stock_code = index["stockCode"]
    fetch = query_json(
        url_suffix="cn/index/tracking-fund",
        query_params={"stockCode": stock_code}
    )
    # print(fetch)
    if fetch['message'] != "success":
        raise Exception

    return fetch["data"]


def fetch_index_tracking_fund_all():
    """获取所有指数的场内基金"""
    with open(DATA_DIR.joinpath("index.pickle"), 'rb') as f:
        index_data = pickle.load(f)
    for index in index_data:
        logging.info(f"fetch {index['stockCode']} tracking fund")
        index["tracking_fund"] = fetch_index_tracking_fund(index)
    with open(DATA_DIR.joinpath("index.pickle"), 'wb') as f:
        pickle.dump(index_data, f)


def calculate_index(index):
    """计算一个指数的数据"""
    stock_code = index["stockCode"]
    with open(DAILY_DIR.joinpath(f"{stock_code}.pickle"), 'rb') as f:
        df = pickle.load(f)
    row_count = df.shape[0]
    if row_count < 2500:
        logging.info(f'\"{stock_code}\" # {index["name"]} 交易日小于2500')

    # 市净率(历史百分位)
    # df['pb_percentile'] = df['pb.mcw'].expanding().apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    df['pb_percentile'] = df['pb.mcw'].rolling(window=3000, min_periods=1).apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    # 滚动市盈率(历史百分位)
    # df['pe_percentile'] = df['pe_ttm.mcw'].expanding().apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    df['pe_percentile'] = df['pe_ttm.mcw'].rolling(window=3000, min_periods=1).apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    # 股息率(历史百分位)
    # df['dyr_percentile'] = df['dyr.mcw'].expanding().apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    df['dyr_percentile'] = df['dyr.mcw'].rolling(window=3000, min_periods=1).apply(lambda x: x.rank(method='min', pct=True).iloc[-1])
    # 滚动市销率(历史百分位)
    # df['ps_percentile'] = df['ps_ttm.mcw'].expanding().apply(lambda x: x.rank(method='min', pct=True).iloc[-1])

    # 布林线均线
    df['SMA'] = df['cp'].rolling(window=BOLL_PERIOD).mean()
    # 布林线标准差
    df['STD'] = df['cp'].rolling(window=BOLL_PERIOD).std()
    # 布林线上轨
    df["UpperBB"] = df['SMA'] + (2 * df['STD'])
    # 布林线下轨道
    df["LowerBB"] = df['SMA'] - (2 * df['STD'])
    # 布林线位置
    df['boll_percentile'] = (df['cp'] - df['LowerBB']) / (df['UpperBB'] - df['LowerBB'])
    #
    # print(df)

    with open(DAILY_DIR.joinpath(f"{stock_code}.pickle"), 'wb') as f:
        pickle.dump(df, f)


def calculate_index_all():
    """计算所有指数的数据"""
    with open(DATA_DIR.joinpath("index.pickle"), 'rb') as f:
        index_data = pickle.load(f)
    for index in index_data:
        logging.info(f"calculate  {index['stockCode']} data")
        # fetch_index_fundamental(i)
        calculate_index(index)


def float_to_percentile(value):
    if value is None:
        return "--"
    return f"{value * 100:.2f}%"


def float_to_2f(value):
    if value is None:
        return "--"
    return f"{value:.2f}"


def get_latest_record(stock_code):
    with open(DAILY_DIR.joinpath(f"{stock_code}.pickle"), 'rb') as f:
        df = pickle.load(f)
    result = df.to_dict('records')[-1]
    result["dyr"] = result["dyr.mcw"]
    result["pb"] = result["pb.mcw"]
    result["pe_ttm"] = result["pe_ttm.mcw"]
    result["ps_ttm"] = result["ps_ttm.mcw"]
    return result


def convert_score(score):
    # 限制输入范围
    if score < 0:
        score = 0
    elif score > 100:
        score = 100

    # 将浮点数转换为整数并计算反转值
    return int(100 - score)


def get_china_standard_time():
    # 创建中国标准时间的时区
    cst = timezone(timedelta(hours=8))
    # 获取当前时间并转换为中国标准时间
    china_time = datetime.now(cst)
    return china_time.strftime('%Y-%m-%d %H:%M:%S')


def html_generator():
    with open(DATA_DIR.joinpath("index.pickle"), 'rb') as f:
        index_data = pickle.load(f)

    for index in index_data:
        # print(index["stockCode"])
        index["latest_record"] = get_latest_record(index["stockCode"])
        if math.isnan(index["latest_record"]["pb_percentile"]) or math.isnan(
                index["latest_record"]["dyr_percentile"]) or math.isnan(
                index["latest_record"]["pe_percentile"]) or math.isnan(index["latest_record"]["boll_percentile"]):
            index["sorted"] = 0
        else:
            index["sorted"] = int(convert_score(index["latest_record"]["pb_percentile"] * 100) * 1.0 +
                                  convert_score(index["latest_record"]["pe_percentile"] * 100) * 1.0 +
                                  convert_score(index["latest_record"]["boll_percentile"] * 100) * 1.0 +
                                  int(index["latest_record"]["dyr_percentile"] * 100) * 1.0 +
                                  int(index["latest_record"]["dyr"] * 2000)
                                  )
        index["inside_fund"] = list(filter(lambda x: (x["exchange"] in ["sz", "sh"]), index["tracking_fund"]))
        # if index["stockCode"] == "000819":
        #     import pprint
        #     pprint.pprint(index)


    index_data.sort(key=lambda x: x['sorted'], reverse=True)
    # 创建一个Jinja2环境
    env = Environment(loader=FileSystemLoader('.'))
    # 加载模板
    template = env.get_template('template.html')

    output = template.render(index_data=index_data, now=get_china_standard_time())

    # 将渲染后的结果写入输出文件
    with open('output/index.html', 'w', encoding="utf-8") as file:
        file.write(output)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    DAILY_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    if should_skip_execution():
        logging.info("跳过初始化")
    else:
        logging.info("执行初始化")
        logging.info("获取全部指数")
        fetch_index()
        logging.info("过滤")
        filter_index()
        logging.info("获取指数基金信息")
        fetch_index_tracking_fund_all()
    logging.info("写入时间戳")
    write_current_time_to_json()
    # logging.info("获取基本面信息")
    fetch_index_fundamental_all()
    calculate_index_all()
    html_generator()


if __name__ == '__main__':
    main()
