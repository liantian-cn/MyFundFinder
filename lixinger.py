#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import pathlib

import requests

BASEURL = "https://open.lixinger.com/api/"

__all__ = ["query_json"]


def get_token():
    """
    获取token
    """
    token = os.getenv("LIXINGER_TOKEN")
    if token is not None:
        return token
    token_cfg = pathlib.Path(__file__).parent.joinpath("token.cfg")
    if not token_cfg.exists():
        token_cfg.touch()
        raise Exception("请先在当前目录下创建token.cfg文件，并写入token")
    token = token_cfg.read_text().strip()
    if token is None:
        raise Exception("请先在当前目录下创建token.cfg文件，并写入token")
    return token


def get_full_url(url_suffix):
    url_suffix = url_suffix.replace('.', '/')
    if url_suffix.startswith('/'):
        url_suffix = url_suffix[1:]
    return BASEURL + url_suffix


def query_json(url_suffix, query_params=None):
    """
    API接口，返回json结构
    params:
        url_suffix: api地址后缀, https://open.lixinger.com/api/ 之后的，可以用/或. 例如a/stock/fs或a.stock.fs
        query_params: API的查询json，不需要填token
    """
    if query_params is None:
        query_params = dict()
    if get_token() is None:
        raise Exception("token未设置")
    query_params["token"] = get_token()

    headers = {"Content-Type": "application/json"}
    response = requests.post(url=get_full_url(url_suffix), data=json.dumps(query_params), headers=headers)
    return response.json()
