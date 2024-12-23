import os
import requests
import re
import pathlib

__all__ = ["sc_send"]
def get_sendkey():
    """
    获取token
    """
    token = os.getenv("FT_SENDKEY")
    if token is not None:
        return token
    token_cfg = pathlib.Path(__file__).parent.joinpath("sendkey.cfg")
    if not token_cfg.exists():
        token_cfg.touch()
        raise Exception("请先在当前目录下创建token.cfg文件，并写入token")
    token = token_cfg.read_text().strip()
    if token is None:
        raise Exception("请先在当前目录下创建token.cfg文件，并写入token")
    return token


def sc_send(title, desp='', options=None):
    if options is None:
        options = {}

    if get_sendkey() is None:
        raise Exception("token未设置")

    sendkey = get_sendkey()

    # 判断 sendkey 是否以 'sctp' 开头，并提取数字构造 URL
    if sendkey.startswith('sctp'):
        match = re.match(r'sctp(\d+)t', sendkey)
        if match:
            num = match.group(1)
            url = f'https://{num}.push.ft07.com/send/{sendkey}.send'
        else:
            raise ValueError('Invalid sendkey format for sctp')
    else:
        url = f'https://sctapi.ftqq.com/{sendkey}.send'
    params = {
        'title': title,
        'desp': desp,
        **options
    }
    headers = {
        'Content-Type': 'application/json;charset=utf-8'
    }
    response = requests.post(url, json=params, headers=headers)
    result = response.json()
    return result



