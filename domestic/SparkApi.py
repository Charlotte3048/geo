import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
import time
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import websocket  # 使用websocket_client


class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.Spark_url + '?' + urlencode(v)
        return url


class SparkSyncClient:
    def __init__(self):
        self.answer = ""
        self.sid = ""

    def on_message(self, ws, message):
        """处理 WebSocket 消息"""
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            print("=== 星火返回错误 ===")
            print(json.dumps(data, ensure_ascii=False, indent=2))
            ws.close()
            return

        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        self.answer += content

        # status == 2 表示最后一条消息
        if status == 2:
            ws.close()

    def chat(self, appid, api_key, api_secret, Spark_url, domain, question):
        """同步调用星火 API"""
        self.answer = ""  # 重置答案
        wsParam = Ws_Param(appid, api_key, api_secret, Spark_url)
        websocket.enableTrace(False)
        wsUrl = wsParam.create_url()

        # 绑定实例方法
        ws = websocket.WebSocketApp(
            wsUrl,
            on_message=self.on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # 传递参数
        ws.appid = appid
        ws.question = question
        ws.domain = domain

        # 运行并阻塞
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

        return self.answer


# 收到websocket错误的处理
def on_error(ws, error):
    print("### Spark WebSocket error:", error)


# 收到websocket关闭的处理
def on_close(ws, one, two):
    pass  # 静默关闭


# 收到websocket连接建立的处理
def on_open(ws):
    thread.start_new_thread(run, (ws,))


def run(ws, *args):
    """发送请求数据"""
    data = json.dumps(gen_params(appid=ws.appid, domain=ws.domain, question=ws.question))
    ws.send(data)


def gen_params(appid, domain, question):
    """
    通过appid和用户的提问来生成请参数
    """
    # ⭐ 关键修改：将 question 转换为消息数组格式
    if isinstance(question, str):
        # 如果是字符串，转换为消息数组
        messages = [
            {"role": "user", "content": question}
        ]
    else:
        # 如果已经是数组，直接使用
        messages = question

    data = {
        "header": {
            "app_id": appid,
            "uid": "1234"
        },
        "parameter": {
            "chat": {
                "domain": domain,
                "temperature": 0.8,
                "max_tokens": 2048,
                "top_k": 5,
                "auditing": "default"
            }
        },
        "payload": {
            "message": {
                "text": messages  # ← 改为数组
            }
        }
    }
    return data
