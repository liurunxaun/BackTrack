from sparkai.llm.llm import ChatSparkLLM
from websocket import create_connection
import time
import json
from time import sleep
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage

# 星火认知大模型的配置
SPARKAI_URL_4_0 = 'wss://spark-api.xf-yun.com/v4.0/chat'

SPARKAI_APP_ID = '06023e43'
SPARKAI_API_SECRET = 'OTE2ODNhMjY1YmZhZjZlYjRlM2FiZGI0'
SPARKAI_API_KEY = '7691129081cbdafccffe856aac0af6fd'
SPARKAI_DOMAIN_4_0 = '4.0Ultra'

SPARKAI_URL_lb = "wss://spark-api.xf-yun.com/v3.5/chat"
SPARKAI_APP_ID_lb = '2597017a'
SPARKAI_API_SECRET_lb = 'NzM2ZGZhMjZkMDM2NDg1NDk0YjZlYWVj'
SPARKAI_API_KEY_lb = '74950dd6f4aede6248158f8cff2f8f21'
SPARKAI_DOMAIN_lb = 'generalv3.5'
spark3_5 = ChatSparkLLM(
    spark_api_url=SPARKAI_URL_lb,
    spark_app_id=SPARKAI_APP_ID_lb,
    spark_api_key=SPARKAI_API_KEY_lb,
    spark_api_secret=SPARKAI_API_SECRET_lb,
    spark_llm_domain=SPARKAI_DOMAIN_lb,
    streaming=False,
)
spark4_0 = ChatSparkLLM(
    spark_api_url=SPARKAI_URL_4_0,
    spark_app_id=SPARKAI_APP_ID,
    spark_api_key=SPARKAI_API_KEY,
    spark_api_secret=SPARKAI_API_SECRET,
    spark_llm_domain=SPARKAI_DOMAIN_4_0,
    streaming=False,
)


def spark_4_0(query):
    messages = [ChatMessage(
        role="user",
        content=query
    )]

    handler = ChunkPrintHandler()
    response = spark4_0.generate([messages], callbacks=[handler])
    return response.generations[0][0].text

def spark_3_5(query):
    messages = [ChatMessage(
        role="user",
        content=query
    )]

    handler = ChunkPrintHandler()
    response = spark3_5.generate([messages], callbacks=[handler])
    return response.generations[0][0].text

def spark_4_0_company(query, ip_port='10.43.108.62:8678'):
    """
    封装的函数，用于通过 WebSocket 查询信息，并统计首字符的响应时间

    参数:
        query (str): 查询的内容，例如 "北京市今天天气怎么样？"
        ip_port (str): WebSocket服务器的IP和端口号，默认为 '10.43.108.62:8678'

    返回:
        tuple: 查询的完整结果 (str)，首字符的响应时间 (float)
    """
    url = 'ws://{}/get_res/'.format(ip_port)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.80 Safari/537.36',
        'Cookie': 'miid=46129921999119747;bid=9;_uab_collina=155930245139965167817273;'
    }
    # 模板字符串
    template = """
       {input_text}
       """
    text = template.format(input_text=query)
    # 准备数据
    data = {
        'query': text,
        'user_id': 'yjyang18',
    }

    # 打开 WebSocket 连接并发送数据
    wss = create_connection(url, header=headers, timeout=10)
    wss.send(json.dumps(data, ensure_ascii=False))
    try:

        final_response = ""  # 用于累积所有的response
        recv = True
        first_response_time = None  # 记录首字符响应时间
        start_time = time.time()  # 请求发送的时间

        while recv:
            data = wss.recv()
            # 检查是否收到空数据
            if not data:
                print("收到空数据，继续接收下一条数据...")
                continue  # 跳过本次循环，继续接收
            t = json.loads(data)
            response = t["response"]

            # 如果是第一次接收到数据，记录首字符响应时间
            if first_response_time is None:
                first_response_time = time.time() - start_time

            final_response += response  # 累积每次收到的response
            # 如果收到 '<end>'，则停止接收
            if response == '<end>':
                recv = False

    finally:
        wss.close()  # 确保连接被关闭

    # 返回完整的累积答案，去掉最后的 <end> 标记，以及首字符的响应时间
    return final_response.replace('<end>', '')
