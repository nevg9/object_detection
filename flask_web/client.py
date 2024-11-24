# 请求server的客户端代码，并把本地的图片文件以二进制的数据传输个python server
import requests


def send_image_to_server(image_path, url, resource_id="12341234", media_type="photo"):
    """
    将本地图片文件以二进制数据的形式传输到Python服务器。

    参数：
    image_path (str): 图片文件的本地路径。

    返回：
    requests.Response: 服务器的响应对象。
    """
    # 读取图片文件内容
    with open(image_path, 'rb') as file:
        image_content = file.read()
    # 定义请求头，包含resourceId和mediaType字段
    headers = {
        'Content-Type': 'application/octet-stream',
        'resourceId': resource_id,
        'mediaType': media_type
    }
    # 发送POST请求，将图片数据作为二进制数据传输
    response = requests.post(url, data=image_content, headers=headers)
    if response.status_code == 200:
        # 请求成功，尝试解析JSON数据
        try:
            response_data = response.json()
            # 访问解析后的数据字典中的键
            print(response_data)
        except ValueError:
            # 如果响应不是JSON格式，将引发ValueError
            print("Response content is not in JSON format.")
    else:
        # 请求失败，打印状态码和原因
        print(f"Request failed with status code {response.status_code}: {response.reason}")


if __name__ == '__main__':
    image_path = '/home/yuzhong/data1/image_classifier_data/test/human/SCNR015X-X00131-20180416-00252.jpg'
    # 空白照片识别的url
    url = 'http://10.253.1.50:5002/predict_classifier'
    # 目标识别的url
    # url = 'http://10.253.1.50:5000/predict_detect'
    send_image_to_server(image_path, url)
