import requests
import os

def send_video_to_server(video_path, url, resource_id="12341234", media_type="video"):
    """
    将本地视频文件以二进制数据的形式传输到Python服务器。
    """
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
    
    headers = {
        'Content-Type': 'application/octet-stream',
        'resourceId': resource_id,
        'mediaType': media_type
    }
    response = requests.post(url, data=video_bytes, headers=headers)

    if response.status_code == 200:
        try:
            response_data = response.json()
            print(video_path, response_data)
        except ValueError:
            print(video_path, "Response is not JSON.")
    else:
        print(video_path, f"Request failed with status code {response.status_code}: {response.reason}")

if __name__ == '__main__':
    url = 'http://localhost:5002/predict_video'
    video_path = '/mnt/data1/animal_sample250.mp4'
    send_video_to_server(video_path, url)