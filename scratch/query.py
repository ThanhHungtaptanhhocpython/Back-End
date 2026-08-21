import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

url = 'http://127.0.0.1:3000/temporalsearch'
data = {
    'query': [
        {'query': 'Một người đứng dưới nước và rọi đèn.'},
        {'query': 'Tiếp theo là cảnh người này kéo lưới cá lúc bình minh'},
        {'query': 'sau đó được một nhóm người khác tiến đến dùng máy quay ghi hình.'}
    ],
    'topk': 5
}
data_bytes = json.dumps(data).encode('utf-8')
req = urllib.request.Request(url, data=data_bytes, headers={'Content-Type': 'application/json'}, method='POST')

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        print(f"Success: {result.get('success')}, Total Items: {result.get('data', {}).get('total_items')}")
        if result['success']:
            for i, item in enumerate(result['data']['items']):
                vid = item.get('video_id', '')
                fid = item.get('frame_id', '')
                print(str(i+1) + '. Video: ' + str(vid) + ', Frame: ' + str(fid))
        else:
            print('Search failed:', result.get('message'))
except Exception as e:
    print('Error:', e)
