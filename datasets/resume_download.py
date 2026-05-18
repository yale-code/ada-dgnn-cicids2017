import requests
import os

url = "https://intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip"
output = "/root/.openclaw/workspace/datasets/dataset.zip"

headers = {}
if os.path.exists(output):
    downloaded = os.path.getsize(output)
    headers['Range'] = f'bytes={downloaded}-'
    print(f"Resuming from {downloaded} bytes...")
else:
    downloaded = 0

with requests.get(url, headers=headers, stream=True, timeout=30) as r:
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0)) + downloaded
    mode = 'ab' if downloaded else 'wb'
    with open(output, mode) as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (1024*1024) < 8192:
                    print(f"Downloaded: {downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB")

print(f"Final size: {os.path.getsize(output)/1024/1024:.1f}MB")
