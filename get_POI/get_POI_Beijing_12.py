import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import time
import os

# ---------------------- 12个北京空气质量站点 ----------------------
sites = [
    ("Aotizhongxin", 39.982, 116.397),
    ("Changping", 40.217, 116.23),
    ("Dingling", 40.292, 116.22),
    ("Dongsi", 39.929, 116.417),
    ("Guanyuan", 39.929, 116.339),
    ("Gucheng", 39.914, 116.184),
    ("Huairou", 40.328, 116.628),
    ("Nongzhanguan", 39.937, 116.461),
    ("Shunyi", 40.127, 116.655),
    ("Tiantan", 39.886, 116.407),
    ("Wanliu", 39.987, 116.287),
    ("Wanshouxigong", 39.878, 116.352)
]

# ---------------------- 20类POI ----------------------
POI_CLASSES = [
    "010000", "030000", "020000", "120000", "090000",
    "080000", "050000", "150000", "070000", "060000",
    "100000", "040000", "220100", "110000", "130000",
    "140000", "160000", "040000", "170000", "190000"
]

POI_NAMES = [
    "餐饮","购物","住宿","住宅","公司","工业","交通","汽车","科教","医疗",
    "体育","娱乐","公园","景点","公共设施","政府","金融","生活服务","物流","道路附属"
]

# ====================== 配置 ======================
KEY = "94da6fd33456df3e03c81d08bd3a5a02"
RADIUS = 1000

# ====================== 获取POI ======================
def fetch_one(args):
    lng, lat, code = args
    url = "https://restapi.amap.com/v3/place/around"
    params = {"key": KEY, "location": f"{lng},{lat}", "radius": RADIUS, "types": code, "offset": 1}
    try:
        return int(requests.get(url, params=params, timeout=5).json().get("count", 0))
    except:
        return 0

def get_site_vector(name, lat, lng):
    print(f"正在抓取 → {name}")
    with ThreadPoolExecutor(max_workers=10) as executor:
        vec = list(executor.map(fetch_one, [(lng, lat, c) for c in POI_CLASSES]))
    return [name, lat, lng] + vec

# ====================== 执行 ======================
results = [get_site_vector(n, la, lo) for n, la, lo in sites]

# ====================== 保存到【当前文件夹】 ======================
df = pd.DataFrame(results, columns=["site_name", "Latitude", "Longitude"] + POI_NAMES)

# 保存路径：当前文件夹 / beijing_12site_20poi.csv
save_path = "beijing_12site_20poi.csv"
df.to_csv(save_path, index=False, encoding="utf-8-sig")

print("\n✅ 全部完成！")
print(f"文件已保存到：{save_path}")
print(df)