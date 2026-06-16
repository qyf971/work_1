import requests
import csv
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------------- 配置 --------------------------
AMAP_KEY = "9b813be6723c89c1249d233511c3a2c7"
# 海外搜索接口（支持全球）
AMAP_URL = "https://restapi.amap.com/v3/place/around"
RADIUS = 1500  # 德里站点建议1.5km半径

# 德里12个监测站（WGS84坐标）
DELHI_SITES = [
    {"name": "Anand Vihar",      "lng": 77.3153, "lat": 28.6507},
    {"name": "Bawana",           "lng": 77.0333, "lat": 28.8000},
    {"name": "DTU",              "lng": 77.1245, "lat": 28.7500},
    {"name": "Dwarka Sector 8",  "lng": 77.1300, "lat": 28.7000},
    {"name": "IGI Airport",      "lng": 77.0800, "lat": 28.5600},
    {"name": "ITO",              "lng": 77.2400, "lat": 28.6300},
    {"name": "Mundka",           "lng": 77.0300, "lat": 28.6800},
    {"name": "DU North Campus",  "lng": 77.2100, "lat": 28.6900},
    {"name": "Okhla Phase 2",    "lng": 77.2700, "lat": 28.5200},
    {"name": "Patparganj",       "lng": 77.3000, "lat": 28.6400},
    {"name": "Pusa",             "lng": 77.1600, "lat": 28.6400},
    {"name": "Wazirpur",         "lng": 77.1600, "lat": 28.7000},
]

# POI分类（适配印度场景）
POI_TYPES = [
    "restaurant","cafe","shop","transport","education",
    "medical","residence","service","entertainment","office","other"
]

# -------------------------- 核心：获取POI --------------------------
def get_poi(lng, lat):
    pois = []
    page = 1
    while True:
        params = {
            "key": AMAP_KEY,
            "location": f"{lng},{lat}",
            "radius": RADIUS,
            "offset": 25,
            "page": page,
            "extensions": "all",
            "output": "json",
            "country": "IND",  # 关键：指定印度
            "language": "en"   # 英文返回
        }
        try:
            # 关闭SSL验证（境外必备）
            resp = requests.get(AMAP_URL, params=params, verify=False, timeout=15)
            data = resp.json()
            if data.get("status") != "1":
                print(f"错误：{data.get('info')}")
                break
            page_pois = data.get("pois", [])
            if not page_pois:
                break
            pois.extend(page_pois)
            if len(page_pois) < 25:
                break
            page += 1
        except Exception as e:
            print(f"请求异常：{e}")
            break
    return pois

# -------------------------- POI → 向量 --------------------------
def poi_to_vec(pois):
    vec = {t:0 for t in POI_TYPES}
    for p in pois:
        t = p.get("type","").lower()
        if any(k in t for k in ["restaurant","cafe","food"]):
            vec["restaurant"] +=1
        elif any(k in t for k in ["shop","mall","store"]):
            vec["shop"] +=1
        elif any(k in t for k in ["station","bus","metro","airport"]):
            vec["transport"] +=1
        elif any(k in t for k in ["school","college","edu"]):
            vec["education"] +=1
        elif any(k in t for k in ["hospital","clinic","medical"]):
            vec["medical"] +=1
        elif any(k in t for k in ["residence","apartment","flat"]):
            vec["residence"] +=1
        else:
            vec["other"] +=1
    return [vec[t] for t in POI_TYPES]

# -------------------------- 批量执行 --------------------------
if __name__ == "__main__":
    headers = ["site","lng","lat","total"] + POI_TYPES
    rows = []
    for site in DELHI_SITES:
        name = site["name"]
        lng, lat = site["lng"], site["lat"]
        print(f"\n处理：{name}")
        pois = get_poi(lng, lat)
        vec = poi_to_vec(pois)
        row = [name, round(lng,5), round(lat,5), len(pois)] + vec
        rows.append(row)
        print(f"POI总数：{len(pois)} | 向量：{vec}")

    # 保存CSV
    with open("delhi_poi_final.csv","w",encoding="utf-8-sig",newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print("\n✅ 完成！文件：delhi_poi_final.csv")