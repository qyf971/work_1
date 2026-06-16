import pandas as pd
import requests
import time
import os

# ---------------------- 德里12个空气质量监测站 ----------------------
sites = [
    ("Anand Vihar", 28.65, 77.31),
    ("Bawana", 28.80, 77.03),
    ("DTU", 28.75, 77.12),
    ("Dwarka-Sector 8", 28.70, 77.13),
    ("IGI Airport (T3)", 28.56, 77.08),
    ("ITO", 28.63, 77.24),
    ("Mundka", 28.68, 77.03),
    ("North Campus-DU", 28.69, 77.21),
    ("Okhla Phase-2", 28.52, 77.27),
    ("Patparganj", 28.64, 77.30),
    ("Pusa", 28.64, 77.16),
    ("Wazirpur", 28.70, 77.16)
]

# ---------------------- OSM 20类POI（对应空气质量研究标准） ----------------------
poi_types = [
    "restaurant", "shop", "hotel", "residential", "office",
    "industrial", "transportation", "car", "education", "hospital",
    "sport", "leisure", "park", "tourism", "public",
    "government", "bank", "service", "warehouse", "road"
]

poi_names = [
    "餐饮", "购物", "住宿", "住宅", "公司",
    "工业", "交通", "汽车", "科教", "医疗",
    "体育", "娱乐", "公园", "景点", "公共设施",
    "政府", "金融", "生活服务", "物流", "道路附属"
]

# ====================== 核心函数：从OSM获取POI数量 ======================
def get_osm_poi(lat, lon, radius=1000):
    """获取半径内的POI数量"""
    overpass_url = "http://overpass-api.de/api/interpreter"

    # 分类查询（OSM标准标签）
    queries = [
        # 1 餐饮
        f'node["amenity"="restaurant"](around:{radius},{lat},{lon});',
        # 2 购物
        f'node["shop"](around:{radius},{lat},{lon});',
        # 3 住宿
        f'node["amenity"="hotel"](around:{radius},{lat},{lon});',
        # 4 住宅
        f'node["building"="residential"](around:{radius},{lat},{lon});',
        # 5 公司
        f'node["building"="office"](around:{radius},{lat},{lon});',
        # 6 工业
        f'node["industrial"](around:{radius},{lat},{lon});',
        # 7 交通
        f'node["highway"="bus_stop"](around:{radius},{lat},{lon});',
        # 8 汽车
        f'node["amenity"="fuel"](around:{radius},{lat},{lon});',
        # 9 科教
        f'node["amenity"="school"](around:{radius},{lat},{lon});',
        # 10 医疗
        f'node["amenity"="hospital"](around:{radius},{lat},{lon});',
        # 11 体育
        f'node["leisure"="sports_centre"](around:{radius},{lat},{lon});',
        # 12 娱乐
        f'node["leisure"](around:{radius},{lat},{lon});',
        # 13 公园
        f'node["leisure"="park"](around:{radius},{lat},{lon});',
        # 14 景点
        f'node["tourism"](around:{radius},{lat},{lon});',
        # 15 公共设施
        f'node["amenity"="public_building"](around:{radius},{lat},{lon});',
        # 16 政府
        f'node["office"="government"](around:{radius},{lat},{lon});',
        # 17 金融
        f'node["amenity"="bank"](around:{radius},{lat},{lon});',
        # 18 生活服务
        f'node["shop"](around:{radius},{lat},{lon});',
        # 19 物流
        f'node["building"="warehouse"](around:{radius},{lat},{lon});',
        # 20 道路附属
        f'node["highway"](around:{radius},{lat},{lon});'
    ]

    counts = []
    for q in queries:
        try:
            query = f"""[out:json][timeout:25];({q});out count;"""
            response = requests.get(overpass_url, params={"data": query})
            data = response.json()
            cnt = data.get("elements", [{}])[0].get("tags", {}).get("nodes", "0")
            counts.append(int(cnt))
        except:
            counts.append(0)
        time.sleep(0.3)
    return counts

# ====================== 批量抓取 ======================
results = []
for name, lat, lon in sites:
    print(f"正在抓取 → {name}")
    vec = get_osm_poi(lat, lon, radius=1000)
    results.append([name, lat, lon] + vec)

# ====================== 保存CSV ======================
df = pd.DataFrame(
    results,
    columns=["site_name", "Latitude", "Longitude"] + poi_names
)

df.to_csv("delhi_12site_osm_poi.csv", index=False, encoding="utf-8-sig")

print("\n✅ OSM POI 抓取完成！")
print("文件已保存到当前目录：delhi_12site_osm_poi.csv")
print(df)