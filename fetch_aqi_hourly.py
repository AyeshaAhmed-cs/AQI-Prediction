import requests

API_KEY = "59741dd6dd39e39a9380da6133bc2f0fe1656336"
CITY = "Karachi"
URL = f"https://api.waqi.info/feed/Karachi/?token=59741dd6dd39e39a9380da6133bc2f0fe1656336"

def fetch_aqi():
    response = requests.get(URL)
    data = response.json()

    if data["status"] != "ok":
        return None

    iaqi = data["data"]["iaqi"]
    time_info = data["data"]["time"]

    row = {
        "city": CITY,
        "timestamp": time_info["iso"],
        "aqi": data["data"]["aqi"],
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "temperature": iaqi.get("t", {}).get("v"),
        "humidity": iaqi.get("h", {}).get("v"),
        "pressure": iaqi.get("p", {}).get("v"),
        "wind_speed": iaqi.get("w", {}).get("v"),
    }

    return row

if __name__ == "__main__":
    print(fetch_aqi())
