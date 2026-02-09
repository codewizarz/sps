import os
import time
import requests
from datetime import datetime, timedelta
from tqdm import tqdm

SAVE_DIR = "nse_fo_bhavcopies"
os.makedirs(SAVE_DIR, exist_ok=True)

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2026, 2, 8)

BASE_URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)

    # Hit homepage first to generate cookies
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except requests.exceptions.RequestException:
        pass  # Continue even if homepage fails, might work directly

    return session


def download_file(session, date_obj):
    date_str = date_obj.strftime("%Y%m%d")
    url = BASE_URL.format(date=date_str)

    file_path = os.path.join(SAVE_DIR, f"{date_str}.zip")

    if os.path.exists(file_path):
        return "exists"

    try:
        response = session.get(url, timeout=20)

        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            return "downloaded"

        elif response.status_code == 404:
            return "missing"

        else:
            return f"error {response.status_code}"

    except requests.exceptions.RequestException:
        return "retry"


def run_downloader():
    session = create_session()

    current = START_DATE
    failures = []

    days = (END_DATE - START_DATE).days

    for _ in tqdm(range(days + 1)):  # Include end date
        if current > END_DATE:
            break

        result = download_file(session, current)

        if result == "retry":
            time.sleep(5)
            session = create_session()
            result = download_file(session, current)

        if result not in ["downloaded", "exists", "missing"]:
            failures.append(current.strftime("%Y%m%d"))

        # DO NOT hammer NSE
        time.sleep(0.4)

        current += timedelta(days=1)

    print("\nDone.")
    print("Failures:", failures)


if __name__ == "__main__":
    run_downloader()
