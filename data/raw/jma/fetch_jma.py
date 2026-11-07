"""Download hourly JMA observations used by DC-DECOM v2 (run once; raw files are kept unchanged).

Service: 気象庁「過去の気象データ・ダウンロード」 https://www.data.jma.go.jp/risk/obsdl/
Endpoint: POST https://www.data.jma.go.jp/risk/obsdl/show/table (form fields as sent by the
web page, see REQUEST below). One request per station: hourly values (aggrgPeriod 9) of
temperature (element 201) and global solar radiation (element 610, MJ/m2 over the previous
hour), 2020-01-01 to 2023-12-31, continuous period. Files are saved byte for byte (Shift_JIS /
cp932, CRLF) as returned. A log with the request, HTTP status, size and sha256 is written to
fetch_log.json. Existing files are not overwritten unless --force is given.

Usage: .venv/bin/python data/raw/jma/fetch_jma.py [--force]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://www.data.jma.go.jp/risk/obsdl/show/table"
STATIONS = (  # (region, city, WMO station number used by obsdl as 's' + number)
    ("Hokkaido", "Sapporo", "47412"),
    ("Tohoku", "Sendai", "47590"),
    ("Kanto", "Tokyo", "47662"),
    ("Kansai", "Osaka", "47772"),
    ("Kyushu", "Fukuoka", "47807"),
)
ELEMENTS = [["201", ""], ["610", ""]]  # 気温, 全天日射量 (前1時間)
YMD = ["2020", "2023", "1", "12", "1", "31"]  # start year, end year, start/end month, start/end day
PAUSE_S = 5.0


def request_fields(station: str) -> dict:
    return {
        "stationNumList": json.dumps([f"s{station}"]),
        "aggrgPeriod": "9",
        "elementNumList": json.dumps(ELEMENTS),
        "interAnnualType": "1",
        "ymdList": json.dumps(YMD),
        "option": json.dumps({"list": ["obs"], "nyear": 0}),
        "downloadFlag": "true",
        "rmkFlag": "1",
        "disconnectFlag": "1",
        "youbiFlag": "0",
        "fukenFlag": "0",
        "kijiFlag": "0",
        "csvFlag": "1",
        "jikantaiFlag": "0",
        "jikantaiList": "[]",
        "ymdLiteral": "1",
    }


def raw_name(region: str, city: str, station: str) -> str:
    return f"jma_hourly_{station}_{city}_2020-2023.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    log_path = HERE / "fetch_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"requests": []}
    for i, (region, city, station) in enumerate(STATIONS):
        out = HERE / raw_name(region, city, station)
        if out.exists() and not args.force:
            print(f"keep {out.name}")
            continue
        if i:
            time.sleep(PAUSE_S)
        fields = request_fields(station)
        body = urllib.parse.urlencode(fields).encode()
        req = urllib.request.Request(URL, data=body, method="POST", headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "dc-decom-v2 research (one request per station)",
        })
        when = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
            status = resp.status
            disp = resp.headers.get("Content-Disposition", "")
        if "attachment" not in disp:
            raise RuntimeError(f"{city}: no CSV attachment (status {status}): {data[:300]!r}")
        out.write_bytes(data)
        entry = {"region": region, "city": city, "station": station, "file": out.name,
                 "url": URL, "method": "POST", "fields": fields, "retrieved_at": when,
                 "http_status": status, "bytes": len(data),
                 "sha256": hashlib.sha256(data).hexdigest()}
        log["requests"] = [e for e in log["requests"] if e.get("file") != out.name] + [entry]
        print(f"saved {out.name} ({len(data)} bytes)")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
