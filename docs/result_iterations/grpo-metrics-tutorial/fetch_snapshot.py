"""Read public SwanLab metrics. No login, credentials, or remote mutations."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import urllib.request

BASE = "https://swanlab.cn/api"
PROJECT = "yyq90/GRPO-Qwen3.8-Smoke"
RUN = "psmh3kg8"
OUT = Path(__file__).parent


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def main():
    project = request("/project/" + PROJECT)
    run = request("/project/" + PROJECT + "/runs/" + RUN)
    inventory = []
    for kind in ["FLOAT", "SYSTEM"]:
        rows = request("/projects/" + PROJECT + "/series?type=" + kind + "&size=500")
        assert isinstance(rows, list) and len(rows) < 500, "Check pagination"
        inventory.extend(rows)

    def fetch(rows):
        return request("/house/metrics/scalar", {
            "projectId": project["cuid"],
            "columns": [{"experimentId": run["cuid"], "key": row["key"]} for row in rows],
            "xType": "step",
        })

    series = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch, [inventory[i:i+8] for i in range(0, len(inventory), 8)]):
            assert isinstance(result, list), result
            series.extend(result)
    config = {k: v["value"] for k, v in run["profile"]["config"].items()}
    snapshot = {
        "fetchedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "url": "https://swanlab.cn/@" + PROJECT + "/v1/lo7lkd/runs/" + RUN + "/chart",
        "name": run["name"], "state": run["state"], "run": RUN,
        "visibility": project["visibility"], "inventory": inventory,
        "config": config, "series": series,
    }
    (OUT / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    populated = [s for s in series if s.get("metrics")]
    print(json.dumps({"inventory": len(inventory), "series": len(series), "populated": len(populated),
                      "empty": [s["key"] for s in series if not s.get("metrics")],
                      "system": [{"key": s["key"], "first": s["metrics"][0], "last": s["metrics"][-1]} for s in populated if s["key"].startswith("__")],
                      "maxTrainingStep": max(p["index"] for s in populated if not s["key"].startswith("__") for p in s["metrics"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
