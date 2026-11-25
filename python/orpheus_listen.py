"""Prepare/serve a local blinded speech panel with write-once responses."""
import argparse
import json
import random
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

RATINGS = {
    "words_in_order": ["correct", "incorrect", "uncertain"],
    "consonants": ["correct", "incorrect", "uncertain"],
    "vowels": ["correct", "incorrect", "uncertain"],
    "tones": ["correct", "incorrect", "uncertain"],
    "phonation": ["natural", "changed", "uncertain"],
    "omissions": ["none", "present", "uncertain"],
    "additions": ["none", "present", "uncertain"],
    "repetitions": ["none", "present", "uncertain"],
    "voice": ["consistent", "changes", "uncertain"],
    "pauses": ["natural", "too short", "too long", "uneven", "uncertain"],
    "naturalness": list("12345"), "continuity": list("12345"),
}


def prepare(folder):
    path = folder / "reveal-key.json"
    if path.exists():
        return
    run = json.loads((folder/"run.json").read_text())
    if len(run["items"]) != len(run["protocol"]["prompts"]):
        raise ValueError("Primary render incomplete")
    key = []
    measurements = {r["id"]: r for r in run["items"]}
    for item in run["protocol"]["prompts"]:
        for arm in ["whole","assembled"]:
            key.append(dict(id=secrets.token_hex(8), arm=arm, text=item["text"], kind=item["kind"],
                            file=f"{item['id']}/{arm}.wav", status=measurements[item['id']][arm+"_status"],
                            chunks=[dict(id=secrets.token_hex(8), file=f"{item['id']}/chunk-{i+1}.wav")
                                    for i in range(len(item["chunks"]))] if arm=="assembled" else []))
    random.SystemRandom().shuffle(key)
    with path.open("x") as stream:
        json.dump(key, stream, indent=2)


def valid_first(data):
    return all(isinstance(data.get(k),str) and data[k].strip() for k in ("listener","dialect","exposure","transcription")) and (
        data.get("confidence") in ["high","medium","low","uncertain"])


def valid_rating(data):
    return all(data.get(k) in choices for k, choices in RATINGS.items())


def connection(folder):
    db = sqlite3.connect(folder/"responses.sqlite3")
    db.execute("CREATE TABLE IF NOT EXISTS responses (listener TEXT, item TEXT, phase TEXT, at TEXT, payload TEXT, PRIMARY KEY(listener,item,phase))")
    return db


def state(db, key, listener):
    records = {(item,phase): json.loads(data) for item,phase,data in db.execute(
        "SELECT item, phase, payload FROM responses WHERE listener=?", (listener,))}
    count = len(key)
    for phase in ("first","rating"):
        for i, item in enumerate(key):
            if (item["id"], phase) not in records:
                data = dict(phase=phase, position=i+1, total=count, id=item["id"])
                if phase == "rating":
                    data.update(text=item["text"], first=records[(item["id"],"first")]["transcription"], ratings=RATINGS)
                return data
    return dict(phase="done", total=count, results=[
        dict(**item, first=records[(item["id"],"first")], rating=records[(item["id"],"rating")])
        for item in key])


def serve(folder, port):
    prepare(folder)
    key = json.loads((folder/"reveal-key.json").read_text())
    assets = {"/audio/"+entry["id"]+".wav": folder/entry["file"] for entry in key}
    for entry in key:
        for chunk in entry["chunks"]:
            assets["/audio/"+chunk["id"]+".wav"] = folder/chunk["file"]
    page = Path(__file__).with_name("orpheus_listen.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, mime, data):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.reply(200,"text/html; charset=utf-8",page)
            elif self.path in assets:
                self.reply(200,"audio/wav",assets[self.path].read_bytes())
            else:
                self.send_error(404)

        def do_POST(self):
            origin = f"http://127.0.0.1:{port}"
            if self.path not in ("/state","/submit") or self.headers.get("Host") != urlparse(origin).netloc or self.headers.get("Origin") != origin:
                return self.send_error(403)
            try:
                size = int(self.headers.get("Content-Length",0))
                if not 0 < size <= 32768:
                    raise ValueError
                req = json.loads(self.rfile.read(size))
                listener = req["session"]
                if not re.fullmatch("[0-9a-f-]{36}",listener):
                    raise ValueError
                with connection(folder) as db:
                    current = state(db,key,listener)
                    if self.path == "/submit":
                        phase = current["phase"]
                        if req["phase"] != phase or req["id"] != current.get("id"):
                            return self.send_error(409)
                        form = req["answers"]
                        if phase == "first" and valid_first(form) or phase == "rating" and valid_rating(form):
                            db.execute("INSERT INTO responses VALUES (?,?,?,?,?)",(listener,req["id"],phase,
                                datetime.now(timezone.utc).isoformat(),json.dumps(form,ensure_ascii=False)))
                        else:
                            raise ValueError
                        current = state(db,key,listener)
                self.reply(200,"application/json",json.dumps(current).encode())
            except sqlite3.IntegrityError:
                self.send_error(409)
            except (KeyError,ValueError,TypeError):
                self.send_error(400)
    ThreadingHTTPServer(("127.0.0.1",port),Handler).serve_forever()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--port",type=int,default=8766)
    args=parser.parse_args()
    serve(args.folder.resolve(),args.port)
