"""Check a segmentation run and report blinded listening progress without listener identifiers."""
import argparse
import hashlib
import json
import sqlite3
import wave
from pathlib import Path


def content(path):
    return json.loads(path.read_text())


def pcm(path):
    with wave.open(str(path)) as reader:
        assert (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) == (1,2,24000)
        return reader.readframes(reader.getnframes())


def report(folder):
    run = content(folder/"run.json")
    protocol = run["protocol"]
    items_by_id = {i["id"]:i for i in run["items"]}
    if len(items_by_id) != len(protocol["prompts"]):
        raise ValueError("Run is incomplete")
    data = []
    for prompt in protocol["prompts"]:
        key = prompt["id"]
        item = items_by_id[key]
        base = folder/key
        assert " ".join(prompt["chunks"]).split() == prompt["text"].split()
        audio = {}
        emissions = []
        for name in ["whole", *(f"chunk-{i+1}" for i in range(len(prompt["chunks"])))]:
            rec = content(base/(name+".json"))
            path = base/(name+".wav")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == rec["sha256"]
            audio[name] = pcm(path)
            assert len(audio[name]) == rec["samples"]*2 == rec["frames"]*4096
            assert rec["output_ids"][-1] == 128258 and rec["stop"]=="speech_eos"
            assert len(rec["output_ids"]) == rec["generated_tokens"] < rec["output_cap"]
            emissions.append({k:v for k,v in rec.items() if k != "output_ids"})
        # Verify all original PCM at every join, including both tone-bearing edges.
        expected = audio["chunk-1"]
        for i,join in enumerate(item["joins"],2):
            expected += bytes(2*join["added_samples"]) + audio[f"chunk-{i}"]
        assert expected == pcm(base/"assembled.wav")
        assert hashlib.sha256((base/"assembled.wav").read_bytes()).hexdigest() == item["assembled_sha256"]
        data.append({**item,"emissions":emissions})
    listening = dict(first_responses=0, ratings=0, completed_panels=0, assessments=[])
    db = folder/"responses.sqlite3"
    if db.exists():
        with sqlite3.connect(db.resolve().as_uri()+"?mode=ro",uri=True) as con:
            counts=dict(con.execute("SELECT phase,count(*) FROM responses GROUP BY phase"))
            listening.update(first_responses=counts.get("first",0),ratings=counts.get("rating",0))
            keymap = {x["id"]:x for x in content(folder/"reveal-key.json")}
            for listener,count in con.execute("SELECT listener,count(*) FROM responses WHERE phase='rating' GROUP BY listener"):
                if count != len(keymap):
                    continue
                listening["completed_panels"] += 1
                rows = []
                for identifier, payload in con.execute("SELECT item,payload FROM responses WHERE listener=? AND phase='rating'",(listener,)):
                    entry = keymap[identifier]
                    responses = json.loads(payload)
                    rows.append(dict(text=entry["text"],arm=entry["arm"],technical_status=entry["status"],ratings=responses))
                listening["assessments"].append(rows)
    return {**{k:v for k,v in run.items() if k not in ("items","protocol")}, "items":data, "listening":listening}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    result = report(args.folder)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(f"Verified {len(result['items'])} whole/chunk pairs.")
    print(json.dumps(result["listening"],ensure_ascii=False))
