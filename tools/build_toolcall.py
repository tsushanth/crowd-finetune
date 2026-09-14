import argparse
import json
import re
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import toolcall

CORPUS_PATH = ROOT / "data" / "toolcall_corpus.jsonl"
ID_RE = re.compile(r"^TC-[0-9]{3}$")


def _validate_schema(tool_name: str, parameters: dict, errors: list[str]) -> None:
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        errors.append(f"{tool_name}: parameters must be a JSON-schema object with type=object")
        return
    props = parameters.get("properties") or {}
    if not isinstance(props, dict) or not props:
        errors.append(f"{tool_name}: parameters.properties must be a non-empty object")
        return
    for key, spec in props.items():
        if not isinstance(spec, dict) or "type" not in spec:
            errors.append(f"{tool_name}.{key}: property spec needs a type")
        if "enum" in spec and not isinstance(spec["enum"], list):
            errors.append(f"{tool_name}.{key}: enum must be a list")
    required = parameters.get("required") or []
    for key in required:
        if key not in props:
            errors.append(f"{tool_name}: required {key!r} is not a declared property")


def _matches_expected(payload: dict) -> list[str]:
    errors = []
    exp = payload.get("expected")
    if exp is None or not exp.get("tool"):
        return errors
    tools = {t["name"]: t for t in payload["tools"]}
    spec = tools.get(exp["tool"])
    if spec is None:
        errors.append(f"expected tool {exp['tool']!r} is not in the toolset")
        return errors
    props = spec["parameters"].get("properties") or {}
    required = spec["parameters"].get("required") or []
    args = exp.get("arguments") or {}
    for key in required:
        if key not in args:
            errors.append(f"expected call missing required argument {key!r}")
    for key in args:
        if key not in props:
            errors.append(f"expected call uses undeclared argument {key!r}")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true",
                    help="rewrite question/golden_answer from each row payload")
    args = ap.parse_args()

    if not CORPUS_PATH.exists():
        raise SystemExit(f"corpus not found: {CORPUS_PATH}")
    rows = [json.loads(l) for l in CORPUS_PATH.read_text().splitlines() if l.strip()]
    seen: set[str] = set()
    seen_q: set[str] = set()
    errors: list[str] = []
    for row in rows:
        cid = row.get("corpus_id")
        if not ID_RE.match(cid or ""):
            errors.append(f"{cid}: corpus_id must match TC-XXX")
        if cid in seen:
            errors.append(f"duplicate corpus_id {cid}")
        seen.add(cid)
        question = row.get("question")
        if question:
            if question in seen_q:
                errors.append(f"{cid}: duplicate question (same as an earlier row)")
            seen_q.add(question)
        payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
        for t in payload.get("tools") or []:
            _validate_schema(t["name"], t.get("parameters") or {}, errors)
        errors.extend(_matches_expected(payload))
        golden = toolcall.golden_text(payload)
        question = toolcall.render_question(payload)
        if golden != row.get("golden_answer"):
            errors.append(f"{cid}: golden_answer != golden_text(payload) (hint: run --regenerate)")
        if question != row.get("question"):
            errors.append(f"{cid}: question != render_question(payload) (hint: run --regenerate)")
        if not toolcall.is_correct(golden, row)["correct"]:
            errors.append(f"{cid}: the golden answer does not validate as correct")

    if args.regenerate:
        with CORPUS_PATH.open("w") as f:
            for row in rows:
                payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
                row["question"] = toolcall.render_question(payload)
                row["golden_answer"] = toolcall.golden_text(payload)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"regenerated {len(rows)} rows in {CORPUS_PATH.name}")
        return

    if errors:
        print(f"{len(errors)} validation errors:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    gold = sum(1 for r in rows if r.get("is_gold"))
    declines = sum(1 for r in rows if toolcall.expects_decline(r))
    print(f"ok: {len(rows)} items ({gold} gold, {len(rows) - gold} candidates, "
          f"{declines} expected-decline, {len(rows) - declines} expected-call)")
    round_trip = toolcall.render_question(json.loads(rows[0]["payload"])) == rows[0]["question"]
    print(f"question/golden round-trip: clean" if not errors and round_trip else "question/golden round-trip: dirty")


if __name__ == "__main__":
    main()