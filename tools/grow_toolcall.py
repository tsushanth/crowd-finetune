import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import toolcall
from tools import build_toolcall

CORPUS_PATH = ROOT / "data" / "toolcall_corpus.jsonl"
BENCH_PATH = ROOT / "data" / "bench.jsonl"
SFT_PATH = ROOT / "data" / "toolcall_sft.jsonl"
ID_RE = re.compile(r"^TC-[0-9]{3}$")

SUITE_TEMPLATES = {
    "suites:home": {
        "set_thermostat": {
            "combos": [
                (18, "eco"), (19, "eco"), (20, "eco"), (21, "eco"), (22, "comfort"),
                (23, "comfort"), (24, "boost"), (22, "boost"), (20, "comfort"), (19, "boost"),
            ],
            "rooms": ["living room", "bedroom", "office", "kitchen"],
            "templates": [
                "Set the {room} thermostat to {tc} degrees in {mode} mode.",
                "Could you set the {room} thermostat to {tc} and switch it to {mode}?",
                "Adjust the {room} heating to {tc} degrees, {mode} mode.",
            ],
        },
        "turn_off_device": {
            "devices": ["oven", "kettle", "lights", "hvac", "tv"],
            "templates": [
                "Turn the {device} off.",
                "Please switch the {device} off.",
                "Could you power down the {device}?",
            ],
        },
        "schedule_scene": {
            "scenes": ["movie_night", "good_night", "morning", "away"],
            "times": ["20:00", "21:30", "07:00", "22:15", "06:30"],
            "templates": [
                "Schedule the {scene} scene at {time}.",
                "Set up {scene} automation to run at {time}.",
                "Automate {scene} every day starting at {time}.",
            ],
        },
        "query_energy": {
            "metrics": ["daily_kwh", "monthly_cost", "peak_power", "current_w"],
            "templates": [
                "What is my {natural} right now?",
                "Show me the household {natural}.",
                "Can you tell me our {natural}?",
            ],
        },
        "declines": [
            "Turn the living room TV on.",
            "Dim the kitchen lights down to 30 percent.",
            "Sign me out of the energy dashboard.",
            "What is the weather forecast for tomorrow?",
            "Play some music in the kitchen.",
            "Set a reminder to water the plants at noon.",
            "Text me when the oven is done preheating.",
            "Turn up the volume of the TV.",
            "Lock the front door from here.",
            "What movies are playing tonight?",
            "Add eggs and milk to the shopping list.",
            "Open the garage door halfway.",
            "Take a photo with the hallway camera.",
        ],
    },
    "suites:crm": {
        "create_ticket": {
            "subjects": [
                "Login page times out", "Invoice shows wrong total",
                "Cannot reset password", "Billing charge duplicated",
                "Dashboard export broken", "Emails not arriving",
            ],
            "emails": ["a.patel@mail.com", "k.ross@mail.com", "m.lee@mail.com", "j.walsh@mail.com"],
            "priorities": ["low", "medium", "high", "urgent"],
            "templates": [
                "Open a {priority} ticket about {subject}. Customer: {email}.",
                "File a ticket for {subject} at {priority} priority. Contact is {email}.",
                "Create a {priority} support ticket: {subject}. Reach them at {email}.",
            ],
        },
        "resolve_ticket": {
            "tickets": [1142, 2247, 3301, 4099],
            "notes": [
                "issue confirmed fixed", "resolved after password reset",
                "user confirmed working", "root cause patched",
            ],
            "templates": [
                "Ticket {tid} is fixed, close it. Note: {note}.",
                "Resolve ticket {tid}. {note}.",
            ],
        },
        "search_kb": {
            "queries": [
                "password reset", "invoice download", "cancel subscription",
                "export data", "API rate limits",
            ],
            "limits": [3, 5, 10],
            "templates": [
                "Show the top {lim} articles about \"{q}\".",
                "Find the best {lim} help articles for \"{q}\".",
            ],
        },
        "update_plan": {
            "emails": ["a.patel@mail.com", "k.ross@mail.com", "m.lee@mail.com"],
            "plans": ["starter", "pro", "enterprise"],
            "templates": [
                "Upgrade account {email} to the {plan} plan.",
                "Change the plan for {email} to {plan}.",
                "Move {email} onto {plan}.",
            ],
        },
        "declines": [
            "Delete the enterprise plan from the catalog.",
            "Send an email to the customer directly.",
            "Issue a refund for the customer's last payment.",
            "Call the customer to follow up.",
            "Export the ticket list to CSV.",
            "Show me the customer's purchase history.",
            "Merge tickets 1142 and 2247.",
            "Assign this task to Sarah on the team.",
            "Generate a monthly sales report.",
            "Send a broadcast message to all customers.",
            "Check whether ticket 1142 is within SLA.",
            "Change the customer's email on file to the new one.",
        ],
    },
    "suites:pay": {
        "send_money": {
            "recipients": ["GreenGrid", "Vera's Cafe", "Sam Carter", "Luna Books"],
            "amounts_usd": [15, 20, 50, 75, 120, 300],
            "amounts_eur": [10, 25, 60, 90, 150, 400],
            "templates": [
                "Pay {recipient} {amount} dollars.",
                "Send {amount} dollars to {recipient}.",
                "Send {recipient} {amount} in euros.",
                "Transfer {amount} euros to {recipient}.",
            ],
        },
        "get_balance": {
            "accounts": ["checking", "savings"],
            "templates": [
                "What is my {account} balance?",
                "Show the balance of the {account} account.",
                "What's the current {account} balance?",
            ],
        },
        "categorize_transaction": {
            "tids": ["txn-001", "txn-002", "txn-003", "txn-004"],
            "categories": ["food", "transport", "utilities", "entertainment", "other"],
            "templates": [
                "Categorize {tid} as {category}.",
                "Mark transaction {tid} under {category}.",
                "Set the category of {tid} to {category}.",
            ],
        },
        "declines": [
            "Refund my recent purchase of headphones.",
            "Reverse the last payment I made.",
            "Apply for a personal loan of 300 dollars.",
            "What is the balance of account A-45?",
            "Shift 500 from savings into checking.",
            "Pay my credit card bill.",
            "Set up automatic monthly savings.",
            "Show my bank statements for March.",
            "Cancel my most recent transaction.",
            "How much did I spend on subscriptions last month?",
            "Split the dinner bill between me and three friends.",
            "Exchange 100 dollars into yen.",
        ],
    },
}

EXTRA_METRIC_NATURAL = {
    "daily_kwh": "energy use today in kWh",
    "monthly_cost": "electricity cost this month",
    "peak_power": "peak power draw",
    "current_w": "current power use in watts",
}


def _arg(key, value):
    return {key: value}


def _expand_suite(name: str, rows_out: list, start_id: int) -> int:
    spec = SUITE_TEMPLATES[name]
    cid = start_id
    for tool, tspec in spec.items():
        if tool == "declines":
            for req in tspec:
                payload = {"request": req, "tools": [], "expected": None}
                rows_out.append((cid, name, payload, "decline"))
                cid += 1
            continue
        if tool == "set_thermostat":
            for tc, mode in tspec["combos"]:
                for room in tspec["rooms"][:3]:
                    for tmpl in tspec["templates"]:
                        text = tmpl.format(room=room, tc=tc, mode=mode).capitalize()
                        payload = {
                            "request": text,
                            "tools": [],
                            "expected": {"tool": tool, "arguments": {"target_c": tc, "mode": mode}},
                        }
                        rows_out.append((cid, name, payload, "exact"))
                        cid += 1
        elif tool == "turn_off_device":
            for dev in tspec["devices"]:
                for tmpl in tspec["templates"]:
                    text = tmpl.format(device=dev).capitalize()
                    payload = {
                        "request": text,
                        "tools": [],
                        "expected": {"tool": tool, "arguments": {"device": dev}},
                    }
                    rows_out.append((cid, name, payload, "exact"))
                    cid += 1
        elif tool == "schedule_scene":
            for scene in tspec["scenes"]:
                for t in tspec["times"][:4]:
                    for tmpl in tspec["templates"]:
                        text = tmpl.format(scene=scene, time=t).capitalize()
                        payload = {
                            "request": text,
                            "tools": [],
                            "expected": {"tool": tool, "arguments": {"scene": scene, "time": t}},
                        }
                        rows_out.append((cid, name, payload, "exact"))
                        cid += 1
        elif tool == "query_energy":
            for metric in tspec["metrics"]:
                for tmpl in tspec["templates"]:
                    text = tmpl.format(natural=EXTRA_METRIC_NATURAL[metric]).capitalize()
                    payload = {
                        "request": text,
                        "tools": [],
                        "expected": {"tool": tool, "arguments": {"metric": metric}},
                    }
                    rows_out.append((cid, name, payload, "exact"))
                    cid += 1
        elif tool == "create_ticket":
            for subject in tspec["subjects"]:
                for email in tspec["emails"][:2]:
                    for prio in tspec["priorities"]:
                        for tmpl in tspec["templates"]:
                            text = tmpl.format(priority=prio, subject=subject, email=email).capitalize()
                            payload = {
                                "request": text,
                                "tools": [],
                                "expected": {
                                    "tool": tool,
                                    "arguments": {"subject": subject, "priority": prio, "customer_email": email},
                                },
                            }
                            rows_out.append((cid, name, payload, "exact"))
                            cid += 1
        elif tool == "resolve_ticket":
            for tid, note in zip(tspec["tickets"], tspec["notes"]):
                for tmpl in tspec["templates"]:
                    text = tmpl.format(tid=tid, note=note).capitalize()
                    payload = {
                        "request": text,
                        "tools": [],
                        "expected": {"tool": tool, "arguments": {"ticket_id": tid, "note": note}},
                    }
                    rows_out.append((cid, name, payload, "exact"))
                    cid += 1
        elif tool == "search_kb":
            for q in tspec["queries"]:
                for lim in tspec["limits"]:
                    for tmpl in tspec["templates"]:
                        text = tmpl.format(q=q, lim=lim).capitalize()
                        payload = {
                            "request": text,
                            "tools": [],
                            "expected": {"tool": tool, "arguments": {"query": q, "limit": lim}},
                        }
                        rows_out.append((cid, name, payload, "exact"))
                        cid += 1
        elif tool == "update_plan":
            for email in tspec["emails"]:
                for plan in tspec["plans"]:
                    for tmpl in tspec["templates"]:
                        text = tmpl.format(email=email, plan=plan).capitalize()
                        payload = {
                            "request": text,
                            "tools": [],
                            "expected": {"tool": tool, "arguments": {"email": email, "plan": plan}},
                        }
                        rows_out.append((cid, name, payload, "exact"))
                        cid += 1
        elif tool == "send_money":
            for recipient in tspec["recipients"]:
                for amount in tspec["amounts_usd"]:
                    text = tspec["templates"][0].format(recipient=recipient, amount=amount).capitalize()
                    payload = {
                        "request": text,
                        "tools": [],
                        "expected": {
                            "tool": tool,
                            "arguments": {"recipient": recipient, "amount_cents": amount * 100, "currency": "USD"},
                        },
                    }
                    rows_out.append((cid, name, payload, "exact"))
                    cid += 1
                for amount in tspec["amounts_eur"]:
                    text = tspec["templates"][2].format(recipient=recipient, amount=amount).capitalize()
                    payload = {
                        "request": text,
                        "tools": [],
                        "expected": {
                            "tool": tool,
                            "arguments": {"recipient": recipient, "amount_cents": amount * 100, "currency": "EUR"},
                        },
                    }
                    rows_out.append((cid, name, payload, "exact"))
                    cid += 1
        elif tool == "get_balance":
            for acc in tspec["accounts"]:
                for tmpl in tspec["templates"]:
                    text = tmpl.format(account=acc).capitalize()
                    payload = {
                        "request": text,
                        "tools": [],
                        "expected": {"tool": tool, "arguments": {"account": acc}},
                    }
                    rows_out.append((cid, name, payload, "exact"))
                    cid += 1
        elif tool == "categorize_transaction":
            for tid in tspec["tids"]:
                for cat in tspec["categories"]:
                    for tmpl in tspec["templates"]:
                        text = tmpl.format(tid=tid, category=cat).capitalize()
                        payload = {
                            "request": text,
                            "tools": [],
                            "expected": {"tool": tool, "arguments": {"transaction_id": tid, "category": cat}},
                        }
                        rows_out.append((cid, name, payload, "exact"))
                        cid += 1
    return cid


def _write(new_rows: list, suite_maps: dict, start_id: int) -> None:
    out = []
    for j, (_, suite, payload, variant) in enumerate(new_rows, start=start_id):
        tools = suite_maps[suite]
        payload["tools"] = tools
        row = {
            "corpus_id": f"TC-{j:03d}",
            "niche": "tool-call",
            "source": f"grow | {suite} | {variant}",
            "is_gold": 0,
            "payload": json.dumps(payload, ensure_ascii=False),
        }
        row["question"] = toolcall.render_question(payload)
        row["golden_answer"] = toolcall.golden_text(payload)
        out.append(row)
    corpus = [json.loads(line) for line in CORPUS_PATH.read_text().splitlines() if line.strip()]
    corpus.extend(out)
    with CORPUS_PATH.open("w") as f:
        for row in corpus:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    first, last = out[0]["corpus_id"], out[-1]["corpus_id"]
    print(f"appended {len(out)} rows ({first}..{last}) -> corpus now {len(corpus)}")


def make_sft() -> None:
    corpus = [json.loads(line) for line in CORPUS_PATH.read_text().splitlines() if line.strip()]
    bench_ids = {json.loads(line)["corpus_id"] for line in BENCH_PATH.read_text().splitlines() if line.strip()}
    rows = []
    for r in corpus:
        if r["corpus_id"] in bench_ids:
            continue
        rows.append({"instruction": r["question"], "output": r["golden_answer"]})
    with SFT_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} SFT rows ({len(corpus) - len(rows)} held out on bench) to {SFT_PATH.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-sft", action="store_true", help="write data/toolcall_sft.jsonl from corpus minus bench")
    args = ap.parse_args()
    if args.make_sft:
        make_sft()
        return
    if not CORPUS_PATH.exists():
        raise SystemExit(f"corpus not found: {CORPUS_PATH}")
    corpus = [json.loads(line) for line in CORPUS_PATH.read_text().splitlines() if line.strip()]
    last_id = max(int(r["corpus_id"].split("-")[1]) for r in corpus if ID_RE.match(r.get("corpus_id", "")))
    suite_keys = {
        "suites:home": ("query_energy", "schedule_scene", "set_thermostat", "turn_off_device"),
        "suites:crm": ("create_ticket", "resolve_ticket", "search_kb", "update_plan"),
        "suites:pay": ("categorize_transaction", "get_balance", "send_money"),
    }
    suite_maps = {}
    for label, toolnames in suite_keys.items():
        for r in corpus:
            payload = json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"]
            names = tuple(sorted(t["name"] for t in payload["tools"]))
            if names == tuple(sorted(toolnames)):
                suite_maps[label] = payload["tools"]
                break
        if label not in suite_maps:
            raise SystemExit(f"could not find existing toolset for {label} in corpus")
    new_rows = []
    cid = last_id + 1
    for label in ("suites:home", "suites:crm", "suites:pay"):
        cid = _expand_suite(label, new_rows, cid)
    dedup = {}
    for cid, suite, payload, variant in new_rows:
        if payload["request"] not in dedup:
            dedup[payload["request"]] = (cid, suite, payload, variant)
    new_rows = list(dedup.values())
    _write(new_rows, suite_maps, last_id + 1)


if __name__ == "__main__":
    main()