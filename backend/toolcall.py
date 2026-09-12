import json

from . import config, gold

NICHE = "tool-call"

SYSTEM_PROMPT = (
    "You are an assistant that routes a user request to exactly ONE tool from the "
    "toolset given in the user message, or correctly declines when no tool fits.\n"
    "Reply with a single JSON object in exactly one of these shapes:\n"
    '{"tool": "<tool name>", "arguments": {"<argument name>": <value>}}\n'
    '{"tool": null}\n'
    "Rules: pick the tool and argument values that best satisfy the request; argument "
    "values must match the declared JSON types and enum members exactly; if a required "
    "argument value is missing, or no tool fits the request, emit {\"tool\": null}.\n"
    "Your entire reply must be that single JSON object and nothing else."
)

_DECLINE_RUBRIC = config.BASE_DIR / "prompts" / "toolcall_decline.md"


def _payload(item: dict) -> dict:
    pl = item.get("payload")
    if isinstance(pl, str):
        return json.loads(pl)
    if isinstance(pl, dict):
        return pl
    return item


def expected(item: dict) -> dict | None:
    return _payload(item).get("expected")


def expects_decline(item: dict) -> bool:
    exp = expected(item)
    return not (exp and exp.get("tool"))


def render_question(payload: dict) -> str:
    tools = sorted(payload["tools"], key=lambda t: t["name"])
    lines = ["TOOLS (JSON schema):"]
    for t in tools:
        lines.append(
            json.dumps(
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    lines.append("")
    lines.append(f"USER REQUEST: {payload['request']}")
    return "\n".join(lines)


def golden_text(payload: dict) -> str:
    exp = payload.get("expected")
    if not (exp and exp.get("tool")):
        return json.dumps({"tool": None}, ensure_ascii=False)
    return json.dumps(
        {"tool": exp["tool"], "arguments": exp.get("arguments") or {}},
        ensure_ascii=False,
    )


def _first_json_object(text: str) -> str | None:
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if start < 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return None


def parse_output(text: str) -> tuple[str, dict | None]:
    if not text or not text.strip():
        return ("empty", None)
    obj_text = _first_json_object(text)
    if obj_text is None:
        return ("empty", None)
    try:
        obj = json.loads(obj_text)
    except json.JSONDecodeError:
        return ("invalid", None)
    if not isinstance(obj, dict):
        return ("invalid", None)
    tool = obj.get("tool", obj.get("name"))
    if tool is None or (isinstance(tool, str) and tool.strip().lower() in ("null", "none", "no_tool", "")):
        return ("decline", None)
    if not isinstance(tool, str) or not tool.strip():
        return ("invalid", None)
    arguments = obj.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return ("invalid", None)
    return ("call", {"tool": tool.strip(), "arguments": arguments})


def _check_value(value, spec, path: str) -> tuple[bool, str]:
    if not isinstance(spec, dict):
        return True, ""
    if "enum" in spec and value not in spec["enum"]:
        return False, f"{path}: {value!r} not in enum {spec['enum']}"
    st = spec.get("type")
    if st is None:
        return True, ""
    if st == "string":
        ok = isinstance(value, str)
    elif st == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif st == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif st == "boolean":
        ok = isinstance(value, bool)
    elif st == "array":
        ok = isinstance(value, list)
    elif st == "object":
        ok = isinstance(value, dict)
    elif st == "null":
        ok = value is None
    else:
        ok = True
    if not ok:
        return False, f"{path}: expected {st}, got {type(value).__name__}"
    if isinstance(value, dict) and spec.get("properties"):
        props = spec["properties"] or {}
        for key, val in value.items():
            if key not in props and spec.get("additionalProperties", True) is False:
                return False, f"{path}.{key}: unexpected property"
            if key in props:
                ok2, why = _check_value(val, props[key], f"{path}.{key}")
                if not ok2:
                    return False, why
    if isinstance(value, list) and spec.get("items"):
        for i, v in enumerate(value):
            ok2, why = _check_value(v, spec["items"], f"{path}[{i}]")
            if not ok2:
                return False, why
    return True, ""


def _validate_call(call: dict, payload: dict) -> tuple[bool, str]:
    tool_spec = {t["name"]: t for t in payload["tools"]}.get(call["tool"])
    if tool_spec is None:
        return False, f"unknown tool {call['tool']!r}"
    schema = tool_spec.get("parameters") or {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    args = call["arguments"]
    for key in required:
        if key not in args:
            return False, f"missing required argument {key!r}"
    for key, val in args.items():
        if key not in props and schema.get("additionalProperties", True) is False:
            return False, f"unexpected argument {key!r}"
        if key in props:
            ok, why = _check_value(val, props[key], key)
            if not ok:
                return False, why
    return True, ""


def is_correct(emitted: str, item: dict) -> dict:
    kind, parsed = parse_output(emitted)
    result = {
        "correct": False,
        "kind": kind,
        "reason": "",
        "emitted": emitted,
        "emitted_call": None,
    }
    if kind == "invalid":
        result["reason"] = "unparseable function-call JSON"
        return result
    if kind in ("empty", "decline"):
        if expects_decline(item):
            result["correct"] = True
            result["reason"] = "correctly declined; no tool fits"
        else:
            result["reason"] = f"declined but a call to {expected(item).get('tool')!r} was expected"
        return result
    if expects_decline(item):
        result["reason"] = f"called {parsed['tool']!r} but no tool fits (decline expected)"
        result["emitted_call"] = parsed
        return result
    exp = expected(item)
    if parsed["tool"] != exp.get("tool"):
        result["reason"] = f"wrong tool: {parsed['tool']!r}, expected {exp.get('tool')!r}"
        result["emitted_call"] = parsed
        return result
    ok, why = _validate_call(parsed, _payload(item))
    if not ok:
        result["reason"] = f"arguments invalid for {parsed['tool']}: {why}"
        result["emitted_call"] = parsed
        return result
    if parsed["arguments"] != (exp.get("arguments") or {}):
        result["reason"] = "arguments validate but differ from the expected call"
        result["emitted_call"] = parsed
        result["expected_arguments"] = exp.get("arguments") or {}
        return result
    result["correct"] = True
    result["reason"] = f"correct call to {parsed['tool']}"
    result["emitted_call"] = parsed
    return result


def base_answer(item: dict) -> str:
    question = item.get("question") or render_question(_payload(item))
    return gold.chat(SYSTEM_PROMPT, question).strip()


def taint_answer(item: dict) -> str:
    payload = _payload(item)
    golden = golden_text(payload)
    if expects_decline(item):
        sys = (
            "You write a single subtly-wrong function-call JSON for a tool-use QA exercise. "
            "The CORRECT answer is to decline (no tool fits) by emitting {\"tool\": null}. "
            "Produce a wrong-but-plausible variant that CALLS one of the tools anyway with "
            "invented but schema-shaped arguments, so it looks convincing at a glance. "
            "Stay close in length and structure to the correct answer. Output ONLY the JSON object."
        )
    else:
        sys = (
            "You write a single subtly-wrong function-call JSON for a tool-use QA exercise. "
            "Take the correct call and produce a variant that LOOKS right but changes exactly "
            "one thing: the wrong tool with plausible arguments, a wrong argument value, an "
            "off-by-one number, a wrong enum member, or a wrong argument type. Stay close in "
            "length and structure to the correct call. Output ONLY the JSON object."
        )
    user = (
        f"tool schemas:\n{json.dumps(payload['tools'], ensure_ascii=False)}\n"
        f"correct answer:\n{golden}\n"
        f"user request:\n{payload['request']}"
    )
    return gold.chat(sys, user).strip()


def judge_decline(item: dict) -> dict:
    payload = _payload(item)
    user = (
        f"tool schemas:\n{json.dumps(payload['tools'], ensure_ascii=False)}\n"
        f"user request:\n{payload['request']}\n"
        f"model response: {{\"tool\": null}}\n"
    )
    out = gold.chat(_DECLINE_RUBRIC.read_text(), user, model=config.JUDGE_MODEL, json_mode=True)
    data = json.loads(out)
    data["pass"] = bool(data.get("pass", False))
    return data