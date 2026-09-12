from . import formats, sandbox as _sandbox

_sbx = _sandbox.Sandbox()


def _reward_call(reward, **kwargs):
    hard = kwargs.pop("hard", False)
    timeout = kwargs.pop("timeout", None)
    return reward(**kwargs, hard=hard, timeout=timeout)


def test_pass_reward(prompts, completions, tests, entry_point, imports, **kwargs):
    if not completions:
        return []
    hard = kwargs.pop("hard", False)
    timeout = kwargs.pop("timeout", None)
    out = []
    for c, t, e, i in zip(completions, tests, entry_point, imports):
        problem = {"tests": list(t), "entry_point": e, "imports": list(i)}
        res = _sbx.run(formats.extract_code(c), problem, timeout=timeout)
        frac = res["fraction"] if not res.get("timeout") else 0.0
        if hard:
            out.append(1.0 if res.get("ok") else 0.0)
        else:
            out.append(frac)
    return out


def code_format_reward(prompts, completions, **kwargs):
    out = []
    for c in completions:
        parsed = formats.parse_completion(c)
        reasoning_ok = bool(parsed["reasoning"])
        answer_ok = bool(parsed["answer"])
        if reasoning_ok and answer_ok:
            out.append(0.5)
        elif answer_ok:
            out.append(0.0)
        else:
            out.append(-1.0)
    return out