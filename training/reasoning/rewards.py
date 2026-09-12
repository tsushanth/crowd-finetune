from . import formats


def exact_match_reward(prompts, completions, answer, **kwargs):
    return [float(formats.exact_match(c, a)) for c, a in zip(completions, answer)]


def format_reward(prompts, completions, **kwargs):
    out = []
    for c in completions:
        parsed = formats.parse_completion(c)
        reasoning_ok = bool(parsed["reasoning"])
        answer_ok = bool(parsed["answer"])
        if reasoning_ok and answer_ok:
            out.append(1.0)
        elif answer_ok:
            out.append(-0.5)
        else:
            out.append(-2.0)
    return out


def length_reward(prompts, completions, **kwargs):
    return [max(0.0, 1 - abs(len(c) - 400) / 600) for c in completions]