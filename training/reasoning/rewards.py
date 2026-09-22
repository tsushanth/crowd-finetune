from . import formats, prm


def exact_match_reward(prompts, completions, answer, **kwargs):
    return [float(formats.numeric_match(c, a)) for c, a in zip(completions, answer)]


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


def make_prm_reward(prm_path, aggregate="min", device=None, batch_size=16):
    """Process reward: PRM score of the reasoning chain in [0, 1], replacing exact-match.

    A completion with no parseable <reasoning> block scores 0. Needs a `question`
    column in the GRPO dataset (TRL forwards extra columns as kwargs).
    """
    state = {}

    def prm_reward(prompts, completions, question, **kwargs):
        if "m" not in state:
            import torch

            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if dev == "cuda" else torch.float32
            state["m"] = prm.PRM.load(prm_path, dev, dtype)
        steps = [prm.split_steps(formats.parse_completion(c)["reasoning"]) for c in completions]
        idx = [i for i, s in enumerate(steps) if s]
        out = [0.0] * len(completions)
        if idx:
            probs = state["m"].score_steps([question[i] for i in idx], [steps[i] for i in idx], batch_size)
            for i, p in zip(idx, probs):
                out[i] = prm.aggregate(p, aggregate)
        return out

    return prm_reward
