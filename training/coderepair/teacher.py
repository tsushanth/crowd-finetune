import time
from pathlib import Path

import httpx

from backend import config

from . import formats


class Teacher:
    def __init__(self, model=None, timeout=180.0):
        self.model = model or config.TEACHER_MODEL
        root = Path(__file__).resolve().parent
        local = root / self.model
        self._local_path = str(local) if local.exists() else None
        self._local = None
        self._tokenizer = None
        self._timeout = timeout
        if self._local_path:
            return
        if not config.BASE_LLM_API_KEY:
            raise RuntimeError("BASE_LLM_API_KEY is not set")
        self._endpoint = config.BASE_LLM_URL.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {config.BASE_LLM_API_KEY}"}

    def generate(self, question, temperature=0.5, max_tokens=1536, retries=3):
        if self._local_path:
            return self._local_generate(question, temperature, max_tokens)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": formats.CODE_SYSTEM},
                {"role": "user", "content": question},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for attempt in range(retries):
            try:
                resp = httpx.post(
                    self._endpoint, headers=self._headers, json=payload,
                    timeout=self._timeout,
                )
                if resp.status_code in (429, 408) or resp.status_code >= 500:
                    time.sleep(5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"].get("content") or ""
                return content.strip()
            except (httpx.HTTPStatusError, httpx.HTTPError, httpx.TimeoutException):
                if attempt == retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))
        raise RuntimeError("teacher request failed")

    def _local_generate(self, question, temperature=0.5, max_tokens=1536):
        if self._local is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._local_path)
            self._local = AutoModelForCausalLM.from_pretrained(
                self._local_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self._local.eval()
        messages = [
            {"role": "system", "content": formats.CODE_SYSTEM},
            {"role": "user", "content": question},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._local.device)
        with torch.no_grad():
            out = self._local.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
            )
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()