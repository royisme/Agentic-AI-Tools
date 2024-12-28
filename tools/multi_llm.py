#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal LLM query script supporting multiple providers:
- openai (GPT-3.5, GPT-4, etc.)
- anthropic (Claude)
- ollama (local Llama, etc.)
- localhttp (custom REST API, e.g. Qwen, GPT4All, etc.)
- gemini (Google's Gemini 2.0 via google-genai SDK)

Usage examples:
  python multi_llm.py --provider openai --model gpt-3.5-turbo --prompt "Hello world"
  python multi_llm.py --provider anthropic --model claude-2 --prompt "Write a poem"
  python multi_llm.py --provider ollama --model llama2-7b --prompt "Translate to French"
  python multi_llm.py --provider localhttp --model Qwen/Qwen2.5-32B-Instruct-AWQ --prompt "Your question"
  python multi_llm.py --provider gemini --model gemini-2.0-flash-exp --prompt "What's the weather today?"
"""

import argparse
import os
import sys
import requests

# 如果要用到 openai / anthropic / google-genai，需要先安装:
#   pip install openai anthropic google-genai requests

# 尝试导入 openai 库
try:
    import openai
except ImportError:
    openai = None

# 尝试导入 anthropic 库
try:
    import anthropic
except ImportError:
    anthropic = None

# ------------------------------------------------------------------------
# 1) Ollama - 调用本地 Ollama HTTP 接口
# ------------------------------------------------------------------------
def query_ollama(prompt: str, model: str = "llama2", **kwargs) -> str:
    """
    Call a local Ollama server via HTTP.
    Default port is typically http://localhost:11411; you can override with env var OLLAMA_URL.
    """
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11411")
    payload = {
        "prompt": prompt,
        "model": model,
        # 可以根据 Ollama 的版本和需求，添加更多超参，比如 temperature, top_p, repeat_penalty 等
    }
    try:
        resp = requests.post(f"{ollama_url}/generate", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        # Ollama 的返回格式可能是流式，需要拼接，也可能一次性
        # 假设最终结果在 data["done"] 或 data["content"]，请根据实际接口进行调整
        if "done" in data:
            return data["done"]
        elif "content" in data:
            return data["content"]
        else:
            return str(data)
    except Exception as e:
        return f"[Ollama Error] {e}"


# ------------------------------------------------------------------------
# 2) OpenAI - 调用官方 OpenAI API (GPT-3.5, GPT-4等)
# ------------------------------------------------------------------------
def query_openai(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7, **kwargs) -> str:
    """
    Call the OpenAI ChatCompletion API.
    Must have openai>=0.27.0 installed and OPENAI_API_KEY set in env.
    """
    if openai is None:
        return "[Error] openai library not installed. Please pip install openai."
    openai.api_key = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_KEY")
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **kwargs
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[OpenAI Error] {e}"


# ------------------------------------------------------------------------
# 3) Anthropic - 调用 Claude (anthropic SDK)
# ------------------------------------------------------------------------
def query_anthropic(prompt: str, model: str = "claude-2", temperature: float = 0.7, **kwargs) -> str:
    """
    Call Anthropic (Claude) API via anthropic package.
    Need ANTHROPIC_API_KEY in env, e.g., pip install anthropic
    """
    if anthropic is None:
        return "[Error] anthropic library not installed. Please pip install anthropic."
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", None)
    if not anthropic_api_key:
        return "[Error] ANTHROPIC_API_KEY not set in environment."

    client = anthropic.Client(anthropic_api_key)
    # Claude需要将用户内容放到 anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT
    chat_prompt = f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}"
    try:
        resp = client.completions.create(
            model=model,
            prompt=chat_prompt,
            max_tokens_to_sample=1024,
            temperature=temperature,
            **kwargs
        )
        return resp.completion.strip()
    except Exception as e:
        return f"[Anthropic Error] {e}"


# ------------------------------------------------------------------------
# 4) localhttp - 调用自建 REST API (如 Qwen, GPT4All, 你之前的 base_url 等)
# ------------------------------------------------------------------------
def query_localhttp(prompt: str, model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ", temperature: float = 0.7, **kwargs) -> str:
    """
    Generic local server endpoint, e.g. your own text-generation web service.
    By default it tries: http://192.168.180.137:8006/v1
    """
    base_url = os.environ.get("LOCAL_LLM_URL", "http://192.168.180.137:8006/v1")
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
        # 可根据后端需要，传其他参数。比如 top_p, max_tokens, etc.
    }
    try:
        resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        # 假设此后端兼容 OpenAI ChatCompletion 风格
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            return str(data)
    except Exception as e:
        return f"[LocalHTTP Error] {e}"


# ------------------------------------------------------------------------
# 5) Gemini 2.0 - 调用 Google GenAI (google-genai SDK)
# ------------------------------------------------------------------------
def query_gemini(prompt: str, model: str = "gemini-2.0-flash-exp", **kwargs) -> str:
    """
    Call Google's Gemini 2.0 via google-genai.
    Requirements:
      pip install google-genai
    Environment:
      os.environ['GOOGLE_API_KEY'] must be set with a valid GenAI key.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "[Error] google-genai library not installed. Please `pip install google-genai`."

    api_key = os.environ.get("GOOGLE_API_KEY", None)
    if not api_key:
        return "[Gemini Error] GOOGLE_API_KEY not set in environment."

    # 创建客户端：v1alpha
    client = genai.Client(http_options={'api_version': 'v1alpha'})

    try:
        # 创建一次性 chat 会话
        chat = client.chats.create(model=model, **kwargs)
        r = chat.send_message(prompt)
        if not r.candidates:
            return "[Gemini Error] No candidates in response."

        # Gemini 2.0 的每个 candidate 有 content.parts 列表
        parts = r.candidates[0].content.parts
        if not parts:
            # 如果 parts 为空，就检查是否有 finish_reason
            finish_reason = r.candidates[0].finish_reason
            return f"[Gemini Notice] finish_reason={finish_reason}"

        # 组装文本
        text_segments = []
        for part in parts:
            if part.text:
                text_segments.append(part.text)
            elif part.executable_code:
                # code block
                text_segments.append(part.executable_code.code)
        return "\n".join(text_segments)

    except Exception as e:
        return f"[Gemini Error] {e}"


# ------------------------------------------------------------------------
# 6) 统一的调度函数，根据 provider 调用对应方法
# ------------------------------------------------------------------------
def query_llm(provider: str, prompt: str, model: str, temperature: float = 0.7, **kwargs) -> str:
    provider = provider.lower()
    if provider == "openai":
        return query_openai(prompt=prompt, model=model, temperature=temperature, **kwargs)
    elif provider == "anthropic":
        return query_anthropic(prompt=prompt, model=model, temperature=temperature, **kwargs)
    elif provider == "ollama":
        return query_ollama(prompt=prompt, model=model, **kwargs)
    elif provider == "localhttp":
        return query_localhttp(prompt=prompt, model=model, temperature=temperature, **kwargs)
    elif provider == "gemini":
        return query_gemini(prompt=prompt, model=model, **kwargs)
    else:
        return f"[Error] Unknown provider '{provider}'. Supported: openai, anthropic, ollama, localhttp, gemini"


def main():
    parser = argparse.ArgumentParser(
        description='Query various LLM providers (OpenAI, Anthropic, Ollama, localhttp, Gemini) with a prompt.'
    )
    parser.add_argument('--provider', type=str, required=True,
                        choices=["openai", "anthropic", "ollama", "localhttp", "gemini"],
                        help='Which provider to call.')
    parser.add_argument('--prompt', type=str, required=True, help='The prompt to send to the LLM.')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                        help='Model name (depends on provider).')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature (default=0.7).')

    args = parser.parse_args()

    # 调用统一的函数
    response = query_llm(
        provider=args.provider,
        prompt=args.prompt,
        model=args.model,
        temperature=args.temperature
    )
    print(response)

if __name__ == "__main__":
    main()

