import requests

def ollama_llm(prompt, model="qwen2:7b"):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return res.json()["response"]

print(ollama_llm("Explain overfitting in simple terms."))
