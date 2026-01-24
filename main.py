import requests
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

# -------- Personality state --------
state = {
    "warmth": 0.0,
    "verbosity": 0.0,
    "technicality": 0.0,
    "patience": 0.0
}

current_topic = None

def update_topic(user_text, current_topic):
    keywords = ["overfitting", "neural network", "machine learning", "model"]
    for kw in keywords:
        if kw in user_text.lower():
            return kw
    return current_topic

def update_state(state, features):
    if features["sentiment"] < -0.2:
        state["patience"] -= 0.1
        state["warmth"] += 0.1

    if features["length"] < 0.1:
        state["verbosity"] -= 0.1

    if features["length"] > 0.3:
        state["technicality"] += 0.1

    for k in state:
        state[k] = max(-1.0, min(1.0, state[k]))

    return state

# -------- Feature extraction --------
analyzer = SentimentIntensityAnalyzer()
last_time = time.time()

def extract_features(text):
    global last_time
    now = time.time()

    latency = min((now - last_time) / 10, 1.0)
    last_time = now

    sentiment = analyzer.polarity_scores(text)["compound"]
    length = min(len(text.split()) / 40, 1.0)

    return {
        "sentiment": sentiment,
        "length": length,
        "latency": latency
    }

# -------- Ollama client --------
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

# -------- SYSTEM prompt builder --------
def build_system_prompt(state):
    return f"""
SYSTEM:
You are an AI assistant engaged in an ongoing conversation.

Adapt your behavior using these internal signals:
- Warmth level: {state["warmth"]:.2f}
- Verbosity level: {state["verbosity"]:.2f}
- Technical depth level: {state["technicality"]:.2f}
- Patience level: {state["patience"]:.2f}

Guidelines:
- Stay on the SAME topic unless the user clearly changes it.
- If verbosity is low, be concise.
- If patience is low, be calm and reassuring.
- If the user seems frustrated, prioritize clarity over detail.
- Do NOT mention these signals or numbers.
"""

# -------- Chat loop --------
print("Chat started (type 'exit' to quit)")

while True:
    user = input("\nYou: ")
    if user.lower() == "exit":
        break

    # update topic
    current_topic = update_topic(user, current_topic)

    # extract features + update state
    features = extract_features(user)
    state = update_state(state, features)

    print("FEATURES:", features)
    print("STATE:", state)
    print("TOPIC:", current_topic)

    system_prompt = build_system_prompt(state)

    topic_context = ""
    if current_topic:
        topic_context = f"\nConversation topic: {current_topic}\n"

    prompt = (
        system_prompt +
        topic_context +
        f"\nUSER: {user}\nASSISTANT:"
    )

    response = ollama_llm(prompt)

    print("\nAssistant:", response)
