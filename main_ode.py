import requests
import time
import torch
import torch.nn as nn
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "qwen2:7b"

STATE_NAMES = ["warmth", "verbosity", "technicality", "patience"]
STATE_DIM = len(STATE_NAMES)
FEATURE_DIM = 3  # sentiment, length, latency

DT = 0.2        # time step
STEPS = 5       # ODE integration steps
DAMPING = 0.05  # stability factor (STEP 7)

# ============================================================
# INITIAL STATE
# ============================================================
state = torch.zeros(STATE_DIM)

# ============================================================
# STATE LOGGING (STEP 8)
# ============================================================
state_history = []

# ============================================================
# TOPIC TRACKING
# ============================================================
current_topic = None

def update_topic(user_text, current_topic):
    keywords = [
        "overfitting",
        "underfitting",
        "neural network",
        "machine learning",
        "deep learning",
        "model",
        "training data"
    ]
    for kw in keywords:
        if kw in user_text.lower():
            return kw
    return current_topic

# ============================================================
# FEATURE EXTRACTION
# ============================================================
analyzer = SentimentIntensityAnalyzer()
last_time = time.time()

def extract_features(text):
    global last_time
    now = time.time()

    latency = min((now - last_time) / 10, 1.0)
    last_time = now

    sentiment = analyzer.polarity_scores(text)["compound"]
    length = min(len(text.split()) / 40, 1.0)

    return torch.tensor(
        [sentiment, length, latency],
        dtype=torch.float32
    )

# ============================================================
# LIQUID NEURAL NETWORK (VECTOR FIELD)
# ============================================================
class LiquidPersonalityNN(nn.Module):
    def __init__(self, state_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + feature_dim, 32),
            nn.Tanh(),
            nn.Linear(32, state_dim)
        )

    def forward(self, state, features):
        x = torch.cat([state, features])
        return self.net(x)  # dstate/dt

lnn = LiquidPersonalityNN(STATE_DIM, FEATURE_DIM)

# ============================================================
# MANUAL NEURAL ODE (STEP 7: STABLE + WARMTH BIAS)
# ============================================================
def evolve_state(state, features):
    for _ in range(STEPS):
        dstate = lnn(state, features)

        # Safety clamp
        dstate = torch.clamp(dstate, -0.5, 0.5)

        # Encourage warmth when patience is low
        patience_idx = STATE_NAMES.index("patience")
        warmth_idx = STATE_NAMES.index("warmth")

        if state[patience_idx] < -0.2:
            dstate[warmth_idx] += 0.05

        # Euler integration + damping
        state = state + DT * dstate
        state = state - DAMPING * state

        # Keep bounded
        state = torch.tanh(state)

    return state

# ============================================================
# SYSTEM PROMPT
# ============================================================
def build_system_prompt(state_dict):
    return f"""
SYSTEM:
You are an AI assistant engaged in an ongoing conversation.

Internal behavioral signals (do not mention explicitly):
- Warmth: {state_dict["warmth"]:.2f}
- Verbosity: {state_dict["verbosity"]:.2f}
- Technical depth: {state_dict["technicality"]:.2f}
- Patience: {state_dict["patience"]:.2f}

Guidelines:
- Stay on the SAME topic unless the user clearly changes it.
- If verbosity is low, be concise.
- If patience is low, be calm and reassuring.
- If the user seems frustrated, prioritize clarity over detail.
- Do NOT mention these signals or numbers.
"""

# ============================================================
# OLLAMA CLIENT
# ============================================================
def ollama_llm(prompt, model=MODEL_NAME):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return res.json()["response"]

# ============================================================
# CHAT LOOP
# ============================================================
print("🌊 Liquid Personality LLM (Manual Neural ODE)")
print("Type 'exit' to quit")

while True:
    user = input("\nYou: ")
    if user.lower() == "exit":
        break

    # Topic continuity
    current_topic = update_topic(user, current_topic)

    # Feature extraction
    features = extract_features(user)

    # Evolve personality
    state = evolve_state(state, features)

    # Convert state to readable form
    state_dict = {
        name: state[i].item()
        for i, name in enumerate(STATE_NAMES)
    }

    # ✅ LOG STATE (STEP 8)
    state_history.append([state_dict[name] for name in STATE_NAMES])

    print("STATE:", state_dict)
    print("TOPIC:", current_topic)

    system_prompt = build_system_prompt(state_dict)

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

# ============================================================
# SAVE STATE TRAJECTORY (STEP 8)
# ============================================================
np.save("state_trajectory.npy", np.array(state_history))
print("✅ Saved personality trajectory to state_trajectory.npy")
