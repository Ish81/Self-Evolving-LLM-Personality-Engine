import numpy as np
import matplotlib.pyplot as plt

states = np.load("state_trajectory.npy")

labels = ["warmth", "verbosity", "technicality", "patience"]

plt.figure(figsize=(10, 6))

for i, label in enumerate(labels):
    plt.plot(states[:, i], label=label)

plt.xlabel("Conversation turn")
plt.ylabel("State value")
plt.title("Liquid Personality State Evolution")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 🔥 SAVE INSTEAD OF SHOW
plt.savefig("liquid_personality_states.png")
print("Plot saved as liquid_personality_states.png")
