# 03_euler_maruyama.py
# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from aim.model import BETA, grad_F

# %%
# 03_euler_maruyama.py
# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from aim.model import BETA, grad_F

# %%
from aim.simulation import euler_maruyama
# %%
#we put a random individual trajectory to see how it looks like. These values are not participant estimates from the paper.
# %%
y0 = np.array([0.95, 0.05])

times, trajectory = euler_maruyama(
    y0=y0,
    dt=0.0005,
    T=10,
    D=0.01,
    seed=42
)

print("Initial state:", trajectory[0])
print("Final state:", trajectory[-1])
plt.figure(figsize=(9, 5))

plt.plot(times, trajectory[:, 0], label="Positive affect y1")
plt.plot(times, trajectory[:, 1], label="Negative affect y2")

plt.xlabel("Time")
plt.ylabel("Affective state")
plt.title("Euler-Maruyama trajectory")
plt.ylim(0, 1)
plt.legend()

plt.show()
lower_hits = np.sum(trajectory <= 0.00001)
upper_hits = np.sum(trajectory >= 0.99999)

print("Lower boundary hits:", lower_hits)
print("Upper boundary hits:", upper_hits)
