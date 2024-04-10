import matplotlib.pyplot as plt
import seaborn as sns

res = {
    1: 0.6776,
    2: 0.7493,
    4: 0.7764,
    8: 0.8184,
    16: 0.9474,
    32: 0.9748,
    64: 0.9898
}

x, y = zip(*res.items())
labels = [f"{t}%" for t in x]
print(plt.style.available)
plt.style.use("ggplot")
plt.plot(labels, y, linestyle='--', marker='o')
plt.xlabel("Dataset size")
plt.ylabel("Accuracy")
plt.savefig("./figs/simple.png", format="png", bbox_inches="tight")