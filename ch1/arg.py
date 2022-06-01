import numpy as np


def main():
    np.random.seed(0)
    rewards = []

    Q: float = 0

    for n in range(1, 11):
        reward: float = np.random.rand()
        # Q = Q + (reward - Q) / n
        Q += (reward - Q) / n
        print(Q)

if __name__ == "__main__":
    main()

