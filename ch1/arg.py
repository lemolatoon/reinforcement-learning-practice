import numpy as np

def main():
    np.random.seed(0)
    rewards = []

    for n in range(1, 11):
        reward: float = np.random.rand()
        rewards.append(reward)
        Q: float = sum(rewards) / n
        print(Q)

if __name__ == "__main__":
    main()

