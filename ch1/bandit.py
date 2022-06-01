import numpy as np


class Bandit:
    def __init__(self, arms: int = 10):
        self.rates: np.ndarray = np.random.rand(arms)
        print(type(self.rates))

    def play(self, arm) -> int:
        rate: float = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0

def main():
    bandit = Bandit()
    print(bandit.rates[0])
    Q: float = 0
    for n in range(1, 101):
        reward = bandit.play(0)
        Q += (reward - Q) / n

    print(Q)

if __name__ == "__main__":
    main()