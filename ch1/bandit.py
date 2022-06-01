import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

np.random.seed(0)


class Bandit:
    def __init__(self, arms: int = 10):
        self.arms = arms
        self.rates: np.ndarray = np.random.rand(arms)

    def play(self, arm: int) -> int:
        rate: float = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0


class NonStatBandit(Bandit):
    def play(self, arm: int) -> int:
        rate: float = self.rates[arm]
        sign = 1 if np.random.random() > 0.5 else -1
        delta = 0.5 * (np.random.randn(self.arms) * sign)
        self.rates += delta   # add noise
        if rate > np.random.rand():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon: float, action_size: int = 10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action: int, reward: int):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)  # max_n (Q_n)


def main():
    avg_total_rewards = []
    avg_rates = []
    for i in range(500):
        total_reward, rates = train()
        avg_total_rewards.append(total_reward)
        avg_rates.append(rates)
    avg_total_rewards = np.array(avg_total_rewards)
    avg_rates = np.array(avg_rates)

    avg_total_rewards = np.mean(avg_total_rewards, axis=0)
    avg_rates = np.mean(avg_rates, axis=0)
    plt_show(avg_total_rewards, avg_rates)


def train() -> Tuple[List[float], List[float]]:
    steps: int = 1000
    epsilon: float = 0.1

    bandit = NonStatBandit()
    agent = Agent(epsilon)
    total_reward: int = 0
    total_rewards: List[int] = []
    rates: List[float] = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))

    return total_rewards, rates


def plt_show(total_rewards: List[float], rates: List[float]):
    plt.ylabel("Total reward")
    plt.xlabel("Steps")
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(rates)
    plt.show()


if __name__ == "__main__":
    main()
