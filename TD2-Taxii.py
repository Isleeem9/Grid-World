import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
   
    Q[s, a] = Q[s, a] + alpha * (
        r + gamma * np.max(Q[sprime]) - Q[s, a]
    )
    return Q


def epsilon_greedy(Q, s, epsilone):
  
    if np.random.rand() < epsilone:
        return np.random.randint(Q.shape[1])  
    else:
        return np.argmax(Q[s])  


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    state, info = env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.01
    gamma = 0.8
    epsilon = 0.2

    n_epochs = 20
    max_itr_per_epoch = 100
    rewards = []

    for e in range(n_epochs):
        total_reward = 0
        S, info = env.reset()

        for _ in range(max_itr_per_epoch):

            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, terminated, truncated, info = env.step(A)

            total_reward += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            S = Sprime

            if terminated or truncated:
                break

        print("episode #", e, "| recompense =", total_reward)
        rewards.append(total_reward)

    print("\nRecompense moyenne =", np.mean(rewards))
    

    env.close()
