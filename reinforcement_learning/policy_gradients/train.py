#!/usr/bin/env python3
"""
Defines function to implement full training with policy gradient
"""


import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    env: initial environment
    nb_episodes [int]: the number of episodes used for training
    alpha [float]: the learning rate
    gamma [float]: the discount factor
    show_result [boolean]:
    determines if the environment is rendered every 1000 episodes
    """

    weight = np.random.rand(4, 2)
    all_scores = []
    for episode in range(nb_episodes):
        state = env.reset()[None, :]
        gradients = []
        rewards = []
        sum_rewards = 0
        
        while True:
            if show_result and (episode % 1000 == 0):
                env.render()
            action, gradient = policy_gradient(state, weight)
            next_state, reward, done, info = env.step(action)

            gradients.append(gradient)
            rewards.append(reward)
            sum_rewards += reward

            if done:
                break
            state = next_state[None, :]

        for i in range(len(gradients)):
            weight += (alpha * gradients[i] *
                       sum([r * (gamma ** r) for t, r in enumerate(
                           rewards[i:])]))

        all_scores.append(sum_rewards)
        print("{}: {}".format(episode, sum_rewards), end="\r", flush=False)

    return all_scores
