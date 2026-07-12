"""Metrics placeholders used across experiments."""
import numpy as np


def pair_accuracy(chosen_scores, rejected_scores):
    return float(np.mean(np.asarray(chosen_scores) > np.asarray(rejected_scores)))


def mean_margin(chosen_scores, rejected_scores):
    return float(np.mean(np.asarray(chosen_scores) - np.asarray(rejected_scores)))


def expected_reward(policy_probs, rewards):
    return np.sum(np.asarray(policy_probs) * np.asarray(rewards), axis=-1)
