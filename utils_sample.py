import d4rl
import gym
import numpy as np
import torch
import time 
from jax import device_get

def parallel_simple_eval_policy(policy_fn, env_name, seed, eval_episodes = 20):
    """
    Evaluate a policy function in parallel across multiple episodes and environments.

    Args:
    - policy_fn: A function that takes state tensors and outputs actions.
    - env_name: Name of the Gym environment.
    - seed: Random seed for environment initialization.
    - eval_episodes: Number of parallel episodes to run.

    Returns:
    - Tuple containing the mean and standard deviation of the normalized scores.
    """
    environments = [gym.make(env_name) for _ in range(eval_episodes)]
    for i, env in enumerate(environments):
        env.seed(seed + 1001 + i)
        env.buffer_state = env.reset()
        env.buffer_return = 0.0

    time_cost = 0.0
    query_times = 0
    # Process environments in parallel
    active_envs = environments.copy()
    while active_envs:
        states = np.array([env.buffer_state for env in active_envs])
        #states_tensor = torch.tensor(states, device="cuda", dtype=torch.float32)
        start_time = time.time()
        with torch.no_grad():
            actions = policy_fn.sample_actions(states, temperature=0.0)
            #actions = policy_fn(states_tensor).detach().cpu().numpy()
        #device_get(actions)
        end_time = time.time()
        time_cost += end_time - start_time
        query_times += 1
        
        next_envs = []
        for env, action in zip(active_envs, actions):
            state, reward, done, _ = env.step(action)
            env.buffer_return += reward
            env.buffer_state = state
            if not done:
                next_envs.append(env)
        active_envs = next_envs

    # Calculate normalized scores
    normalized_scores = [d4rl.get_normalized_score(env_name, env.buffer_return) for env in environments]
    mean_score = np.mean(normalized_scores)
    std_score = np.std(normalized_scores)

    return mean_score, std_score, time_cost, query_times
