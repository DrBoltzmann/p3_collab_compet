import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from ddpg_agent import Agent
from collections import deque
from datetime import datetime, timedelta
from unityagents import UnityEnvironment

env = UnityEnvironment(file_name = 'Tennis.app')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode = True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])
print('The state for the second agent looks like:', states[1])

agents = Agent(state_size = state_size, 
               action_size = action_size,
               num_agents = num_agents, 
               random_seed = 0)
print(agents.actor_local)
print(agents.critic_local)

def ddpg(n_episodes = 5000, max_t = 1000):
    scores_deque = deque(maxlen = 100)
    scores = []
    avg_score_list = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode = True)[brain_name]
        state = env_info.vector_observations
        agents.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            action = agents.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agents.step(state, action, rewards, next_state, dones)
            state = next_state
            score += rewards
            if np.any(dones):
                break 
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        avg_score = np.mean(scores_deque)
        avg_score_list.append(avg_score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.3f}'.\
              format(i_episode, avg_score, np.mean(score)), end="")      
        if (i_episode % 100) == 0 or (avg_score > 0.5):
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            torch.save(agents.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agents.critic_local.state_dict(), 'checkpoint_critic.pth') 
            if avg_score > 0.5:
                print('\nEnvironment solved in {:d} episodes!'.format(i_episode))
                break
        
    return scores, avg_score_list

scores, avg_score_list = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.plot(np.arange(1, len(avg_score_list) + 1), avg_score_list)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

env.close()
