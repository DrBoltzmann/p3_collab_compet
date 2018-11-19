### Task

The goal here is to create two agents capable of playing table tennis with each other. When the agent hits the ball over the net, it receives a reward of +0.1. If the ball hits the ground or goes out of bounds, the agent receives a reward of -0.01. Thus the goal of each agent is to keep the ball in play.

The state space has 8 dimensions (corresponding to the position and velocity of the ball and racket). Given this information, each agent has to learn how to best select actions. The action space has 2 dimensions; each entry corresponds to movement toward or away from the net, and jumping.

The task is episodic, and in order to solve the environment, the agent must get an average score of +0.5 over 100 consecutive episodes (after taking the maximum over both agents).

### Result

The best result I managed to obtain was solving the taks in 1614 episodes. The plot below shows how the scores as the episodes elapsed.

![scores X episodes](https://github.com/thiagomarzagao/p3_collab_compet/blob/master/Figure_1.png)

### The DDPG algorithm

To achieve that result I used an adapted version of the [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) (DDPG) [code](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py) provided in lesson 5. The DDPG algorithm is similar to the [DQN](https://www.nature.com/articles/nature14236) algorithm in that there are two neural networks with identical architecture. But DQN can't handle continuous action spaces - which is the case here, as each of the four dimensions of the action space ranges from -1 to +1. The DDPG algorithm handles that by having one neural net - the actor - pick (what it believes to be) the best policy for each state and having the other neural net - the critic - evaluate those policy choices. That way we are able to work with continuous spaces. DDPG also uses replay buffer, just like DQN, but with "soft" updates - we don't outright clone one network into the other; instead we make them closer to each other at each update. (Though as we saw in the previous module it's possible to use soft updates with the DQN algorithm too.)

The structure of both my neural nets (actor and critic) is the same: one input layer of size 8 (the size of the state space), two hidden layers of size 256 each, and one output layer of size 2 (the size of the action space). The activation function is ReLU except for the output layer, where I used tanh. Following a comment I saw on Student Hub I batch-normalized the output of the first hidden layer - that greatly improved the result.

I tried other network architectures (adding and subtracting hidden layers and changing the size of each hidden layer) but they didn't improve the model.

As for the hyperparameters, this is what got me the best results:

BUFFER_SIZE = int(1e6)  # replay buffer size

BATCH_SIZE = 256        # minibatch size

GAMMA = 0.99            # discount factor

TAU = 1e-3              # for soft update of target parameters

LR_ACTOR = 2e-4         # learning rate of the actor 

LR_CRITIC = 2e-4        # learning rate of the critic

WEIGHT_DECAY = 0        # L2 weight decay

### Ideas for future improvements

In the future it might be worth it to try prioritized experience replay, as well as more recent approaches (like [Levine et al 2018](https://journals.sagepub.com/doi/abs/10.1177/0278364917710318)). Also, this took a while to run, so in the future I'd like to try using my laptop's GPU (right now that would be difficult because the latest CUDA realease is not yet compatible with XCode 10).
