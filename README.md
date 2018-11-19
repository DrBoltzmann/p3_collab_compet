### What's this repo about?

This is the final project of Udacity's deep reinforcement learning nanodegree. The goal here is to create two agents capable of playing table tennis with each other. When the agent hits the ball over the net, it receives a reward of +0.1. If the ball hits the ground or goes out of bounds, the agent receives a reward of -0.01. Thus the goal of each agent is to keep the ball in play.

The state space has 8 dimensions (corresponding to the position and velocity of the ball and racket). Given this information, each agent has to learn how to best select actions. The action space has 2 dimensions; each entry corresponds to movement toward or away from the net, and jumping.

The task is episodic, and in order to solve the environment, the agent must get an average score of +0.5 over 100 consecutive episodes (after taking the maximum over both agents).

### Dependencies

1. [Python 3.6](https://www.python.org/). It has to be 3.6 (3.5 or 3.7 won't work, because of PyTorch and Unity ML-Agents Toolkit).

2. [NumPy](http://www.numpy.org/)

3. [Matplotlib](https://matplotlib.org/)

4. [PyTorch](https://pytorch.org/)

5. [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents), which is an open-source plugin that enables games and simulations to serve as environments for training intelligent agents. Download their Banana app from one of the links below.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - MacOS: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the environment.

### Replication instructions

Install all dependencies, download `main.py`, `ddpg_agent.py`, and `model.py`, put it all in the same folder, add the Tennis app to that folder, and run `python main.py`.
