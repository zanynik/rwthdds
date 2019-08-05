import os
checkpoint_base_dir = 'checkpoints_BO/'

# Combination of base-dir and environment-name.
checkpoint_dir = None

# Full path for the log-file for rewards.
log_reward_path = None

# Full path for the log-file for Q-values.
log_q_values_path = None


def update_paths(env_name):
    """
    Update the path-names for the checkpoint-dir and log-files.
    
    Call this after you have changed checkpoint_base_dir and
    before you create the Neural Network.
    
    :param env_name:
        Name of the game-environment you will use in OpenAI Gym.
    """

    global checkpoint_dir
    global log_reward_path
    global log_q_values_path

    # Add the environment-name to the checkpoint-dir.
    checkpoint_dir = os.path.join(checkpoint_base_dir, env_name)

    # Create the checkpoint-dir if it does not already exist.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # File-path for the log-file for episode rewards.
    log_reward_path = os.path.join(checkpoint_dir, "log_reward.txt")

    # File-path for the log-file for Q-values.
    log_q_values_path = os.path.join(checkpoint_dir, "log_q_values.txt")


########################################################################
# Classes used for logging data during training.


class Log:
    """
    Base-class for logging data to a text-file during training.

    It is possible to use TensorFlow / TensorBoard for this,
    but it is quite awkward to implement, as it was intended
    for logging variables and other aspects of the TensorFlow graph.
    We want to log the reward and Q-values which are not in that graph.
    """

    def __init__(self, file_path):
        """Set the path for the log-file. Nothing is saved or loaded yet."""

        # Path for the log-file.
        self.file_path = file_path

        # Data to be read from the log-file by the _read() function.
        self.count_episodes = None
        self.count_states = None
        self.data = None

    def _write(self, count_episodes, msg):
        """
        Write a line to the log-file. This is only called by sub-classes.
        
        :param count_episodes:
            Counter for the number of episodes processed during training.

        :param count_states: 
            Counter for the number of states processed during training.

        :param msg:
            Message to write in the log.
        """

        with open(file=self.file_path, mode='a', buffering=1) as file:
            msg_annotated = "{0}\t{1}\n".format(count_episodes, msg)
            file.write(msg_annotated)

    def _read(self):
        """
        Read the log-file into memory so it can be plotted.

        It sets self.count_episodes, self.count_states and self.data
        """

        # Open and read the log-file.
        with open(self.file_path) as f:
            reader = csv.reader(f, delimiter="\t")
            self.count_episodes, self.count_states, *data = zip(*reader)

        # Convert the remaining log-data to a NumPy float-array.
        self.data = np.array(data, dtype='float')


class LogReward(Log):
    """Log the rewards obtained for episodes during training."""

    def __init__(self):
        # These will be set in read() below.
        self.episode = None
        self.mean = None

        # Super-class init.
        Log.__init__(self, file_path=log_reward_path)

    def write(self, count_episodes, reward_episode, reward_mean):
        """
        Write the episode and mean reward to file.
        
        :param count_episodes:
            Counter for the number of episodes processed during training.

        :param count_states: 
            Counter for the number of states processed during training.

        :param reward_episode:
            Reward for one episode.

        :param reward_mean:
            Mean reward for the last e.g. 30 episodes.
        """

        msg = "{0:.1f}\t{1:.1f}".format(reward_episode, reward_mean)
        self._write(count_episodes=count_episodes, msg=msg)

    def read(self):
        """
        Read the log-file into memory so it can be plotted.

        It sets self.count_episodes, self.count_states, self.episode and self.mean
        """

        # Read the log-file using the super-class.
        self._read()

        # Get the episode reward.
        self.episode = self.data[0]

        # Get the mean reward.
        self.mean = self.data[1]


class LogQValues(Log):
    """Log the Q-Values during training."""

    def __init__(self):
        # These will be set in read() below.
        self.min = None
        self.mean = None
        self.max = None
        self.std = None

        # Super-class init.
        Log.__init__(self, file_path=log_q_values_path)

    def write(self, q_values):
        """
        Write basic statistics for the Q-values to file.

        :param count_episodes:
            Counter for the number of episodes processed during training.

        :param count_states: 
            Counter for the number of states processed during training.

        :param q_values:
            Numpy array with Q-values from the replay-memory.
        """

        msg = "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(np.min(q_values),
                                                          np.mean(q_values),
                                                          np.max(q_values),
                                                          np.std(q_values))

        self._write(count_episodes=count_episodes,
                    msg=msg)

    def read(self):
        """
        Read the log-file into memory so it can be plotted.

        It sets self.count_episodes, self.count_states, self.min / mean / max / std.
        """

        # Read the log-file using the super-class.
        self._read()

        # Get the logged statistics for the Q-values.
        self.min = self.data[0]
        self.mean = self.data[1]
        self.max = self.data[2]
        self.std = self.data[3]

########################################################################
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA):
        super(DeepQNetwork, self).__init__()
        #self.conv1 = nn.Conv2d(3, 32, 8, stride=4, padding=1)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        #self.fc1 = nn.Linear(128*23*16, 512)
        self.fc1 = nn.Linear(128*49, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 6)
        #self.optimizer = optim.SGD(self.parameters(), lr=self.ALPHA, momentum=0.9)
        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        observation = T.Tensor(observation).to(self.device)
        #observation = observation.view(-1, 3, 210, 160).to(self.device)
        #observation = observation.view(-1, 4, 84, 84)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        #observation = observation.view(-1, 128*23*16).to(self.device)
        observation = observation.view(-1, 128*49)
        observation = F.relu(self.fc1(observation))
        observation = F.relu(self.fc2(observation))
        observation = F.relu(self.fc3(observation))
        observation = F.relu(self.fc4(observation))
        actions = self.fc5(observation)
        return actions

class Agent(object):
    def __init__(self, gamma, epsilon, alpha,
                 maxMemorySize, epsEnd=0.05,
                 replace=10000, actionSpace=[0,1,2,3,4,5]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.ALPHA = alpha
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt = replace
        self.Q_eval = DeepQNetwork(alpha)
        #self.Q_next = DeepQNetwork(alpha)
        self.log_reward = LogReward()

    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr%self.memSize] = [state, action, reward, state_]
        self.memCntr += 1

    def chooseAction(self, observation): 
        actions = self.Q_eval.forward([observation])
        action = T.argmax(actions[0]).item()
        #print (action)
        return action


    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        #if self.replace_target_cnt is not None and \
        #   self.learn_step_counter % self.replace_target_cnt == 0:
        #    self.Q_next.load_state_dict(self.Q_eval.state_dict())

#         if self.memCntr+batch_size < self.memSize:
#             memStart = int(np.random.choice(range(self.memCntr)))
#         else:
#             memStart = int(np.random.choice(range(self.memSize-batch_size-1)))
#         miniBatch=self.memory[memStart:memStart+batch_size]
#         memory = np.array(miniBatch)

        miniBatch = random.sample(self.memory, batch_size)
        memory = np.array(miniBatch)
        # convert to list because memory is an array of numpy objects
        Qpred = self.Q_eval.forward(list(memory[:,0][:])).to(self.Q_eval.device)
        Qnext = self.Q_eval.forward(list(memory[:,3][:])).to(self.Q_eval.device)
        #print('Qpred : ', Qpred)
        #print('Qnext : ', Qnext)
        
        #We find max of Next State Q value for each item in memory list
        Qnextmax, ind = T.max(Qnext, dim=1)
        rewards = T.Tensor(list(memory[:,2])).to(self.Q_eval.device)
        actions = T.Tensor(list(memory[:,1])).to(self.Q_eval.device).numpy().astype(int)
        #We use a new variable to copy Pred values
        Qtarget = Qpred
        #print('Val :{} Ind:{}'.format(Qnextmax,ind))
        #print('Rewards :{} Actions:{}'.format(rewards,actions))
        
        #We have to replace Qtarg value as per the action with Reward + gamma max Qnext
        
        
        
        indices = np.arange(batch_size)
    
        Qtarget[indices,actions] = rewards + self.GAMMA*(Qnextmax)
        
        #print('max Qnext', T.max(Qnext, dim=1))
        #print('Qtarget indices,actions', Qtarget[indices,actions])
        #print('Qtarget', Qtarget)
        
        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        #Qpred.requires_grad_()
        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

#################################################################################################################


import gym
import atari_wrappers
from atari_wrappers import wrap_deepmind
#from deepQModel import DeepQNetwork, Agent
from utils import plotLearning
import numpy as np 
from gym import wrappers
from collections import deque
import argparse

if __name__ == '__main__':
    # Description of this program.
    desc = "Reinformenct Learning (Q-learning) for Atari Games using PyTorch"

    # Create the argument parser.
    parser = argparse.ArgumentParser(description=desc)

    # Add arguments to the parser.
    parser.add_argument("--env", required=False, default=None,
                        help="name of the game-environment in OpenAI Gym")

    parser.add_argument("--episodes", required=False, type=int, default=1000,
                        help="number of episodes to run")

	
    # Parse the command-line arguments.
    args = parser.parse_args()

    # Get the arguments.
    env_name = args.env
    numGames = args.episodes

    update_paths(env_name=env_name)

    env = gym.make(env_name)
    env = wrap_deepmind(env,frame_stack=True, pytorch_img=True)
    brain = Agent(gamma=0.95, epsilon=1.0, 
                  alpha=0.003, maxMemorySize=5000,
                  replace=None)
    # The number of possible actions that the agent may take in every step.

    while brain.memCntr < brain.memSize:
        observation = env.reset()
        done = False
        while not done:
            # 0 no action, 1 fire, 2 move right, 3 move left, 4 move right fire, 5 move left fire
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            if done and info['ale.lives'] == 0:
                reward = -100                  
            brain.storeTransition(observation, action, reward, 
                                observation_)
            observation = observation_
    print('done initializing memory')

    scores = []
    epsHistory = []
    batch_size=32
    # uncomment the line below to record every episode. 
    # env = wrappers.Monitor(env, "tmp/space-invaders-1", video_callable=lambda episode_id: True, force=True)
    for i in range(numGames):
        print('starting game ', i+1, 'epsilon: %.4f' % brain.EPSILON)
        count_episodes = i+1
        epsHistory.append(brain.EPSILON)
        brain.learn(batch_size)
        done = False
        observation = env.reset()
        #fx = deque(3*[np.sum(observation[15:200,30:125], axis=2)],3)
        #frames = [np.sum(observation[15:200,30:125], axis=2)]
        score = 0
        newgame = 0
        #lastAction = 0   
        while not done :
            rand = np.random.random()
            if (rand < 1 - brain.EPSILON) and (newgame > 2) :
                action = brain.chooseAction(observation)
            else:
                action = env.action_space.sample()
                newgame+=1
            brain.steps += 1


            observation_, reward, done, info = env.step(action)
            #observation_, reward, done, info = env.step(env.action_space.sample())
            score += reward
            #fx.append(np.sum(observation_[15:200,30:125], axis=2))
            if done and info['ale.lives'] != 0:
                reward = -100
                done = False
            if done and info['ale.lives'] == 0:
                reward = -100
            brain.storeTransition(observation, action, reward, 
                                  observation_)
            observation = observation_
        scores.append(score)
        reward_mean = np.mean(scores[-30:])
        brain.log_reward.write(count_episodes=count_episodes,
                                          reward_episode=score,
                                          reward_mean=reward_mean)
        print("score: {} Mean score :{:.1f}".format(score, reward_mean))
            #env.render(

    x = [i+1 for i in range(numGames)]
    fileName = str(env_name)+'-'+ str(numGames) + 'Games' + 'Gamma' + str(brain.GAMMA) + \
               'Alpha' + str(brain.ALPHA) + 'Memory' + str(brain.memSize)+ '.png'    
    plotLearning(x, scores, epsHistory, fileName)
	
	#Recording Final Game
    done = False
    env = wrappers.Monitor(env, str(env_name)+'-'+ str(numGames) + 'Games Movie', force = True)
    observation = env.reset()
    newgame = 0   
    while not done :
        #After 2 random moves we have fx filled with 3 stack of states
        if (newgame > 2):
            action = brain.chooseAction(observation)
        else:
            action = env.action_space.sample()
            newgame+=1
        observation_, reward, done, info = env.step(action)
        observation = observation_
        if done and info['ale.lives'] != 0:
            done = False
        if done and info['ale.lives'] == 0:
            env.close()
            break