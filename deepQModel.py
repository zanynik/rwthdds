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