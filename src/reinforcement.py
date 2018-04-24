import numpy as np
import torch
from torch.autograd import Variable
# import torch.optim as optim
from torch.distributions import Categorical
try:
    import src.active_learning as al
except:
    import active_learning as al
from tqdm import tqdm

class Environment(object):
    """ environment for the RL learner to interact with and recieve reward from """
    def __init__(self, model, train_x, train_y, val_x, val_y, loss_func, optimizer, usps_data=None, use_cuda='def', params='default'):
        self.use_cuda = torch.cuda.is_available() if use_cuda=='def' else False 
        self.model = model.cuda() if self.use_cuda else model
        self.usps_train_loader = usps_data
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.loss_func = loss_func
        self.optimizer = optimizer
        if params=='default':
            self.set_params(al_itrs=20, npoints=20, batch_size=10, epochs_per_train=5, shuffle=True)
        elif isinstance(params, dict):
            self.set_params(meta_epochs=params['meta_epochs'], npoints=params['npoints'], \
                            batch_size=params['batch_size'], epochs_per_train=params['epochs_per_train'])

    def set_params(self, **kwargs):
        """ Set active learning parameters """
        keys = kwargs.keys()
        if 'batch_size' in keys:
            self.batch_size = kwargs['batch_size']
        if 'epochs_per_train' in keys:
            self.ept = kwargs['epochs_per_train']
        if 'npoints' in keys:
            self.npoints = kwargs['npoints']
        if 'al_itrs' in keys:
            self.al_itrs = kwargs['al_itrs']
        if 'shuffle' in keys:
            self.shuffle = kwargs['shuffle']

    def train_usps(self, epochs, model, opt):
        """ train the learner on the ups handwritten dataset 
        --- note this is the dataset I use as the transfer for mnist 
        """
        losses,n_itr = [],0
        for e in range(epochs):
            for batch_x, batch_y in self.usps_train_loader:
                batch_x = Variable(batch_x.cuda()) if self.use_cuda else Variable(batch_x)
                batch_y = Variable(batch_y.cuda()) if self.use_cuda else Variable(batch_y)

                result = model(batch_x)
                loss = self.loss_func(result, batch_y)
                opt.zero_grad()
                loss.backward()
                opt.step()

                n_itr+=1
                losses.append(loss)

        if self.use_cuda:
            losses = [itm.data.cpu().numpy()[0] for itm in losses]
        else:
            losses = [itm.data.numpy()[0] for itm in losses]

        return list(range(n_itr)), losses

    # The incoming state is the marginals on all the points
    def select_action(self, agent, state):
        """ Get the probabilities for different actions and sample to select one. """
        probs = agent(state)
        m = Categorical(probs)
        action = m.sample()
        agent.saved_log_probs.append(m.log_prob(action))
        return action.data[0]

    def finish_experiment(self, agent, optimizer, gamma=0.99):
        """ run after the training episode is done, we can finally backprop over the expectation of the rewards. 
        -------
        Args: agent; the RL agent (pytorch nn.Module class)
              optimizer; the optimizer for the agent
        -------
        Returns: None, just backprops the expectation of the gradient over time 
        """
        rewards = self.get_rewards(agent, gamma=gamma)

        policy_loss = []
        for log_prob, reward in zip(agent.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        del agent.rewards[:]
        del agent.saved_log_probs[:]

    def get_rewards(self, agent, gamma=0.99):
        """ Get the discounted rewards from the episode of training. Discounts based on the gamma parameter """
        rewards, R = [], 0
        for r in agent.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards).cuda() if self.use_cuda else torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        return rewards

    def reset_env(self, lr, reset_type='active', usps_init_epochs=1):
        """ Function to reset the environment i.e. re make the base model and start the whole process again 
        --- another reinforcement learning iteration must start with this 
        --------
        You need to specify the reset type - active for the reinforcement learning for active learning setup 
        and - transfer for the reinforcement learning setup for learning to either transfer or pick the active 
        learning policy 
        """
        if reset_type=='active':
            model_try = type(self.model)()
            optimizer_try = type(self.optimizer)(model_try.parameters(), lr=lr)
        elif reset_type=='transfer':
            model_try = type(self.model)()
            optimizer_try = type(self.optimizer)(model_try.parameters(), lr=lr)
            _, _ = self.train_usps(usps_init_epochs, model_try, optimizer_try)

        experiment = al.ExperiAL(model_try, self.train_x, self.train_y, self.val_x, self.val_y, self.loss_func, optimizer_try)
        # Note the meta epochs below (in AL) refers to the number of times we iterate a AL policy, only 1 obviously.
        experiment.set_params(meta_epochs=1, npoints=self.npoints, batch_size=self.batch_size, epochs_per_train=self.ept)
        return experiment, model_try, optimizer_try

    def run_experiments(self, agent, optimizer_agent, policy_key, n_experiments, tar_cost=0.3,gamma=0.9,rtype='active', lr=0.01):
        """ Function to run the experiments over one RL agent.  The idea is you pass the RL agent and it gets exposed to 'n_experiments' 
        instances of the environment and each of those is like one 'game' in the usual RL setup. The policy key is the key of different 
        policies you want the RL agent to learn to use.  The agent and optimizer agent are just the RL agent and optimizer.
        
        ---------
        tar_cost governs the difference in reward between the active learning policy and the transfer. the active learning reward is 
        just subtracted by the value of tar_cost. - Note there are defintely better ways to do this.
        --------
        rtype just denotes the type of RL experiemnt you are running.
        """
        running_reward = 1.0
        policies_chosen, rewards_total,accs_total = [],[],[]
        for i_exp in tqdm(range(n_experiments)):
            ### Reset the envorinment
            experiment, model_try, optimizer_try = self.reset_env(lr=lr, reset_type=rtype, usps_init_epochs=1)

            # Calculate the initial state
            train_tensor = self.train_x.cuda() if self.use_cuda else self.train_x
            state = model_try(Variable(train_tensor))

            # Run the experiment up to training on 1000 points
            policies,track_reward,accs,rwd = [],[],[],0
            for t in range(self.al_itrs):
                action = self.select_action(agent, state)
                if policy_key[action]=='transfer':
                    _,_ = self.train_usps(self.ept, model_try, optimizer_try)
                    _,_ = experiment._train(experiment.lab_x, experiment.lab_y)
                    acc = al.accuracy(model_try, self.val_x, self.val_y, self.use_cuda)
                    accs.append(acc)
                    reward = acc
                    # Below is the option where the reward is based on the increase in accuracy, not total accuracy 
                    #reward, rwd = acc-rwd, acc
                elif policy_key[action]!='transfer':
                    _, _ = experiment.active_learn(policy=policy_key[action])
                    _,_ = experiment._train(experiment.lab_x, experiment.lab_y)
                    acc = al.accuracy(model_try, self.val_x, self.val_y, self.use_cuda)
                    accs.append(acc)
                    reward = acc-tar_cost
                    # Below is the option where the reward is based on the increase in accuracy, not total accuracy 
                    #reward,rwd = acc-rwd, acc
                policies.append(policy_key[action])
                track_reward.append(reward)
                state = model_try(Variable(train_tensor))
                agent.rewards.append(reward)

            policies_chosen.append(policies)
            rewards_total.append(np.array(track_reward))
            accs_total.append(np.array(accs))
            running_reward = running_reward * 0.99 + t * 0.01
            self.finish_experiment(agent, optimizer_agent, gamma=gamma)

        return policies_chosen, np.array(rewards_total), np.array(accs_total)
