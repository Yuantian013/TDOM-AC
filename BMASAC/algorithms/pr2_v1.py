import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork,GaussianPolicy
from utils.utils import tensor_from_tensor_list
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import PR2Agent
from torch.optim import Adam
import gym
from gym import spaces, logger

MSELoss = torch.nn.MSELoss()

class PR2(object):
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=128,alpha=0.5,
                 discrete_action=False,action_space=None,Diff=None):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.Diff = Diff
        self.alpha=alpha
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.obs_dims = [agent['num_in_pol'] for agent in agent_init_params]
        self.act_dim = [agent['num_out_pol'] for agent in agent_init_params]
        self.agents = [PR2Agent(lr=lr, action_space=self.act_dim, hidden_dim=hidden_dim,Diff = self.Diff, num_agents=self.nagents, obs_dims=self.obs_dims,alpha=self.alpha, **params)
                       for params in agent_init_params]

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.opp_mod_dev = 'cpu'  # device for opponent model
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0


    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]



    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(observations, idx, explore=explore) for idx, a in enumerate(self.agents)]


    def belief(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.belief(observations, idx, explore=False) for idx, a in enumerate(self.agents)]

    def update(self, sample, agent_i, parallel=False, logger=None):
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        #Critic Update
        curr_agent.critic_optimizer.zero_grad()
        with torch.no_grad():
            observations = tensor_from_tensor_list(obs)
            # Get the next state
            batch_next_state = tensor_from_tensor_list(next_obs)
            # Get the next action # 13
            next_action_pi,next_action_entr,_,_ = curr_agent.policy.sample(batch_next_state)
            # get the next opponent output
            next_op_feed_in =  torch.cat((batch_next_state, next_action_pi), dim=1)

            Batch_NEXT_Q = []
            for samp in range(16):
                next_op_action_preds = []
                for next_op in range(curr_agent.num_agents - 1):
                    baction, _, _, _ = curr_agent.opponent_model[next_op].sample(next_op_feed_in)
                    if next_op_action_preds == []:
                        next_op_action_preds = baction
                    else:
                        next_op_action_preds = torch.concat([next_op_action_preds, baction], dim=1)
                next_actions = torch.cat((next_op_action_preds, next_action_pi), dim=1)
                next_vf_in_pi = torch.cat((next_actions, batch_next_state), dim=1)
                if Batch_NEXT_Q==[]:
                    next_Q = curr_agent.target_critic(next_vf_in_pi)-curr_agent.alpha*next_action_entr
                    # next_Q = curr_agent.target_critic(next_vf_in_pi)
                    Batch_NEXT_Q = next_Q
                else:
                    next_Q = curr_agent.target_critic(next_vf_in_pi) - curr_agent.alpha * next_action_entr
                    # next_Q = curr_agent.target_critic(next_vf_in_pi)
                    Batch_NEXT_Q = torch.cat((Batch_NEXT_Q, next_Q), dim=0)

            # target Q = R + γV
            target_value = (rews[agent_i].view(-1, 1) + self.gamma * Batch_NEXT_Q.mean() * (
                        1 - dones[agent_i].view(-1, 1)))

            batch_actions_without_current = tensor_from_tensor_list(acs, excl_idx=agent_i)
            batch_current_agent_action = tensor_from_tensor_list([acs[agent_i]])
            batch_actions = torch.concat([batch_actions_without_current,batch_current_agent_action],dim=1)
            vf_in = torch.cat((batch_actions, observations), dim=1)

        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()
        curr_agent.policy_optimizer.step()

        # update opponent model
        observations = tensor_from_tensor_list(obs)
        for o in range(curr_agent.num_agents - 1):
            curr_agent.opponent_model[o].zero_grad()

        with torch.no_grad():
            # action_pi, _,_,_ = curr_agent.policy.sample(observations)
            action_pi = tensor_from_tensor_list([acs[agent_i]])
            op_feed_in = torch.cat((observations, action_pi), dim=1)
            est_Q = []

            for samp in range(16):
                op_action_preds = []
                for op in range(curr_agent.num_agents - 1):
                    baction, _, _, _ = curr_agent.opponent_model[op].sample(op_feed_in)
                    if op_action_preds == []:
                        op_action_preds = baction
                    else:
                        op_action_preds = torch.concat([op_action_preds, baction], dim=1)
                actions = torch.cat((op_action_preds, action_pi), dim=1)
                vf_in_pi = torch.cat((actions, observations), dim=1)
                if est_Q == []:
                    V = curr_agent.critic(vf_in_pi)
                    est_Q = V
                else:
                    V = curr_agent.critic(vf_in_pi)
                    est_Q = torch.cat((est_Q, V), dim=0)
        sum_opp_vf_loss = 0
        act_dim = acs[0].size()[1]
        for num in range(curr_agent.num_agents - 1):
            update_op_action_preds = []
            update_op_entro = []
            for op in range(curr_agent.num_agents - 1):
                baction, ben, _, _ = curr_agent.opponent_model[op].sample(op_feed_in)
                if update_op_action_preds == []:
                    update_op_action_preds = baction
                    update_op_entro = ben
                else:
                    update_op_action_preds = torch.concat([update_op_action_preds, baction], dim=1)
                    update_op_entro = torch.concat([update_op_entro, ben], dim=1)
            true_actions = tensor_from_tensor_list(acs, excl_idx=agent_i)

            true_actions[:, int(act_dim * num):int(act_dim * num + act_dim)] = update_op_action_preds[:,
                                                                               int(act_dim * num):int(
                                                                                   act_dim * num + act_dim)]
            bentropy_preds = update_op_entro[:, int(1 * num):int(1 * num + 1)]
            update_op_actions = torch.cat((true_actions, action_pi), dim=1)
            Update_in = torch.cat((update_op_actions, observations), dim=1)
            Update_V = curr_agent.critic(Update_in)
            opp_vf_loss = (curr_agent.alpha * bentropy_preds - (est_Q.mean()-Update_V)).mean()

            opp_vf_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.opponent_model[num].parameters(), 0.5)
            curr_agent.opponent_model_optimizer[num].step()

            sum_opp_vf_loss = sum_opp_vf_loss + opp_vf_loss.item()

        ############################################################################# update agent policy

        curr_agent.policy_optimizer.zero_grad()
        with torch.no_grad():
            observations= tensor_from_tensor_list(obs)
        action_pi, entropy_pi, _,_ = curr_agent.policy.sample(observations)

        op_feed_in = torch.cat((observations, action_pi), dim=1)
        Batch_Q = []
        for samp in range(16):
            op_action_preds = []
            for op in range(curr_agent.num_agents - 1):
                baction, bentropy, _, _ = curr_agent.opponent_model[op].sample(op_feed_in)
                if op_action_preds == []:
                    op_action_preds = baction
                else:
                    op_action_preds = torch.concat([op_action_preds, baction], dim=1)
            actions = torch.cat((op_action_preds, action_pi), dim=1)
            vf_in_pi = torch.cat((actions, observations), dim=1)
            if Batch_Q == []:
                Q = curr_agent.critic(vf_in_pi)
                Batch_Q = Q
            else:
                Q = curr_agent.critic(vf_in_pi)
                Batch_Q = torch.cat((Batch_Q, Q), dim=0)

        ent = (entropy_pi).mean()
        pol_loss = (curr_agent.alpha * entropy_pi- Batch_Q.mean()).mean()


        pol_loss.backward()

        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.)
        curr_agent.policy_optimizer.step()



        return pol_loss.item(), vf_loss.item(), sum_opp_vf_loss,curr_agent.alpha.item(),ent.item()
        # return pol_loss.item(), vf_loss.item(), 0
    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            for o in range(self.nagents - 1):
                a.opponent_model[o].train()
            a.critic.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
            for o in range(self.nagents - 1):
                a.opponent_model[o].eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="PR2", adversary_alg="PR2",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,alpha=0.5,Diff=False):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "PR2" :
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'alpha':alpha,
                     'Diff':False}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_diff_env(cls,alg_types=None,gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,alpha=0.1,Diff=True):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []

        alg_types = alg_types
        for algtype in zip(alg_types):

            agent_init_params.append({'num_in_pol': 3,
                                      'num_out_pol': 1,
                                      'num_in_critic': 4})

        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': False,
                     'Diff':True,
                     'alpha':alpha}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance