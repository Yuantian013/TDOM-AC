import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork,GaussianPolicy
from utils.utils import tensor_from_tensor_list
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import BMASACAgent
from torch.optim import Adam
import gym
from gym import spaces, logger

MSELoss = torch.nn.MSELoss()

class BMASAC(object):
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.999, tau=0.01, lr=0.01, hidden_dim=128,alpha=1,
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

        self.Diff=Diff
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.obs_dims = [agent['num_in_pol'] for agent in agent_init_params]
        self.act_dim = [agent['num_out_pol'] for agent in agent_init_params]
        self.agents = [BMASACAgent(lr=lr, action_space=self.act_dim, hidden_dim=hidden_dim, num_agents=self.nagents,
                                   obs_dims=self.obs_dims, Diff=self.Diff, **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = 0.95
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

        curr_agent.critic_optimizer.zero_grad()
        with torch.no_grad():
            observations = tensor_from_tensor_list(obs)
            batch_next_state = tensor_from_tensor_list(next_obs)
            next_action_preds = []
            next_bent = 0
            count = 0
            # get the next opponent actions & entropies
            for next_op in range(curr_agent.num_agents - 1):
                baction, bentropy, _,_ = curr_agent.opponent_model[next_op].sample(batch_next_state)
                count+=1
                if next_action_preds == []:
                    next_action_preds = baction
                    next_bent = bentropy
                else:
                    next_action_preds = torch.concat([next_action_preds, baction], dim=1)
                    next_bent = next_bent+bentropy

            # merge the policy input: next_obs+next_op
            next_belief_input = torch.cat((batch_next_state, next_action_preds), dim=1)
            # get the next policy output
            next_action_pi, next_entropy_pi, _,_ = curr_agent.policy.sample(next_belief_input)
            # merge the Q input : next_op+next_a+next_obs
            next_actions = torch.cat((next_action_preds, next_action_pi), dim=1)
            next_vf_in_pi = torch.cat((next_actions, batch_next_state), dim=1)
            # next V = Q - a*Ent
            next_Q = curr_agent.target_critic(next_vf_in_pi) - self.alpha * (next_entropy_pi+next_bent)
            # next_Q = curr_agent.target_critic(next_vf_in_pi) - self.alpha * (next_entropy_pi)

            # target Q = R + γV

            target_value = (rews[agent_i].view(-1, 1) + self.gamma * next_Q * (
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
            curr_agent.opponent_model_optimizer[o].zero_grad()

        # opponent q function input
        # opponent action predictions
        o = 0
        sum_opp_vf_loss=0
        for a, agent in enumerate(self.agents):
            if agent is curr_agent:
                continue

            curr_action_preds = []
            curr_ben_preds = []

            for opu in range(curr_agent.num_agents - 1):
                baction, bentropy, _, _ = curr_agent.opponent_model[opu].sample(observations)
                if curr_action_preds == []:
                    curr_action_preds = baction
                    curr_ben_preds = bentropy
                else:
                    curr_action_preds = torch.concat([curr_action_preds, baction], dim=1)
                    curr_ben_preds = torch.concat([curr_ben_preds, bentropy], dim=1)

            true_actions_without_current_opp = tensor_from_tensor_list(acs, excl_idx=a)

            act_dim = acs[0].size()[1]

            current_op_prediction = curr_action_preds[:,int(act_dim*o):int(act_dim*o+act_dim)]
            bentropy_preds = curr_ben_preds[:,int(1*o):int(1*o+1)]
            temp_true = true_actions_without_current_opp
            vf_in_op = torch.concat([torch.cat([temp_true, current_op_prediction], dim=1), observations], dim=1)
            Q_op=agent.critic(vf_in_op)

            opp_vf_loss = (agent.alpha.detach() * bentropy_preds - Q_op).mean()

            opp_vf_loss.backward()
            torch.nn.utils.clip_grad_norm_(curr_agent.opponent_model[o].parameters(), 0.5)
            curr_agent.opponent_model_optimizer[o].step()
            o += 1
            sum_opp_vf_loss = sum_opp_vf_loss + opp_vf_loss.item()

        ############################################################################# update agent policy
        # Empirical Improvement  + Belief based Improvement
        ########################################################Empirical Improvement:
        curr_agent.policy_optimizer.zero_grad()
        # with torch.no_grad():
        #     observations_1 = tensor_from_tensor_list(obs)
        #     temp = tensor_from_tensor_list(acs, excl_idx=agent_i)
        #     belief_input_1 = torch.cat((observations_1, temp), dim=1)
        #
        # action_pi_1, entropy_pi_1, _,_ = curr_agent.policy.sample(belief_input_1)
        # actions_1 = torch.cat((temp, action_pi_1), dim=1)
        # vf_in_pi_1 = torch.cat((actions_1, observations_1), dim=1)
        # Q1 = curr_agent.critic(vf_in_pi_1)
        #######################################################Belief based Improvement:
        # curr_agent.policy_optimizer.zero_grad()
        with torch.no_grad():
            observations = tensor_from_tensor_list(obs)
            curr_action_preds = []
            for opu in range(curr_agent.num_agents - 1):
                 baction,_,_,_ = curr_agent.opponent_model[opu].sample(observations)
                 if curr_action_preds == []:
                     curr_action_preds=baction
                 else:
                     curr_action_preds=torch.concat([curr_action_preds,baction],dim=1)

            belief_input = torch.cat((observations, curr_action_preds), dim=1)

        action_pi, entropy_pi, _,_ = curr_agent.policy.sample(belief_input)
        actions = torch.cat((curr_action_preds, action_pi), dim=1)
        vf_in_pi = torch.cat((actions, observations), dim=1)
        Q2 = curr_agent.critic(vf_in_pi)

        # Q = 1/2*(Q1+Q2)
        #
        ent = (entropy_pi).mean()

        # pol_loss = (curr_agent.alpha * 1/2*(entropy_pi+entropy_pi_1)-Q).mean()
        pol_loss = (curr_agent.alpha * entropy_pi - Q2).mean()


        pol_loss.backward()

        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.)
        curr_agent.policy_optimizer.step()

        # curr_agent.alpha_optim.zero_grad()
        #
        # alpha_loss = -(curr_agent.log_alpha * (ent + curr_agent.target_entropy).detach()).mean()
        # curr_agent.alpha_optim.zero_grad()
        # alpha_loss.backward()
        #
        # curr_agent.alpha_optim.step()
        # curr_agent.alpha = curr_agent.log_alpha.exp()



        return pol_loss.item(), vf_loss.item(), sum_opp_vf_loss,curr_agent.alpha.item(),ent.item()

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
    def init_from_env(cls, env, agent_alg="BMASAC", adversary_alg="BMASAC",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,Diff=None,alpha=0.5):
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
            if algtype == "TDOM-AC":
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
    def init_diff_env(cls,alg_types=None,gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,alpha=0.1,otherEnv='Diff'):
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