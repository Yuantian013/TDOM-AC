import torch
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam,SGD
from einops import rearrange
from .networks import MLPNetwork,GaussianPolicy,GaussianPrior,ROMMEOPolicy
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
from utils.utils import tensor_from_tensor_list

class MADDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])




class BMASACAgent():
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, num_agents, obs_dims, hidden_dim=128,
                 lr=0.01, alpha= 0.5,action_space=None,Diff=False):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.Diff = Diff
        if self.Diff==True:
            opp_dim = 2
        else:
            opp_dim = sum(obs_dims)
        self.num_agents = num_agents
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.opponent_model = [GaussianPolicy(opp_dim, num_out_pol,
                                         hidden_dim=hidden_dim,
                                         constrain_out=True,
                                         action_space=action_space) for _ in range(self.num_agents - 1)]

        self.policy = GaussianPolicy(opp_dim+num_out_pol*(self.num_agents-1), num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 action_space=action_space)

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        self.target_entropy = torch.Tensor([1]).item()
        self.alpha = torch.ones(1)
        # self.alpha = torch.zeros(10)
        # For now, not modeled
        """self.opponent_model_critic = MLPNetwork(num_in_critic, num_agents,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)"""
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr,weight_decay=0.00)
        self.opponent_model_optimizer = [Adam(self.opponent_model[o].parameters(), lr=lr,weight_decay=0.00) for o in range(self.num_agents - 1)]
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr,weight_decay=0.00)


        # self.mix_optimizer = Adam([self.critic.parameters(),self.policy.parameters()], lr=lr)

    def step(self, all_obs, idx, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            all_obs (PyTorch Variable): Observations for all agents
            idx: index of agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        all_obs = tensor_from_tensor_list(all_obs)

        # prediction for opponents
        with torch.no_grad():
            if self.Diff == True:
                bsample,_,bmean,_ =self.opponent_model[0].sample(all_obs)
                if explore:
                    curr_action_preds = bmean
                else:
                    curr_action_preds = bmean
            else:
                curr_action_preds = []
                for opu in range(self.num_agents - 1):
                    baction, bentropy, bmean, _ = self.opponent_model[opu].sample(all_obs)
                    if explore:
                        if curr_action_preds == []:
                            curr_action_preds = bmean
                        else:
                            curr_action_preds = torch.concat([curr_action_preds, bmean], dim=1)
                    else:
                        if curr_action_preds == []:
                            curr_action_preds = bmean
                        else:
                            curr_action_preds = torch.concat([curr_action_preds, bmean], dim=1)
            action_preds=curr_action_preds
        # use concatenation of observations and predicted opponent actions as input for action
        input = torch.concat((all_obs, action_preds), dim=1)
        saction, log_prob, mean,_= self.policy.sample(input)
        if explore:
            action=saction
        else:
            action= mean
        return action
    def belief(self, all_obs, idx, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            all_obs (PyTorch Variable): Observations for all agents
            idx: index of agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        all_obs = tensor_from_tensor_list(all_obs)

        # prediction for opponents
        with torch.no_grad():
            if self.Diff == True:
                for opu in range(self.num_agents - 1):
                    baction, bentropy, bmean, _ = self.opponent_model[opu].sample(all_obs)
                curr_action_preds=bmean
            else:
                curr_action_preds = []
                for opu in range(self.num_agents - 1):
                    baction, bentropy, bmean, _ = self.opponent_model[opu].sample(all_obs)
                    if curr_action_preds == []:
                        curr_action_preds = bmean
                    else:
                        curr_action_preds = torch.concat([curr_action_preds, bmean], dim=1)
            action_preds=curr_action_preds
        return action_preds
    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'opponent_model': [self.opponent_model[o].state_dict() for o in range(self.num_agents - 1)],
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'opponent_model_optimizer': [self.opponent_model_optimizer[o].state_dict() for o in range(self.num_agents - 1)]}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        for o in range(self.num_agents - 1):
            self.opponent_model[o].load_state_dict(params['opponent_model'][o])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        for o in range(self.num_agents - 1):
            self.opponent_model_optimizer[o].load_state_dict(params['opponent_model_optimizer'][o])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])


class ROMMEOAgent():
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, num_agents, obs_dims, hidden_dim=128,
                 lr=0.01, alpha= 0.5,action_space=None,Diff=None):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.Diff=Diff
        if self.Diff==True:
            opp_dim = 2
        else:
            opp_dim = sum(obs_dims)
        self.num_agents = num_agents
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.opponent_model = [ROMMEOPolicy(opp_dim, num_out_pol,
                                         hidden_dim=hidden_dim,
                                         constrain_out=True,
                                         action_space=action_space) for _ in range(self.num_agents - 1)]

        self.prior = GaussianPrior(opp_dim, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 action_space=action_space)

        self.policy = ROMMEOPolicy(opp_dim+num_out_pol*(self.num_agents-1), num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 action_space=action_space)


        self.alpha = torch.ones(1)
        # self.alpha = torch.zeros(10)
        # For now, not modeled
        """self.opponent_model_critic = MLPNetwork(num_in_critic, num_agents,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)"""
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr,weight_decay=0.00)
        self.opponent_model_optimizer = [Adam(self.opponent_model[o].parameters(), lr=lr,weight_decay=0.00) for o in range(self.num_agents - 1)]
        self.prior_optimizer = Adam(self.prior.parameters(), lr=lr, weight_decay=0.00)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr,weight_decay=0.00)


    def step(self, all_obs, idx, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            all_obs (PyTorch Variable): Observations for all agents
            idx: index of agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        all_obs = tensor_from_tensor_list(all_obs)

        # prediction for opponents
        with torch.no_grad():
            if self.Diff == True:
                _, _, bmean, _,_,_ =self.opponent_model[0].sample(all_obs)
                curr_action_preds=bmean
                # action_preds = torch.tensor([[10.]])
            else:
                curr_action_preds = []
                for opu in range(self.num_agents - 1):
                    baction, bentropy, bmean, _,_,_ = self.opponent_model[opu].sample(all_obs)
                    if explore:
                        if curr_action_preds == []:
                            curr_action_preds = baction
                        else:
                            curr_action_preds = torch.concat([curr_action_preds, baction], dim=1)
                    else:
                        if curr_action_preds == []:
                            curr_action_preds = bmean
                        else:
                            curr_action_preds = torch.concat([curr_action_preds, bmean], dim=1)
            action_preds=curr_action_preds
        # use concatenation of observations and predicted opponent actions as input for action
        input = torch.concat((all_obs, action_preds), dim=1)
        saction, log_prob, mean,_,_,_= self.policy.sample(input)
        if explore:
            action=saction
        else:
            action= mean
        return action
    def belief(self, all_obs, idx, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            all_obs (PyTorch Variable): Observations for all agents
            idx: index of agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        all_obs = tensor_from_tensor_list(all_obs)

        # prediction for opponents
        with torch.no_grad():
            if self.Diff == True:
                for opu in range(self.num_agents - 1):
                    baction, bentropy, bmean, _, _, _= self.opponent_model[opu].sample(all_obs)
                curr_action_preds=bmean
            else:
                curr_action_preds = []
                for opu in range(self.num_agents - 1):
                    baction, bentropy, bmean, _ ,_,_= self.opponent_model[opu].sample(all_obs)
                    if curr_action_preds == []:
                        curr_action_preds = bmean
                    else:
                        curr_action_preds = torch.concat([curr_action_preds, bmean], dim=1)
            action_preds=curr_action_preds
        return action_preds
    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'opponent_model': [self.opponent_model[o].state_dict() for o in range(self.num_agents - 1)],
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'opponent_model_optimizer': [self.opponent_model_optimizer[o].state_dict() for o in range(self.num_agents - 1)]}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        for o in range(self.num_agents - 1):
            self.opponent_model[o].load_state_dict(params['opponent_model'][o])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        for o in range(self.num_agents - 1):
            self.opponent_model_optimizer[o].load_state_dict(params['opponent_model_optimizer'][o])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])


class PR2Agent():
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, num_agents, obs_dims, hidden_dim=128,
                 lr=0.01, alpha= 0.5,action_space=None,Diff=None):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.Diff=Diff
        if self.Diff==True:
            opp_dim = 2
        else:
            opp_dim = sum(obs_dims)
        self.num_agents = num_agents
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.opponent_model = [GaussianPolicy(opp_dim+num_out_pol, num_out_pol,
                                         hidden_dim=hidden_dim,
                                         constrain_out=True,
                                         action_space=action_space) for _ in range(self.num_agents - 1)]

        self.policy = GaussianPolicy(opp_dim, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 action_space=action_space)

        self.alpha = torch.ones(1)
        # self.alpha = torch.zeros(10)
        # For now, not modeled
        """self.opponent_model_critic = MLPNetwork(num_in_critic, num_agents,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)"""
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr,weight_decay=0.00)
        self.opponent_model_optimizer = [Adam(self.opponent_model[o].parameters(), lr=lr,weight_decay=0.00) for o in range(self.num_agents - 1)]
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr,weight_decay=0.00)
        # self.mix_optimizer = Adam([self.critic.parameters(),self.policy.parameters()], lr=lr)

    def step(self, all_obs, idx, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            all_obs (PyTorch Variable): Observations for all agents
            idx: index of agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        all_obs = tensor_from_tensor_list(all_obs)


        # use concatenation of observations and predicted opponent actions as input for action
        saction, log_prob, mean,_= self.policy.sample(all_obs)
        if explore:
            action=saction
        else:
            action= mean
        return action
    def belief(self, all_obs, idx, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            all_obs (PyTorch Variable): Observations for all agents
            idx: index of agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        all_obs = tensor_from_tensor_list(all_obs)
        _, _, mean, _ = self.policy.sample(all_obs)

        op_feed_in = torch.cat((all_obs, mean), dim=1)
        # prediction for opponents
        op_action_preds=[]
        for op in range(self.num_agents - 1):
            _, _, bmean, _ = self.opponent_model[op].sample(op_feed_in)
            if op_action_preds == []:
                op_action_preds = bmean
            else:
                op_action_preds = torch.concat([op_action_preds, bmean], dim=1)

        return op_action_preds
    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'opponent_model': [self.opponent_model[o].state_dict() for o in range(self.num_agents - 1)],
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'opponent_model_optimizer': [self.opponent_model_optimizer[o].state_dict() for o in range(self.num_agents - 1)]}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        for o in range(self.num_agents - 1):
            self.opponent_model[o].load_state_dict(params['opponent_model'][o])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        for o in range(self.num_agents - 1):
            self.opponent_model_optimizer[o].load_state_dict(params['opponent_model_optimizer'][o])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])