import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
from algorithms.bmasac import BMASAC
from algorithms.pr2 import PR2
from algorithms.rommeo import ROMMEO


def run(config):
    agent_model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.agent_run_num))
    agent_model_path = agent_model_path / 'model.pt'

    adversary_model_path = (Path('./models') / config.env_id / config.model_name /
                        ('run%i' % config.adversary_run_num))
    adversary_model_path = adversary_model_path / 'model.pt'


    if config.agent_run_num==0:
        agent = BMASAC.init_from_save(agent_model_path)
    elif config.agent_run_num==1:
        agent = ROMMEO.init_from_save(agent_model_path)
    elif config.agent_run_num==2:
        agent = PR2.init_from_save(agent_model_path)

    if config.adversary_run_num==0:
        adversary = BMASAC.init_from_save(adversary_model_path)
    elif config.adversary_run_num==1:
        adversary = ROMMEO.init_from_save(adversary_model_path)
    elif config.adversary_run_num==2:
        adversary = PR2.init_from_save(adversary_model_path)

    env = make_env(config.env_id, discrete_action=agent.discrete_action)
    agent.prep_rollouts(device='cpu')
    adversary.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval
    average_adv = 0
    for ep_i in range(config.n_episodes):

        obs = env.reset()
        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        # env.render('human')
        ep_adv=0
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            agent_torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(agent.nagents)]


            adversary_torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                        requires_grad=False)
                               for i in range(adversary.nagents)]
            # get actions as torch Variables
            agent_torch_actions = agent.step(agent_torch_obs, explore=False)
            adversary_torch_actions = adversary.step(adversary_torch_obs, explore=False)

            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy().flatten() for ac in agent_torch_actions]
            adversary_actions = [ac.data.numpy().flatten() for ac in adversary_torch_actions]
            actions = np.vstack((adversary_actions[:-1],agent_actions[-1]))
            obs, rewards, dones, infos = env.step(actions)
            ep_adv = ep_adv+rewards[-1]-rewards[0]


            if config.save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            # env.render('human')
        average_adv = average_adv + ep_adv
        print(ep_i,ep_adv,average_adv/(ep_i+1))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="simple_tag", help="Name of environment")
    parser.add_argument("--model_name", default="Provide_Models",
                        help="Name of model")
    # TDOM-AC:0 ROMMEO:1 PR2:2
    parser.add_argument("--agent_run_num", default=0, type=int)#9 & 12
    parser.add_argument("--adversary_run_num", default=0, type=int)  # 9 & 12
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=100, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--fps", default=30, type=int)

    config = parser.parse_args()

    run(config)
