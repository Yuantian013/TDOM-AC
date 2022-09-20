import argparse
import torch
import time
import os
from utils.utils import tensor_from_tensor_list
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils import logger
from gym import spaces, logger
from algorithms.maddpg import MADDPG
from algorithms.tdomac import BMASAC
from algorithms.rommeo import ROMMEO
from algorithms.pr2_comp import PR2
# from algorithms.pr2_v1 import PR2
MSELoss = torch.nn.MSELoss()
USE_CUDA = False  # torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
    # get_env_fn(0)

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))


    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)


    from utils import logger
    algs = config.agent_alg
    sd = int(config.seed)
    if algs == 'TDOM-AC':
        agent = BMASAC.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim,Diff=None,alpha=config.alpha)
        log_path = './Log/Tag/TDOM-AC/' + str(sd)
        output = np.zeros([4, 5])
    elif algs == 'ROMMEO':
        agent = ROMMEO.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim,Diff=None,alpha=config.alpha)
        log_path = './Log/Tag/ROMMEO/' + str(sd)
        output = np.zeros([4, 6])
    elif algs == 'PR2':
        agent = PR2.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim,Diff=None,alpha=config.alpha)
        log_path = './Log/Tag/PR2/' + str(sd)
        output = np.zeros([4, 5])



    replay_buffer = ReplayBuffer(config.buffer_length, agent.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0



    logger.configure(dir=log_path, format_strs=['csv'])
    total_num_step = 0
    MA_step=0
    MA_AR = 0
    MA_ADV = 0
    MA_ADR = 0
    MA_Lost = 10
    MA_Out = 10
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):

        obs = env.reset()

        # (16) (16) (16) (14)
        agent.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):

            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(agent.nagents)]
            # get actions as torch Variables
            torch_agent_actions = agent.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [np.array(ac.data.numpy()) for ac in torch_agent_actions]

            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]

            next_obs, rewards, dones, infos = env.step(actions)

            agent_actions = [np.array(ac.data.numpy()) for ac in torch_agent_actions]

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

            total_num_step+=1
            obs = next_obs
            t += config.n_rollout_threads

            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    agent.prep_training(device='gpu')
                else:
                    agent.prep_training(device='cpu')


                for u_i in range(1):
                    for a_i in range(agent.nagents):
                        sample = replay_buffer.sample((config.batch_size),to_gpu=USE_CUDA)
                        if algs == 'TDOM-AC':
                            # policy loss,critic loss, belief loss, alpha, entropy
                            output[a_i][0], output[a_i][1], output[a_i][2], output[a_i][3], output[a_i][
                                4] = agent.update(sample, a_i,
                                                  logger=logger)
                        elif algs == 'ROMMEO':
                            # policy loss,critic loss, belief loss, alpha, entropy, prior loss
                            output[a_i][0], output[a_i][1], output[a_i][2], output[a_i][3], output[a_i][
                                4],output[a_i][5] = agent.update(sample, a_i,
                                                  logger=logger)

                        elif algs == 'PR2':
                            # policy loss,critic loss, belief loss, alpha, entropy
                            output[a_i][0], output[a_i][1], output[a_i][2], output[a_i][3],output[a_i][4] = agent.update(sample, a_i,
                                                                                                      logger=logger)
                    agent.update_all_targets()
                agent.prep_rollouts(device='cpu')

        if (len(replay_buffer) >= config.batch_size) and ep_i % 10 ==0:
            adv_r = 0
            agent_r = 0
            adversaries_r = 0
            pred_loss = np.zeros([4])
            total_survive_time = 0
            out=0
            lost=0
            for eval_time in range(10):
                obs = env.reset()
                test_return = np.zeros([4])

                for step in range(25):
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                          requires_grad=False)
                                 for i in range(agent.nagents)]
                    # get actions as torch Variables
                    torch_agent_actions = agent.step(torch_obs, explore=False)
                    # convert actions to numpy arrays
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

                    torch_agent_prediction = agent.belief(torch_obs, explore=False)
                    for agent_num in range(4):
                        with torch.no_grad():
                            real_actions = tensor_from_tensor_list(torch_agent_actions, excl_idx=agent_num)
                            pred_actions = torch_agent_prediction[agent_num]
                            pred_loss[agent_num] = pred_loss[agent_num]+MSELoss(real_actions,pred_actions).item()
                    actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                    next_obs, rewards, dones, infos = env.step(actions)
                    obs = next_obs
                    test_return = test_return + np.reshape(rewards, [4])
                    total_survive_time = total_survive_time + 1
                    break_rewards = np.reshape(rewards, [4])
                    if break_rewards[3] < -1 and break_rewards[0] < 9:
                        out+=1
                        break
                    if break_rewards[0] < -0.5:
                        lost+=1
                        break

                adv_r = adv_r + (test_return[3] - test_return[0])
                agent_r = agent_r + (test_return[3])
                adversaries_r = adversaries_r + (test_return[0])
            if MA_step == 0:
                MA_step = total_survive_time/10
                MA_AR =  agent_r/10
                MA_ADR = adversaries_r/10
                MA_ADV = adv_r/10
                MA_Lost = lost
                MA_Out = out
            else:
                MA_step = 0.9 * MA_step + 0.1 * total_survive_time / 10
                MA_AR = 0.9 * MA_AR + 0.1 * agent_r / 10
                MA_ADR = 0.9 * MA_ADR + 0.1 * adversaries_r / 10
                MA_ADV = 0.9 * MA_ADV + 0.1 * adv_r / 10
                MA_Lost = 0.9*MA_Lost+0.1*lost
                MA_Out = 0.9*MA_Out+0.1*out
            print("--------------------------------------------------------------------")
            print("Total Step:{}, MA Step:{}, MA ADV:{}, MA Lost:{}, MA Out:{}, Agent Return:{}, Adversary Return:{}, Advantage:{} ".format(total_num_step,MA_step,MA_ADV,MA_Lost,MA_Out,agent_r/10,adversaries_r/10,adv_r/10))
            logger.logkv("total_episode", ep_i)
            logger.logkv("total_numsteps", total_num_step)
            logger.logkv("Agent Return", agent_r/10)
            logger.logkv("Adversary Return", adversaries_r / 10)
            logger.logkv("Advantage", adv_r / 10)
            logger.logkv("Step", total_survive_time / 10)
            logger.logkv("OUT", out)
            logger.logkv("LOST", lost)
            logger.logkv("AD 1 Prediction Loss", pred_loss[0]/total_survive_time)
            logger.logkv("AD 2 Prediction Loss", pred_loss[1]/total_survive_time)
            logger.logkv("AD 3 Prediction Loss", pred_loss[2]/total_survive_time)
            logger.logkv("AG 1 Prediction Loss", pred_loss[3] / total_survive_time)
            for i in range(4):
                logger.logkv("policy_loss" + str(i), output[i][0])
                logger.logkv("critic_loss" + str(i), output[i][1])
                logger.logkv("belief_loss" + str(i), output[i][2])
                logger.logkv("alpha" + str(i), output[i][3])
                logger.logkv("ent" + str(i), output[i][4])
                if algs == 'ROMMEO':
                    logger.logkv("prior_loss" + str(i), output[i][5])

            logger.dumpkvs()
            agent.save(run_dir / 'model.pt')

parser = argparse.ArgumentParser()
parser.add_argument("--env_id", default="simple_tag", help="Name of environment")
parser.add_argument("--model_name", default="BMASAC",
                        help="Name of directory to store " +
                             "model/training contents")
parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
parser.add_argument("--n_rollout_threads", default=1, type=int)
parser.add_argument("--n_training_threads", default=6, type=int)
parser.add_argument("--buffer_length", default=int(1e6), type=int)
parser.add_argument("--n_episodes", default=40000, type=int)
parser.add_argument("--episode_length", default=25, type=int)
parser.add_argument("--steps_per_update", default=1, type=int)
parser.add_argument("--batch_size",
                        default=256, type=int,
                        help="Batch size for model training")
parser.add_argument("--n_exploration_eps", default=25000, type=int)
parser.add_argument("--init_noise_scale", default=0.3, type=float)
parser.add_argument("--final_noise_scale", default=0.0, type=float)
parser.add_argument("--save_interval", default=500, type=int)
parser.add_argument("--hidden_dim", default=256, type=int)
parser.add_argument("--lr", default=0.0003, type=float)
parser.add_argument("--tau", default=0.01, type=float)
parser.add_argument("--alpha", default=0.5, type=float)
parser.add_argument("--agent_alg",
                        default="PR2", type=str,
                        choices=['TDOM-AC', 'MADDPG', 'PR2','ROMMEO'])
parser.add_argument("--adversary_alg",
                        default="PR2", type=str,
                        choices=['TDOM-AC', 'MADDPG', 'PR2','ROMMEO'])
parser.add_argument("--discrete_action",
                        action='store_true')

config = parser.parse_args()

run(config)
