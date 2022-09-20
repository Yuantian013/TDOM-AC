import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from algorithms.tdomac import BMASAC
from algorithms.rommeo import ROMMEO
from algorithms.pr2 import PR2
from utils.utils import tensor_from_tensor_list
import gym
from gym import spaces, logger
import time
MSELoss = torch.nn.MSELoss()
USE_CUDA = False  # torch.cuda.is_available()

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
    # seed=np.random.randint(low=0,high=9999)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    algs=2*[config.agent_alg]
    sd = int(config.seed)
    if algs[0]=='TDOM-AC':
        agent = BMASAC.init_diff_env(alg_types=algs, tau=config.tau,
                                     lr=config.lr, alpha=config.alpha, hidden_dim=config.hidden_dim)
        log_path = './Log/Diff/TDOM-AC/' + str(sd)
    elif algs[0]=='ROMMEO':
        agent = ROMMEO.init_diff_env(alg_types=algs, tau=config.tau,
                                     lr=config.lr, alpha=config.alpha, hidden_dim=config.hidden_dim)
        log_path = './Log/Diff/ROMMEO/' + str(sd)
    elif algs[0]=='PR2':
        agent = PR2.init_diff_env(alg_types=algs, tau=config.tau,
                                  lr=config.lr, alpha=config.alpha, hidden_dim=config.hidden_dim)
        log_path = './Log/Diff/PR2/' + str(sd)

    action_space_1 = spaces.Box(low=np.array([-10.0]), high=np.array([10.]))
    action_space_2 = spaces.Box(low=np.array([-10.0]), high=np.array([10.]))

    replay_buffer = ReplayBuffer(config.buffer_length, agent.nagents,
                                 [obsp.shape[0] for obsp in [action_space_1,action_space_2]],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in [action_space_1,action_space_2]])
    t = 0
    i = 0
    from utils import logger

    logger.configure(dir=log_path, format_strs=['csv'])
    mean_time = 0
    time_count = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        obs = np.array([[1.],[1.]])
        obs = np.reshape(obs,[1,2])
        agent.prep_rollouts(device='cpu')
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable

            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(agent.nagents)]

            torch_agent_actions = agent.step(torch_obs, explore=True)


            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

            actions = [[10*ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            # print(actions)
            F1 = 0.8 * (-((actions[0][0] + 5) / 3) ** 2 - ((actions[0][1] + 5) / 3) ** 2)
            F2 = (-((actions[0][0] - 5) / 1) ** 2 - ((actions[0][1] - 5) / 1) ** 2)+10
            r = max(F1,F2)
            # if r[0] > 0:
            #     print(r[0],F1,F2)
            rewards=np.array([[r[0],r[0]]])
            # print(agent_actions,rewards)
            done = True
            dones = np.array([[done,done]])
            next_obs= obs
            replay_buffer.push(obs, agent_actions, rewards,next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            i+=1
            if (len(replay_buffer) >= 16 and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    agent.prep_training(device='gpu')
                else:
                    agent.prep_training(device='cpu')
                output=np.zeros([2,6])
                start = time.clock()
                for u_i in range(1):
                    for a_i in range(agent.nagents):
                        sample = replay_buffer.sample(min(config.batch_size,len(replay_buffer)),
                                                      to_gpu=USE_CUDA)

                        if algs[0] == 'TDOM-AC':
                            # policy loss,critic loss, belief loss, alpha, entropy
                            output[a_i][0], output[a_i][1], output[a_i][2], output[a_i][3], output[a_i][4] = agent.update(sample, a_i,logger=logger)
                        elif algs[0] == 'ROMMEO':
                            # policy loss,critic loss, belief loss, alpha, entropy, prior loss
                            output[a_i][0], output[a_i][1], output[a_i][2], output[a_i][3], output[a_i][4], output[a_i][5] = agent.update(sample, a_i,logger=logger)
                        elif algs[0] == 'PR2':
                            # policy loss,critic loss, belief loss, alpha, entropy
                            output[a_i][0], output[a_i][1], output[a_i][2], output[a_i][3], output[a_i][4] = agent.update(sample, a_i,
                                                  logger=logger)
                    agent.update_all_targets()
                agent.prep_rollouts(device='cpu')
                end = time.clock()
                runtime = end-start
                mean_time = mean_time+runtime
                time_count = time_count +1
                # print(mean_time/time_count)
            if (len(replay_buffer) >= 16) and ep_i % 10 == 0:
                torch_agent_actions = agent.step(torch_obs, explore=False)
                agent_actions = [10 * ac.data.numpy() for ac in torch_agent_actions]
                actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
                torch_agent_prediction = agent.belief(torch_obs, explore=False)
                pred_loss = np.zeros([2])
                for agent_num in range(2):
                    with torch.no_grad():
                        real_actions = tensor_from_tensor_list(torch_agent_actions, excl_idx=agent_num)
                        pred_actions = torch_agent_prediction[agent_num]
                        pred_loss[agent_num] = MSELoss(real_actions, pred_actions).item()

                F1 = 0.8 * (-((actions[0][0] + 5) / 3) ** 2 - ((actions[0][1] + 5) / 3) ** 2)
                F2 = (-((actions[0][0] - 5) / 1) ** 2 - ((actions[0][1] - 5) / 1) ** 2) + 10
                r = max(F1, F2)
                logger.logkv("total_numsteps", ep_i)
                logger.logkv("Play 1",actions[0][0][0])
                logger.logkv("Play 2", actions[0][1][0])
                logger.logkv("P1 Loss",pred_loss[0])
                logger.logkv("P2 Loss", pred_loss[1])
                logger.logkv("B1 Loss",output[0][2])
                logger.logkv("B2 Loss", output[1][2])
                logger.logkv("Return", r[0])
                logger.dumpkvs()
                print("Episode: {}, F1:{}, F2: {},Reward:{}".format(i, round(F1[0],2), round(F2[0],2), round(r[0]), 2))
                print("P1_Loss:{},P2_Loss:{}".format(pred_loss[0],pred_loss[1]))

                # print("Agent:{}, policy_loss: {},critic_loss:{},belief_loss:{},alpha:{}".format(10,round(output[0][0],2),round(output[0][10],2),round(output[0][2],2),output[0][3]))
                # print("Agent:{}, policy_loss: {},critic_loss:{},belief_loss:{},alpha:{}".format(2, round(output[10][0], 2),
                #                                                                        round(output[10][10], 2),
                #                                                                        round(output[10][2], 2),output[10][3]))
            if done:
                break




parser = argparse.ArgumentParser()
parser.add_argument("--env_id", default="Diff", help="Name of environment")
parser.add_argument("--model_name", default="TDOM-AC",
                        help="Name of directory to store " +
                             "model/training contents")
parser.add_argument("--seed",
                        default=999, type=int,
                        help="Random seed")
parser.add_argument("--n_rollout_threads", default=1, type=int)
parser.add_argument("--n_training_threads", default=6, type=int)
parser.add_argument("--buffer_length", default=int(1e6), type=int)
parser.add_argument("--n_episodes", default=2501, type=int)
parser.add_argument("--episode_length", default=25, type=int)
parser.add_argument("--steps_per_update", default=1, type=int)

# best parameters 64 0.003,10 2501/1001 unit 64
#
parser.add_argument("--batch_size",
                        default=512, type=int,
                        help="Batch size for model training")
parser.add_argument("--n_exploration_eps", default=25000, type=int)
parser.add_argument("--init_noise_scale", default=0.3, type=float)
parser.add_argument("--final_noise_scale", default=0.0, type=float)
parser.add_argument("--save_interval", default=500, type=int)
parser.add_argument("--hidden_dim", default=128, type=int)
parser.add_argument("--lr", default=0.01, type=float)
parser.add_argument("--tau", default=0.01, type=float)
parser.add_argument("--alpha", default=0., type=float)
parser.add_argument("--agent_alg",
                        default="TDOM-AC", type=str,
                        choices=['TDOM-AC', 'ROMMEO', 'PR2'])
parser.add_argument("--adversary_alg",
                        default="TDOM-AC", type=str,
                        choices=['TDOM-AC', 'ROMMEO', 'PR2'])
parser.add_argument("--discrete_action",
                        action='store_true')

config = parser.parse_args()

run(config)


# 64 0.01 *[10/2 not min] *alpha= 10 much better # 决策的时候永远用mean的op

