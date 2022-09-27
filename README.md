# TDOM-AC
Code for [Multi-agent Actor-Critic with Time Dynamical Opponent Model](https://arxiv.org/pdf/2204.05576)


## Introduction
In multi-agent reinforcement learning, multiple agents learn simultaneously while interacting with a common environment and each other. Since the agents adapt their policies during learning, not only the behavior of a single agent becomes non-stationary, but also the environment as perceived by the agent. This renders it particularly challenging to perform policy improvement. In this paper, we propose to exploit the fact that the agents seek to improve their expected cumulative reward and introduce a novel Time Dynamical Opponent Model (TDOM) to encode the knowledge that the opponent policies tend to improve over time. We motivate TDOM theoretically by deriving a lower bound of the log objective of an individual agent and further propose Multi-Agent Actor-Critic with Time Dynamical Opponent Model (TDOM-AC). We evaluate the proposed TDOM-AC on a differential game and the Multi-agent Particle Environment. We show empirically that TDOM achieves superior opponent behavior prediction during test time. The proposed TDOM-AC methodology outperforms state-of-the-art Actor-Critic methods on the performed experiments in cooperative and especially in mixed cooperative-competitive environments. TDOM-AC results in a more stable training and a faster convergence.

## Dependencies
```bash
conda create --name tdomac python=3.7
conda activate tdomac
git clone https://github.com/Yuantian013/TDOM-AC
cd TDOM-AC-main
pip install -r requirements.txt
```
## OpenAI Baseline
- Download and install the OpenAI Baseline code into the main folder [here](https://github.com/openai/baselines)
by following the `README`.

## Multi-Agent Particle Environments

- Download and install the MPE code into the main folder [here](https://github.com/openai/multiagent-particle-envs)
by following the `README`.

- Ensure that `multiagent-particle-envs` has been added to your `PYTHONPATH` (e.g. in `~/.bashrc` or `~/.bash_profile`).

### Training
You can simply train agents via different algorithms and hyper-paramters.
```bash
cd BMASAC
# Differential Game
python training_diff.py 
# Navigation
python train_navi.py
# Predator and Prey
python train_tag.py
```
You can change the algorithm and hyper-paramters by changing the arguments.

### Plot
You can plot all the recorded information of your training 
![image](https://github.com/Yuantian013/TDOM-AC/blob/main/git1.png)：
![image](https://github.com/Yuantian013/TDOM-AC/blob/main/git2.png)：
![image](https://github.com/Yuantian013/TDOM-AC/blob/main/git3.png)：
```bash
cd BMASAC
python plot.py
```

You can switch different algorithms by commenting or changing the [alg_list],you can switch different env by commenting or changing the [env], and you can switch different results by commenting or changing the [content]

### Visulization
You can visulize your agent behavior and even mix different agents that trained by different algoirthms together.
![image](https://github.com/Yuantian013/TDOM-AC/blob/main/p7p.gif)：

```bash
cd BMASAC
python mix_evaluate.py
```

We provide different model for evaluate and visulize the performance of [Predator and Prey] task. You can choose Predator and Prey trained by different algorithm by changing the [agent_run_num] and [adversary_run_num].


## Citation
Please cite our work if you find it useful.
```bibtex
@misc{tian2022multi,
  title={Multi-agent Actor-Critic with Time Dynamical Opponent Model},
  author={Tian, Yuan and Kladny, Klaus-Rudolf and Wang, Qin and Huang, Zhiwu and Fink, Olga},
  eprint={2204.05576},
  archivePrefix={arXiv},
  year={2022}
}
```
For questions regarding the code, please open an issue or contact Yuan via email {yutian} AT ethz.ch
