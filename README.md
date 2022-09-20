# TDOM-AC
Code for [Multi-agent Actor-Critic with Time Dynamical Opponent Model](https://arxiv.org/pdf/2204.05576)


## Introduction
In multi-agent reinforcement learning, multiple agents learn simultaneously while interacting with a common environment and each other. Since the agents adapt their policies during learning, not only the behavior of a single agent becomes non-stationary, but also the environment as perceived by the agent. This renders it particularly challenging to perform policy improvement. In this paper, we propose to exploit the fact that the agents seek to improve their expected cumulative reward and introduce a novel \textit{Time Dynamical Opponent Model} (TDOM) to encode the knowledge that the opponent policies tend to improve over time. We motivate TDOM theoretically by deriving a lower bound of the log objective of an individual agent and further propose \textit{Multi-Agent Actor-Critic with Time Dynamical Opponent Model} (TDOM-AC). We evaluate the proposed TDOM-AC on a differential game and the Multi-agent Particle Environment. We show empirically that TDOM achieves superior opponent behavior prediction during test time. The proposed TDOM-AC methodology outperforms state-of-the-art Actor-Critic methods on the performed experiments in cooperative and \textbf{especially} in mixed cooperative-competitive environments. TDOM-AC results in a more stable training and a faster convergence.


### Training 
training_diff.py/train_navi.py/train_tag.py
You can change the algorithm and parameters by changing the arguments

### Plot
plot.py
You can switch different algorithms by commenting or changing the [alg_list],you can switch different env by commenting or changing the [env], and you can switch different results by commenting or changing the [content]

### Visulization
mix_evaluate.py

We provide different model for evaluate and visulize the performance of [Predator and Prey] task. You can choose Predator and Prey trained by different algorithm by changing the [agent_run_num] and [adversary_run_num].

## Run E2GAN search on CIFAR-10

## Citation
Please cite our work if you find it useful.
```bibtex
@InProceedings{Tian_2020_ECCV,
author = {Yuan Tian, Qin Wang, Zhiwu Huang, Wen Li, Dengxin Dai, Minghao Yang, Jun Wang, Olga Fink},
title = {Off-Policy Reinforcement Learning for Efficient and Effective GAN Architecture Search},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2020}
}
```
For questions regarding the code, please open an issue or contact Yuan and Qin via email {yutian} AT ethz.ch
