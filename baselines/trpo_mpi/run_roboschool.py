#!/usr/bin/env python
# noinspection PyUnresolvedReferences

from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym, roboschool
import logging
from baselines import logger
from baselines.pposgd.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
import baselines.common.tf_util as U
import sys
num_cpu=8

def train(env_id, num_timesteps, seed):
    whoami = mpi_fork(num_cpu)
    if whoami == "parent":
        return
    logger.session().__enter__()
    sess = U.single_threaded_session()
    sess.__enter__()
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=32, num_hid_layers=2)
    env = bench.Monitor(env, osp.join(logger.get_dir(), "%i.monitor.json" % rank), allow_early_resets=True)
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    pi = trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    if rank == 0:
        input()
        play(env, 10, pi)
    env.close()

def play(env, num_runs, policy_fn):
    for _ in range(num_runs):
        ob = env.reset()
        done = False
        while not done:
            ac, _ = policy_fn.act(True, ob)
            ob, rew, done, _ = env.step(ac)
            env.render()

def main():
    train('RoboschoolHopper-v1', num_timesteps=1e6, seed=0)

if __name__ == '__main__':
    main()
