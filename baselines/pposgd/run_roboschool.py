#!/usr/bin/env python
from mpi4py import MPI
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging, roboschool
from baselines import logger
from baselines.common.mpi_fork import mpi_fork
from baselines.pposgd import mlp_policy, pposgd_simple, pposgd_mpi
import sys

NUM_CPU = 8

def train(env_id, seed):

    whoami = mpi_fork(NUM_CPU)
    if whoami == "parent":
        return

    U.make_session(num_cpu=1).__enter__()
    logger.session(dir='logs', format_strs=['stdout', 'tensorboard']).__enter__()
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pi = pposgd_mpi.learn(env, policy_fn,
            max_timesteps=1e6,
            timesteps_per_batch=512,
            clip_param=0.2, entcoeff=0,
            optim_epochs=15, optim_stepsize=2e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95
        )
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
    train('RoboschoolHumanoid-v1', seed=0)


if __name__ == '__main__':
    main()
    # bench.load_results('logs')
