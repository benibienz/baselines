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

ENV = 'RoboschoolWalker2d-v1'
NUM_CPU = 8
LOGDIR = '/Users/beni/PycharmProjects/baselines/baselines/logs/' + ENV

def train(env_id, seed):

    whoami = mpi_fork(NUM_CPU)
    if whoami == "parent":
        return

    U.make_session(num_cpu=1).__enter__()
    logger.session(dir=LOGDIR, format_strs=['stdout', 'tensorboard']).__enter__()
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
    env.init_dynamics_logger(overwrite=False, rank=rank)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pi = pposgd_mpi.learn(env, policy_fn,
            max_timesteps=2e6,
            timesteps_per_batch=256,
            clip_param=0.2, entcoeff=0,
            optim_epochs=5, optim_stepsize=4e-4, optim_batchsize=64,
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
    train(ENV, seed=0)


if __name__ == '__main__':
    main()
    # bench.load_results('logs')
    ''' RoboschoolInvertedPendulum-v0
        RoboschoolInvertedPendulumSwingup-v0
        RoboschoolInvertedDoublePendulum-v0
        RoboschoolReacher-v0
        RoboschoolHopper-v0
        RoboschoolWalker2d-v0
        RoboschoolHalfCheetah-v0
        RoboschoolAnt-v0
        RoboschoolHumanoid-v0
        RoboschoolHumanoidFlagrun-v0
        RoboschoolHumanoidFlagrunHarder-v0
        RoboschoolPong-v0'''
