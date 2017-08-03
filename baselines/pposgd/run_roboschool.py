#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging, roboschool
from baselines import logger
import sys

def train(env_id, num_eps, seed):
    from baselines.pposgd import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    logger.session().__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pi = pposgd_simple.learn(env, policy_fn,
            max_episodes=num_eps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95
        )
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
    train('RoboschoolInvertedPendulumSwingup-v1', num_eps=2000, seed=0)


if __name__ == '__main__':
    main()
