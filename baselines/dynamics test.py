from baselines import bench
import roboschool, gym
import os
import numpy as np

ENV = 'RoboschoolWalker2d-v1'
LOGDIR = '/Users/beni/PycharmProjects/baselines/baselines/logs/' + ENV + '/'
STEPS = 10

def test_write(env):
    env.dylog.delete_logs()
    ob = env.reset()
    done = False
    for i in range(STEPS):
        while not done:
            ac = env.action_space.sample()
            ob, rew, done, _ = env.step(ac)
            # env.render()
    # env.dylog._save_log()
    return env


def test_read(dl):
    dl.load_logs()
    # print(env.dylog.transitions)
    return dl

def get_all_vals(transitions):
    all_vals = []
    for i in range(len(transitions)):
        transition = transitions[i]
        for val in transition[0]:
            all_vals.append(val)
        for val in transition[1]:
            all_vals.append(val)
        all_vals.append(transition[2])
        all_vals.append(transition[3])
    return all_vals


if __name__ == '__main__':

    if not os.path.exists(LOGDIR):
        os.mkdir(LOGDIR)

    env = gym.make(ENV)
    env = bench.Monitor(env, LOGDIR + 'monitor.json', allow_early_resets=True, log_dynamics=True)

    env = test_write(env)

    dl = bench.DynamicsLogger(LOGDIR)
    dl.load_logs()

    assert len(env.dylog.transitions_copy) == len(dl.transitions), 'env: {}, loaded: {}'.format(len(env.dylog.transitions_copy),
                                                                                           len(dl.transitions))

    all_vals_env = get_all_vals(env.dylog.transitions_copy)
    all_vals_loaded = get_all_vals(dl.transitions)

    print(all_vals_env)
    print(all_vals_loaded)

    for i in range(len(all_vals_env)):
        if not np.isclose(all_vals_env[i], all_vals_loaded[i]):
            print(all_vals_env[i], all_vals_loaded[i])
    np.testing.assert_allclose(all_vals_env, all_vals_loaded, rtol=0.0001)





