# pylint: disable=C0103
import numpy as np
import time


def apply_controller(env,
                     policy,
                     max_steps,
                     preprocess=None,
                     callback=None,
                     realtime=True):
    '''
        Starts the env and applies the current policy to the env for a duration
        specified by H (in seconds). If  H is not set, it will run for self.H
        seconds.
        @param env interface to the system being controller
        @param policy Interface to the controller to be applied to the system
        @param max_steps Horizon for applying controller (in seconds)
        @param callback Callable object to be called after every time step
    '''
    fnname = 'apply_controller'
    # initialize policy if needed
    if hasattr(policy, 'get_params'):
        p = policy.get_params()
        if len(p) == 0:
            policy.init_params()
        # making sure we initialize the policy before resetting the plant
        policy(np.zeros((policy.D, )))

    # start robot
    print fnname, 'Starting run'
    if hasattr(env, 'dt'):
        H = max_steps * env.dt
        print fnname, 'Running for %f seconds' % (H)
    else:
        print fnname, 'Running for %d steps' % (max_steps)
    x_t = env.reset()

    # data corresponds to state at time t, action at time t, reward after
    # applying action at time t
    data = []

    # do rollout
    dt = env.dt
    t_ = time.time()
    for t in range(max_steps):
        # preprocess state
        x_t_ = preprocess(x_t) if callable(preprocess) else x_t

        #  get command from policy
        u_t = policy(x_t_, t=t)
        if isinstance(u_t, list) or isinstance(u_t, tuple):
            u_t = u_t[0].flatten()
        else:
            u_t = u_t.flatten()

        # apply control and step the env
        x_next, c_t, done, info = env.step(u_t)
        info['done'] = done

        # append to dataset
        data.append((x_t, u_t, c_t, info))

        # send data to callback
        if callable(callback):
            callback(x_t, u_t, c_t, info)

        # break if done
        if done:
            break

        # replace current state
        x_t = x_next

        exec_time = time.time() - t_
        if realtime:
            time.sleep(max(0.75 * dt - exec_time, 0))
        t_ = time.time()

    states, actions, costs, infos = zip(*data)

    msg = 'Done. Stopping robot.'
    if all([v is not None for v in costs]):
        run_value = np.array(costs).sum()
        msg += ' Value of run [%f]' % run_value
    print fnname, msg

    # stop robot
    if hasattr(env, 'stop'):
        env.stop()

    return states, actions, costs, infos