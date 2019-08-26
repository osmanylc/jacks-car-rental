import pickle
import numpy as np
from matplotlib import pyplot as plt


def policy_iteration(mdp):
    # ::::: Initialization :::::
    v = mdp.initialize_v(mdp.states)
    pi = mdp.initialize_pi(mdp.states, mdp.s_to_a)
    tolerance = 1e-3

    v_list = [v.copy()]
    pi_list = [pi.copy()]

    policy_stable = False
    policy_its = 1

    while not policy_stable:
        print(f'::::: Policy Iteration {policy_its}')
        # ::::: Policy Evaluation :::::
        # Iterate until residual is smaller than tolerance
        num_it = 1
        resid = np.inf
        while resid > tolerance:
            print(f'::::: Running Policy Evaluation {num_it}...')
            resid = 0
            for s in mdp.states:
                v_old = v[s]
                a = pi[s]
                p_sa = mdp.p(s,a)

                v_new = 0
                for (ss, r), p in p_sa.items():
                    v_new += p * (r + mdp.discount * v[ss])
                v[s] = v_new
                
                resid = max(resid, abs(v_old - v[s])) 

            print(f'Residual: {resid:10.4f}')
            num_it += 1
        v_list.append(v.copy())

        # ::::: Policy Improvement :::::
        print('::::: Running Policy Improvement...')
        policy_stable = True
        num_a_changed = 0
        for s in mdp.states:
            old_a = pi[s]
            a_vals = dict()

            # Evaluate the value of taking each action at state s
            for a in mdp.s_to_a(s):
                a_vals[a] = 0
                p_sa = mdp.p(s, a)

                for (ss, r), p in p_sa.items():
                    a_vals[a] += p * (r + mdp.discount * v[ss])
            
            pi[s] = max(a_vals.keys(), key=lambda a: a_vals[a])
            
            if old_a != pi[s] and a_vals[old_a] < a_vals[pi[s]]:
                num_a_changed += 1
                policy_stable = False
        print(f'Number of state actions changed: {num_a_changed}')
        print('\n')
        pi_list.append(pi.copy())
        visualize_policy(pi)
        policy_its += 1

    with open('policy_iteration_results_4.7.pkl', 'wb') as f:
        pickle.dump({'pi_list': pi_list, 'v_list': v_list}, f)
    
    return pi_list, v_list


def visualize_policy(pi):
    out = []
    
    for s1 in range(20, -1, -1):
        for s2 in range(21):
            val = pi[(s1, s2)]
            out.append(f'{val:2}')
        out.append('\n')
    
    out = ''.join(out)
    print(out)


def d_to_arr(d):
    d_arr = np.zeros((21, 21))

    for (s1, s2), val in d.items():
        d_arr[s1, s2] = val
    
    return d_arr


def visualize_values(v):
    v_arr = d_to_arr(v)

    plt.pcolormesh(v_arr)
    plt.colorbar()

    plt.title('Optimal Policy')
    plt.ylabel('Cars in Location 1')
    plt.xlabel('Cars in Location 2')

    plt.show()


def load_v_pi(fname):
    with open(fname, 'rb') as f:
        d = pickle.load(f)
    return d['v_list'][-1], d['pi_list'][-1]
