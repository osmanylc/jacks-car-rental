import mdp_models as mm
import policy_iteration as polit

# Create MDPs for each problem
jcr = mm.JCR_MDP()
jcr2 = mm.JCR_MDP_2()


for mdp in [jcr, jcr2]:
    # Solve both problems with policy iteration
    # and plot results.
    pi_list, v_list = polit.policy_iteration(mdp)
    pi = pi_list[-1]
    v = v_list[-1]

    # Plot optimal policy
    polit.visualize_policy_plot(pi, 'Optimal Policy')
    # Plot optimal value function
    polit.visualize_values(v, 'Optimal Value Function')
