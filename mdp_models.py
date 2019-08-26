import numpy as np
from itertools import product

class JCR_MDP:
    def __init__(self):
        self.states = self.init_states()
        self.discount = .9
        self.rental_rate_1 = 3
        self.rental_rate_2 = 4
        self.return_rate_1 = 3
        self.return_rate_2 = 2
        self.epsilon = 1e-10

        self.rent_rw = 10
        self.move_rw = -2
        self.p_tab = {}

        # We're making tables to store the Poisson PDF and CDF for speed 
        name_to_rate = dict(zip(['rent_1', 'rent_2', 'ret_1', 'ret_2'], 
            [self.rental_rate_1, self.rental_rate_2, 
             self.return_rate_1, self.return_rate_2]))
        
        self.pdfs = {name: self.make_poisson_pdf_table(rate) 
            for name, rate in name_to_rate.items()}
        self.cdfs = {name: self.make_poisson_cdf_table(rate) 
            for name, rate in name_to_rate.items()}

    def pdf(self, name, x):
        return self.pdfs[name][x]

    def cdf(self, name, x):
        return self.cdfs[name][x]
    
    def make_poisson_pdf_table(self, mu):
        poisson_pdf_table = {}
        
        for x in range(21):
            poisson_pdf_table[x] = self.poisson_pdf(mu, x)
        
        return poisson_pdf_table

    def make_poisson_cdf_table(self, mu):
        poisson_cdf_table = {}

        for x in range(21):
            poisson_cdf_table[x] = self.poisson_cdf(mu, x)

        return poisson_cdf_table

    def poisson_pdf(self, mu, x):
        return np.exp(-mu) * (mu ** x) / np.math.factorial(x)

    def poisson_cdf(self, mu, x):
        return sum(self.poisson_pdf(mu, x) for x in range(x+1))

    def p(self, s, a):
        ''' 
        p(s', r | s, a)
        
        Probabiliy of going to state ss and getting reward r, 
        given that we are at state s and take action a.
        
        The output is a dictionary mapping (s', r) pairs to 
        their probabilities.
        '''
        pdf = {}

        if a not in self.s_to_a(s):
            return pdf

        if (s, a) in self.p_tab:
            return self.p_tab[(s, a)]

        for ss1, ss2 in product(range(21), range(21)):
            ss = (ss1, ss2)
            p_r = self.r(s, a, ss)

            for r, p in p_r.items():
                pdf[(ss, r)] = p

        self.p_tab[(s, a)] = pdf
        
        return pdf

    def r(self, s, a, ss):
        """
        Distribution of rewards for (s, a) -> s' transition: 

        p(r, s' | s, a), but s' is fixed so the output is supposed
        to add up to p(s' | s, a).
        """
        p_r = dict()
        s1, s2 = s
        s1_beg, s2_beg = s1 - a, s2 + a

        ss1, ss2 = ss

        # Let's think in term of how many items we can rent out, 
        # since this determines how much reward we get.
        min_s1_rent = max(0, s1_beg - ss1)
        max_s1_rent = 20 - ss1
        min_s2_rent = max(0, s2_beg - ss2)
        max_s2_rent = 20 - ss2

        s1_rent_range = np.arange(min_s1_rent, max_s1_rent + 1)
        s2_rent_range = np.arange(min_s2_rent, max_s2_rent + 1)

        for s1_rent, s2_rent in product(s1_rent_range, s2_rent_range):
            r = self.calc_r(a, (s1_beg, s2_beg), (s1_rent, s2_rent)) # get the reward value

            # Get number of cars returned
            s1_ret = s1_rent + ss1 - s1_beg
            s2_ret = s2_rent + ss2 - s2_beg

            # Calculate the probability of these rental numbers
            p_s1_rent = self.pdf('rent_1', s1_rent)
            if ss1 == 0:  # add all additional rentals possible if s'1 = 0
                p_s1_rent += 1 - self.cdf('rent_1', s1_rent)

            p_s2_rent = self.pdf('rent_2', s2_rent)
            if ss2 == 0:  # add all additional rentals possible if s'2 = 0
                p_s2_rent += 1 - self.cdf('rent_2', s2_rent)

            p_s1_ret = self.pdf('ret_1', s1_ret)
            if s1_rent == max_s1_rent:  # add additional rets due to full lot
                p_s1_ret += 1 - self.cdf('ret_1', s1_ret)

            p_s2_ret = self.pdf('ret_2', s2_ret)
            if s2_rent == max_s2_rent: # add additional rets due to full lot
                p_s2_ret += 1 - self.cdf('ret_2', s2_ret)

            # Get the final probability
            p = p_s1_rent * p_s2_rent * p_s1_ret * p_s2_ret

            # Add this probability to the reward value
            if r in p_r:
                p_r[r] += p
            else:
                p_r[r] = p
        
        return p_r

    def calc_r(self, a, s_beg, s_rent):
        r = 0

        # Action reward
        r += self.move_rw * abs(a)

        # Add rental reward
        r += self.rent_rw * sum(s_rent)

        return r

    def s_to_a(self, s):
        c1, c2 = s # extract num cars in locations 1 and 2

        # how many cars we can move from 1 to 2
        # min(space in 2, cars in 1, 5)
        space2 = 20 - c2
        upper_lim = min(5, c1, space2)

        # how many cars we can move from 2 to 1
        space1 = 20 - c1
        lower_lim = -min(5, c2, space1)

        return list(range(lower_lim, upper_lim + 1))

    def init_states(self):
        states = []

        for c1 in range(21):
            for c2 in range(21):
                states.append((c1, c2))

        return states

    @staticmethod
    def initialize_v(states):
        # Give a value of 0 to every state
        v = dict()

        for s in states:
            v[s] = 0

        return v

    @staticmethod
    def initialize_pi(states, s_to_a):
        # Initialize the policy to take an arbitrary action for each state
        pi = dict()

        for s in states:
            # pi[s] = s_to_a(s)[0]
            pi[s] = 0

        return pi

    def __str__(self):
        return "JCR_MDP"



class JCR_MDP_2(JCR_MDP):
    def __init__(self):
        super(JCR_MDP_2, self).__init__()
        self.lot_rw = -4

    
    def calc_r(self, a, s_beg, s_rent):
        r = 0

        # Action reward
        r += self.move_rw * abs(a)

        if a > 0:
            r -= self.move_rw

        # Add full lot reward
        r += self.lot_rw * sum(1 if x > 10 else 0 for x in s_beg)

        # Add rental reward
        r += self.rent_rw * sum(s_rent)

        return r

    def __str__(self):
        return "JCR_MDP_2"
