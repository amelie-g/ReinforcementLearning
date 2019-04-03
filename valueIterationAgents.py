# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.cur_values = util.Counter()

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # keep track of iterations
        it_counter = 0

        # loop through iterations
        while it_counter != self.iterations:

            # copy dictionary into cur_values
            self.cur_values = self.values.copy()

            # get states and loop through them
            states = self.mdp.getStates()
            for state in states:

                # if terminal state continue
                if self.mdp.isTerminal(state):
                    continue

                # get possible actions and loop through them
                actions = self.mdp.getPossibleActions(state)

                # dictionary of actions
                action_scores = util.Counter()

                # loop through actions
                for action in actions:

                    # get q value given current state and action
                    score = self.computeQValueFromValues(state, action)

                    # add qvalue to dictionary
                    action_scores[action] = score

                # add max q value to dictionary
                self.values[state] = action_scores[action_scores.argMax()]

            # update the counter
            it_counter += 1

        # copy dictionary again
        self.cur_values = self.values.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # get transition states and probabilities
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        q = 0

        # loop through transition states and probabilities
        for nextState, probability in transitions:

            # calculate q value using probability, reward, discount, and values in dictionary
            # sum probability, discount, and the next state's value
            q += self.cur_values[nextState] * self.discount * probability

            # add reward
            q += self.mdp.getReward(state, action, nextState)

        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # get possible actions given a state
        actions = self.mdp.getPossibleActions(state)

        # allocate a best choice for action
        best_action = None

        # set max to negative infinity
        max = float('-inf')

        # loop through actions
        for action in actions:

            # if there are no legal actions, return None
            if self.mdp.isTerminal(state):
                return None

            # compute q value based on given state and action
            bestQ = self.computeQValueFromValues(state, action)

            # reset max if less than q value
            if max < bestQ:
                # reset action choice and max
                best_action = action
                max = bestQ

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
