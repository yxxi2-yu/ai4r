#!/usr/bin/env python

import numpy as np

class RLPolicy:
    """
    This class implements a simple wrapper for calling an RL policy

    Class variables:
    - rl_model   :  the trained rl model
    """

    def __init__(self, rl_model):
        """
        Initialization function for the "RLPolicy" class.
        """
        self.rl_model = rl_model


    def compute_action(self, observation, info_dict, terminated, truncated):
        # Call the rl model
        action = self.rl_model.predict(observation, deterministic=True)
        # Return the action
        return action

