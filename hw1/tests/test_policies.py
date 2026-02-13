from cs224r.policies.MLP_policy import MLPPolicySL
import numpy as np
import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)

class TestMLPPolicySL:

    def test_get_action(self):
        ob_dim = 3
        ac_dim = 2
        n_layers = 2
        size = 16

        policy = MLPPolicySL(
            ac_dim=ac_dim,
            ob_dim=ob_dim,
            n_layers=n_layers,
            size=size,
        )

        # Test single observation
        observation = np.array([[1.0, 2.0, 3.0]])
        logger.debug("observation shape %s", observation.shape)
        action = policy.get_action(observation)
        assert action.shape == (1, ac_dim), f"Expected action shape (1, {ac_dim}), but got {action.shape}"
        
        logger.debug("Single observation action: %s", action)

        # Test batch of observations
        observations = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        actions = policy.get_action(observations)
        assert actions.shape == (2, ac_dim), f"Expected actions shape (2, {ac_dim}), but got {actions.shape}"

        logger.debug("Batch observations actions: %s", actions)