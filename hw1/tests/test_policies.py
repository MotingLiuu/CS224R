from cs224r.policies.MLP_policy import MLPPolicySL
from cs224r.infrastructure import pytorch_util as ptu
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

    def test_forward(self):
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

        observation = np.array([[1.0, 2.0, 3.0]])
        observation_tensor = ptu.from_numpy(observation)
        dist = policy.forward(observation_tensor)

        logger.debug("Distribution output: %s", dist)

        assert isinstance(dist, torch.distributions.Independent), "Expected output to be an Independent distribution"

    def test_update(self):
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

        observations = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        actions = np.array([[0.5, -0.5], [-0.5, 0.5]])

        update_info = policy.update(observations, actions)
        logger.debug("Update info: %s", update_info)

        assert 'Training Loss' in update_info, "Expected 'Training Loss' key in update info"