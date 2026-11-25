"""Portable tri-mode recipe checks; set SUPERREACH_SCENE for native transport."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco as mj
import numpy as np
import torch

from superreach.plant import Plant, load_model, new_actor, targets
from superreach.recipe import actor_lr, directional_terms, effective, mode_at, observed, schedule


class RecipeTests(unittest.TestCase):
    def test_goal_switch_schedule(self):
        codes = schedule(512, 'cpu', torch.Generator().manual_seed(7))
        self.assertEqual(set(mode_at(codes, 0).tolist()), {0, 1, 2})
        switching = codes // 9 < 301
        self.assertTrue(switching.any())
        switch = codes[switching] // 9
        self.assertTrue(((switch >= 80) & (switch <= 220)).all())
        self.assertTrue((mode_at(codes[switching], switch - 1) != mode_at(codes[switching], switch)).all())
        self.assertEqual(actor_lr(10000), .0002)
        self.assertEqual(actor_lr(15000), .00005)

    def test_noisy_packet_keeps_command_and_validity(self):
        row = torch.linspace(0, 1, 20)[None]
        noisy = observed(row, torch.ones(1, 7)*.0015, torch.Generator().manual_seed(7))
        self.assertTrue((noisy[:, :12] != row[:, :12]).all())
        self.assertTrue(torch.equal(noisy[:, 12:], row[:, 12:]))

    def test_directional_penalty(self):
        error = torch.tensor([[.05, 0., 0.]], dtype=torch.float64).repeat(2, 1)
        speed = .35 * np.tanh(3 * .05 / .35)
        velocity = torch.tensor([[speed, 0., 0.], [-speed, 0., 0.]], dtype=torch.float64)
        approach, away = directional_terms(error, error.norm(dim=-1), velocity, error, error, torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(approach, torch.tensor([0., -.8*(2*speed/.35)**2], dtype=torch.float64))
        torch.testing.assert_close(away, torch.tensor([0., -8*.05*speed], dtype=torch.float64))

    @unittest.skipUnless(os.getenv('SUPERREACH_SCENE'), 'Set SUPERREACH_SCENE to enable native tests')
    def test_native_transport_and_fk(self):
        model = load_model(Path(os.environ['SUPERREACH_SCENE']))
        actor = new_actor(model)
        step = mj.mj_step
        for delay in range(6):
            plant = Plant(model, 42, noisy=True, delay=delay)
            original = plant.history.clone()
            previous = plant.issued.clone()
            sent = previous + .01
            applied = []
            def spy(model, data):
                applied.append(data.ctrl[:6].copy())
                step(model, data)
            with patch.object(mj, 'mj_step', side_effect=spy):
                state = plant.step(sent)
            self.assertEqual(len(applied), 10)
            np.testing.assert_array_equal(applied, torch.cat([previous]*delay+[sent]*(10-delay)))
            torch.testing.assert_close(torch.tensor(np.mean(applied, 0)).float(), effective(previous, sent, delay)[0])
            torch.testing.assert_close(plant.history[:, :-1], original[:, 1:], rtol=0, atol=0)
            torch.testing.assert_close(plant.history[:, -1, 12:18], sent, rtol=0, atol=0)
            pose, _ = targets(actor.fk, state[:, :6], sent)
            np.testing.assert_allclose(pose[0,:3].numpy(), plant.data.site_xpos[actor.fk.site], atol=1e-6, rtol=0)
