# bestballsim/tests/test_byes.py
# -*- coding: utf-8 -*-
# Copyright (C) 2021 Eric Truett
# Licensed under the MIT License

import numpy as np
from numpy.core.numeric import ones
import pandas as pd

import pytest


from bestballsim.byes import *


def onesie_data(n=2, w=16):
    return np.random.randint(low=1, high=30, size=(n, w))


@pytest.fixture
def season_data(test_directory):
    return pd.read_csv(test_directory / 'season_data.csv')


@pytest.fixture
def weekly_data(test_directory):
    return pd.read_csv(test_directory / 'weekly_data.csv')


def test_addbye_invalid_n_same_bye(tprint):
    """Tests byesim on the onesie positions with invalid n_same_bye"""
    players = onesie_data()
    with pytest.raises(ValueError):
        new_players = addbye(players, n_same_bye=3)


def test_addbye_2D_valid_n_same_bye():
    """Tests byesim on the onesie positions with valid n_same_bye"""
    players = onesie_data()
    new_players = addbye(players, n_same_bye=0)
    assert new_players[0, 0] == 0
    assert new_players[1, 0] != 0

    players = onesie_data(n=4)
    new_players = addbye(players, n_same_bye=1)
    assert new_players[0, 0] == 0
    assert new_players[1, 0] != 0
    assert new_players[2, 0] != 0
    assert new_players[3, 0] == 0


def test_addbye_3D_valid_n_same_bye(tprint):
    """Tests byesim on the onesie positions with valid n_same_bye"""
    players = onesie_data()
    rows = 5
    sp = shuffled_players(players, rows=rows)
    new_players = addbye(sp, n_same_bye=0)
    assert np.array_equal(new_players[:, 0, 0], np.zeros(rows))
    assert np.array_equal(new_players[:, 1, 1], np.zeros(rows))

    players = onesie_data(n=4)
    rows = 5
    sp = shuffled_players(players, rows=rows)
    new_players = addbye(sp, n_same_bye=1)
    assert np.array_equal(new_players[:, 0, 0], np.zeros(rows))
    assert np.array_equal(new_players[:, 1, 1], np.zeros(rows))
    assert np.array_equal(new_players[:, 3, 0], np.zeros(rows))


def test_shuffled_indices():
    """Tests shuffled_indices"""
    # test that shuffle changes order
    high = 16
    rows = 1
    players = onesie_data()
    idx = shuffled_indices(0, high, rows)
    assert idx.shape == (rows, high)
    shuffled_players = players[:, idx[0]]
    assert players.tolist() != shuffled_players.tolist()
    assert set(players[0]) == set(shuffled_players[0])


def test_bb_scoring_onesie():
    """Tests bb_scoring for onesie position"""
    players = np.array([[1, 2, 3, 10], [10, 10, 10, 1]])
    assert np.array_equal(bbscoring(players, 1), np.array([10, 10, 10, 10]))


def test_bb_scoring_multi(tprint):
    """Tests bb_scoring for multiplayer position"""
    players = (
        np.array([
            [1, 2, 2, 1], 
            [10, 4, 1, 1],
            [20, 20, 20, 20]
        ])
    )
    
    scoring = bbscoring(players, 2)
    expected_scoring = np.array([[20] * 4, [10, 4, 2, 1]])
    assert np.array_equal(scoring, expected_scoring)


def test_shuffled_players(tprint):
    """Tests shuffled_players"""
    players = onesie_data()
    low = 0
    high = players.shape[1]
    rows = 5
    new_players = shuffled_players(players, low, high, rows)
    assert new_players.shape == (rows, players.shape[0], players.shape[1])


def test_byesim(tprint):
    """Tests byesim"""
    n_players = 5
    players = onesie_data(n=n_players)
    sp = shuffled_players(players, low=0, high=players.shape[1], rows=n_players)
    bs = byesim(players, n_same_bye=2, n_slots=1, shuffles=1)

