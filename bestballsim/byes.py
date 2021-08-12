import logging

import numpy as np
import pandas as pd


logging.getLogger(__name__).addHandler(logging.NullHandler())
RNG = np.random.default_rng()


def addbye(players: np.ndarray, n_same_bye: int) -> np.ndarray:
    """Adds bye week to array
    
    Args:
        players (np.ndarray): ndarray of scores
        n_same_bye (int): the number of overlapping byes (in addition to first player)

    Returns:
        np.ndarray

    """
    # start by giving everyone the same bye (week 1 a/k/a index 0)
    # then roll byes sequentially
    # the second player would get bye in week 2 for all of the relevant rows
    # the third player would get bye in week 3 for all of the relevant rows
    # then the remaining players keep the same bye with the original player

    if players.ndim == 2:
        if n_same_bye + 1 > players.shape[0]:
            raise ValueError(f'n_same_bye is greater than number of players')
        sp = np.insert(players, 0, 0, axis=-1)   
        byes_to_add = players.shape[0] - n_same_bye
        if byes_to_add > 0:        
            for i in np.arange(1, byes_to_add):
                sp[i, :] = np.roll(sp[i, :], shift=i, axis=-1)
        return sp
    
    elif players.ndim == 3: 
        sp = np.insert(players, 0, 0, axis=-1)   
        byes_to_add = players.shape[1] - n_same_bye       
        for i in np.arange(1, byes_to_add):
            sp[:, i, :] = np.roll(sp[:, i, :], shift=i, axis=-1)
        return sp


def bbscoring(players: np.ndarray, n_slots: int) -> np.ndarray:
    """Calculates bestball scoring for n_slots
    
    Args:
        players (np.ndarray): the 2D array of scores
        n_slots (int): the number of players who count

    Returns:
        np.ndarray of shape 1, players.shape[1]

    """
    if n_slots == 1:
        return players.max(axis=0)
    index_array = np.argpartition(-players, kth=1, axis=0)
    return np.take_along_axis(players, index_array[0:n_slots], axis=0)


def byesim(players: np.ndarray, 
           n_same_bye: int, 
           n_slots: int,
           shuffles: int = 100) -> np.ndarray:
    """Simulates bye week
    
    Args:
        players (np.ndarray): the 2D array of scores
        n_same_bye (int): number of players that share same bye (0 for none, 1 means both)
        n_slots (int): the number of players who count
        shuffles (int): the number of times to shuffle weekly scoring

    Returns:
        np.ndarray

    """
    # step one: get the shuffled players
    # note that "shuffled players" are different combinations of weekly scores
    # sp has a shape of (rows, players.shape[0], players.shape[1])
    sp = shuffled_players(players, low=0, high=players.shape[1], rows=shuffles)

    # step two: add byes
    sp = addbye(players, n_same_bye)

    # now calculate the scores
    return np.array([bbscoring(sp[i], n_slots) for i in np.arange(sp.shape[0])])


def shuffled_indices(low: int = 0, high: int = 16, rows: int = 100) -> np.ndarray:
    """Gets 2D array of shuffled indices
       Used to randomly sort scores to smooth out
       effect of too good or terrible overlap of big and small weeks

    Args:
        low (int): low index value, default 0
        high (int): high index value, default 16
        rows (int): the number of rows in the shuffled array, default 100

    Returns:
        np.ndarray

    """
    return RNG.integers(low, high, size=(rows, high)).argsort(axis=1)


def shuffled_players(players: np.ndarray, low: int = 0, high: int = 16, rows: int = 100) -> np.ndarray:
    """Shuffles 2D array of players
    
    Args:
        players (np.ndarray): the players array
        low (int): low index value, default 0
        high (int): high index value, default 16
        rows (int): the number of rows in the shuffled array, default 100

    Returns:
        np.ndarray of shape (rows, players.shape[0], players.shape[1])

    """
    # get the shuffled indices
    idx = shuffled_indices(low, high, rows * players.shape[0]).reshape(rows, players.shape[0], high)

    # duplicate the players to match the shape of the shuffled indices
    nplayers = np.repeat(players[np.newaxis,...], idx.shape[0], axis=0)

    # make sure that the shape is the same
    assert idx.shape == nplayers.shape

    # get the player values according to the shuffled indices
    return np.take(nplayers, idx)



