import math

from byu_pytest_utils import max_score
from tsp_core import Timer, generate_network, score_tour
from math import inf

from tsp_solve import greedy_tour, dfs, branch_and_bound, branch_and_bound_smart



def assert_valid_tour(edges, tour):
    """
    Length is number of vertices
    Not vertices repeated
    Non-infinite score
    """
    assert len(tour) == len(edges)
    assert len(tour) == len(set(tour))
    assert not math.isinf(score_tour(tour, edges))


def assert_valid_tours(edges, stats):
    for stat in stats:
        assert_valid_tour(edges, stat.tour)


def test_greedy():
    graph = [
        [0, 9, inf, 8, inf],
        [inf, 0, 4, inf, 2],
        [inf, 3, 0, 4, inf],
        [inf, 6, 7, 0, 12],
        [1, inf, inf, 10, 0]
    ]
    # timer = Timer(10)
    timer = Timer(1000000000000)
    stats = greedy_tour(graph, timer)
    assert_valid_tours(graph, stats)

    # print(stats)
    assert stats[0].tour == [1, 4, 0, 3, 2]
    assert stats[0].score == 21

    assert len(stats) == 1

def test_dfs():
    graph = [
        [0, 9, inf, 8, inf],
        [inf, 0, 4, inf, 2],
        [inf, 3, 0, 4, inf],
        [inf, 6, 7, 0, 12],
        [1, inf, inf, 10, 0]
    ]
    # timer = Timer(10)
    timer = Timer(1000000000000)
    stats = dfs(graph, timer)
    assert_valid_tours(graph, stats)

    scores = {
        tuple(stat.tour): stat.score
        for stat in stats
    }
    assert scores[0, 3, 2, 1, 4] == 21
    assert len(scores) == 1

def my_test_branch_and_bound():
    graph = [
        [0, 9, inf, 8, inf],
        [inf, 0, 4, inf, 2],
        [inf, 3, 0, 4, inf],
        [inf, 6, 7, 0, 12],
        [1, inf, inf, 10, 0]
    ]
    # timer = Timer(10)
    timer = Timer(1000000000000)
    stats = branch_and_bound(graph, timer)
    assert_valid_tours(graph, stats)

    scores = {
        tuple(stat.tour): stat.score
        for stat in stats
    }
    assert scores[0, 3, 2, 1, 4] == 21
    assert len(scores) == 1



def test_branch_and_bound():
    """
    - Greedy should run almost instantly.
    - B&B should search the entire space in less than 3 minutes.
      (A good implementation should finish in seconds).
    - B&B should find a better score than greedy (on this graph).
    """

    # locations, edges = generate_network(
    #     15,
    #     euclidean=True,
    #     reduction=0.2,
    #     normal=False,
    #     seed=312,
    # )    
    
    locations, edges = generate_network(
        # 15,
        10,
        euclidean=True,
        reduction=0.2,
        normal=False,
        seed=306,
    )

    timer = Timer(5)
    greedy_stats = greedy_tour(edges, timer)
    assert not timer.time_out()
    assert_valid_tours(edges, greedy_stats)
    greedy_score = score_tour(greedy_stats[-1].tour, edges)

    timer = Timer(120)
    stats = branch_and_bound(edges, timer)
    assert not timer.time_out()
    assert_valid_tours(edges, stats)
    bnb_score = score_tour(stats[-1].tour, edges)

    assert bnb_score < greedy_score


# test_greedy()
# test_dfs()
test_branch_and_bound()