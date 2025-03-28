import math
import random

from tsp_core import Tour, SolutionStats, Timer, score_tour, Solver
from tsp_cuttree import CutTree
from math import inf

def random_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))

    while True:
        if timer.time_out():
            return stats

        tour = random.sample(list(range(len(edges))), len(edges))
        n_nodes_expanded += 1

        cost = score_tour(tour, edges)
        if math.isinf(cost):
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        if stats and cost > stats[-1].score:
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        stats.append(SolutionStats(
            tour=tour,
            score=cost,
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        ))

    if not stats:
        return [SolutionStats(
            [],
            math.inf,
            timer.time(),
            1,
            n_nodes_expanded,
            n_nodes_pruned,
            cut_tree.n_leaves_cut(),
            cut_tree.fraction_leaves_covered()
        )]


def greedy_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    return []


class DFS_Solver:
    """
    Solves the TSP using a depth-first search approach.
    """

    def __init__(self, timer: Timer):
        """
        Initializes the object with the given attributes.

        Args:
            attribute1: Description of attribute1.
            attribute2: Description of attribute2.
        """
        self.timer = timer
        self.stats = []
        self.BSSF = math.inf

    def solve(self, edges: list[list[float]]) -> list[SolutionStats]:
        """
        Solves the TSP using depth-first search.

        Returns:
            A list of SolutionStats objects containing the results.
        """
        # Add the code for solving the TSP here

        currnode = 0 # Index of Starting Node
        stack = []
        stack.append(currnode) # Starting Node
        cost = 0
        visited = {}
        visited.add(currnode) # Starting Node

        self.dfs_recursive(edges, currnode, visited, stack, cost)

        pass

    def dfs_recursive(self, edges: list[list[float]], currnode: int, visited: set, stack: list, cost: float):
        """
        A brief description of what this method does.

        Args:
            parameter1: Description of parameter1.
        """
        
        for edge_index in range(0, len(edges[currnode])):
            if edges[currnode][edge_index] == math.inf:
                continue

            if edges[currnode][edge_index] == 0:
                if len(stack) == len(edges): # Complete path was found
                    # Check if the path is better than the current best
                    if cost < self.BSSF:
                        self.BSSF = cost

                        path = stack.copy()

                        self.stats.append(SolutionStats(
                            tour=path,
                            score=cost,
                            time=self.timer.time(),
                            max_queue_size=1,
                            n_nodes_expanded=n_nodes_expanded,
                            n_nodes_pruned=n_nodes_pruned,
                            n_leaves_covered=cut_tree.n_leaves_cut(),
                            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
                        ))
                continue
                
            if edge_index in visited:
                continue
            self.dfs_recursive(edges, edge_index, visited, stack, cost + edges[currnode][edge_index])
            

        pass  # 'pass' is a placeholder, replace it with your code

    # Add more methods here if needed


def dfs(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    dfs_solver = DFS_Solver(timer)
    display_graph(edges)
    return []
    # return dfs_solver.solve(edges)


def branch_and_bound(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    # set the diagonal to inf before reducing matrices
    for i in range(len(edges)): edges[i][i] = inf  
    return []

"""
For branch and bound lower bound, usining the reduced cost matrix:
    If entering node F, then infinity out all visited nodes rows and coloms
    Because entering node F, we can not leave it again, so we need to infinity out the colum of F
    Calculate the reduced cost matrix
    Add the cost of the reduced cost matrix + current cost (cost to F's parent) + cost to F
    If the reduced cost matrix is less than the current best, then we can continue
    If the reduced cost matrix is equal to or greater than the current best, then we can prune



    Actually, another method that is probebly better is to just skip the entered and exited nodes 
        (instead of calling them visited, split it out into 2 groups for the edge case of the start node and current edge)
    Make a reduced cost matrix at the begining with all the nodes
    Then, when entering a node, add it to the entered nodes set
    when computing the cost matrix, make a copy of the origional one, but only have the 
    nested for loop including nodes that have not been entered or exited:

    for i in range(len(edges)):
        if i not in entered_nodes:
            for j in range(len(edges)):
                if j not in entered_nodes:
                    append the value to the new matrix
                    pass
                    
    After that, compute the reduced cost matrix
    then add together reduced cost matrix + cost to parent + cost to current node to = lower bound
    If the lower bound is less than the bssf, then we can continue
"""


def branch_and_bound_smart(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    return []

















def display_graph(edges: list[list[float]]):
    """
    Prints a nicely formatted representation of the TSP graph (adjacency matrix)
    to the terminal.

    Args:
        edges: A list of lists representing the cost matrix of the graph.
               Inner lists represent the costs from one node to all others.
               math.inf indicates no direct connection.
    """
    num_nodes = len(edges)
    if num_nodes == 0:
        print("Graph is empty.")
        return

    # Determine the maximum width needed for cell values (including "inf")
    max_width = 3  # For "inf"
    for row in edges:
        for cost in row:
            if cost != math.inf:
                max_width = max(max_width, len(f"{cost:.1f}")) # Adjust precision as needed

    # Print header row (node indices)
    header = " " * (max_width + 1)  # Space for the row index
    for i in range(num_nodes):
        header += f"{i:<{max_width + 1}}"
    print(header)
    print("-" * (len(header)))

    # Print each row of the adjacency matrix
    for i in range(num_nodes):
        row_str = f"{i:<{max_width}}|"
        for j in range(num_nodes):
            cost = edges[i][j]
            if cost == math.inf:
                row_str += f"{'inf':<{max_width + 1}}"
            else:
                row_str += f"{cost:<{max_width + 1}.1f}" # Adjust precision as needed
        print(row_str)

if __name__ == '__main__':
    # Example usage with a sample edges list
    sample_edges = [
        [0.0, 10.0, 15.0, math.inf],
        [10.0, 0.0, 35.0, 25.0],
        [15.0, 35.0, 0.0, 30.0],
        [math.inf, 25.0, 30.0, 0.0]
    ]
    print("Sample TSP Graph:")
    display_graph(sample_edges)

    print("\nAnother Example:")
    another_graph = [
        [0, 5, math.inf, 10],
        [5, 0, 8, math.inf],
        [math.inf, 8, 0, 3],
        [10, math.inf, 3, 0]
    ]
    display_graph(another_graph)

    print("\nEmpty Graph:")
    display_graph([])