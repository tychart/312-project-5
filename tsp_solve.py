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
    display_graph(edges)

    greedy_solver = Greedy_Solver(edges, timer)
    stats = greedy_solver.solve()
    pretty_print_solution_stats(stats)
 
    
    return stats



class Greedy_Solver:
    def __init__(self, edges: list[list[float]], timer: Timer):
        self.edges = edges
        self.timer = timer
        self.stats = []
        # self.BSSF = math.inf
        self.n_nodes_expanded = 0
        self.n_nodes_pruned = 0
        self.cut_tree = CutTree(len(edges))

    def solve(self) -> list[SolutionStats]:

        for nodeindex in range(len(self.edges)):
            if self.timer.time_out():
                return
            self.greedy_helper([], nodeindex)
        return self.stats

        
    def greedy_helper(self, visited: list[int], currnode: int):
        if self.timer.time_out():
            self.n_nodes_pruned += 1
            self.cut_tree.cut(visited)
            return False
        if min(self.edges[currnode]) == math.inf:
            self.n_nodes_pruned += 1
            self.cut_tree.cut(visited)
            return False
        
        self.n_nodes_expanded += 1
        visited.append(currnode)
        nextnode = find_min_unvisited(self.edges[currnode], visited)
        
        if nextnode is None:
            if len(visited) == len(self.edges) and self.edges[currnode][visited[0]] != math.inf:
                # Complete path was found
                
                is_better = False

                # Check if the path is better than the previous solution found
                # (As per the instructions, althrough I have no idea why)
                if len(self.stats) > 0:
                    is_better = score_tour(visited, self.edges) < self.stats[-1].score
                
                if is_better or len(self.stats) == 0:
                    self.stats.append(SolutionStats(
                        tour=visited,
                        score=score_tour(visited, self.edges),
                        time=self.timer.time(),
                        max_queue_size=1,
                        n_nodes_expanded=self.n_nodes_expanded,
                        n_nodes_pruned=self.n_nodes_pruned,
                        n_leaves_covered=self.cut_tree.n_leaves_cut(),
                        fraction_leaves_covered=self.cut_tree.fraction_leaves_covered()
                    ))
                    return True
            self.n_nodes_pruned += 1
            self.cut_tree.cut(visited)
            return False
        return self.greedy_helper(visited, nextnode)

def find_min_unvisited(edges_list: list[float], visited: list[int]) -> int | None:
    """
    Finds the minimum float in a list of edges that is not present in the
    already_visited list.

    Args:
        edges_list: A list of float values representing edges.
        visited: A list of float values that have already been visited.

    Returns:
        The minimum float from the 'edges' list that is not in 'already_visited',
        or None if all edges have been visited or the 'edges' list is empty.
    """
    min_unvisited = float('inf')
    min_index = -1
    found = False

    for index in range(len(edges_list)):
        if edges_list[index] < min_unvisited and index not in visited:
            min_unvisited = edges_list[index]
            min_index = index
            found = True

    if found:
        return min_index
    else:
        return None


class DFS_Solver:
    """
    Solves the TSP using a depth-first search approach.
    """

    def __init__(self, edges: list[list[float]], timer: Timer):
        self.edges = edges
        self.timer = timer
        self.stats = []
        self.BSSF = math.inf
        self.n_nodes_expanded = 0
        self.n_nodes_pruned = 0
        self.cut_tree = CutTree(len(edges))
        

    def dfs_solve(self) -> list[SolutionStats]:
        """
        Solves the TSP using depth-first search.

        Returns:
            A list of SolutionStats objects containing the results.
        """
        # Add the code for solving the TSP here

        currnode = 0 # Index of Starting Node
        visited = []

        self.dfs_recursive(currnode, visited.copy())

        if len(self.stats) == 0:
                return [SolutionStats(
                [],
                math.inf,
                self.timer.time(),
                1,
                self.n_nodes_expanded,
                self.n_nodes_pruned,
                self.cut_tree.n_leaves_cut(),
                self.cut_tree.fraction_leaves_covered()
            )]
        
        # return [self.stats[-1]]
        return [self.stats[-1]]
    
    def branch_and_bound_solve(self) -> list[SolutionStats]:
        """
        Solves the TSP using branch and bound.

        Returns:
            A list of SolutionStats objects containing the results.
        """

        greedy_solver = Greedy_Solver(self.edges, self.timer)
        greedy_stats = greedy_solver.solve()

        if len(greedy_stats) != 0:
            self.BSSF = greedy_stats[-1].score
        else:
            self.BSSF = math.inf
        self.BSSF = math.inf

        inital_reduced_cost_matrix, initial_rcm_initial_cost = self.generate_initial_reduced_cost_matrix()

        self.branch_and_bound_recursive(0, inital_reduced_cost_matrix, initial_rcm_initial_cost, [])

        if len(self.stats) == 0:
            if self.BSSF != math.inf:
                return [greedy_stats[-1]]
            return [SolutionStats(
                [],
                math.inf,
                self.timer.time(),
                1,
                self.n_nodes_expanded,
                self.n_nodes_pruned,
                self.cut_tree.n_leaves_cut(),
                self.cut_tree.fraction_leaves_covered()
            )]

        return self.stats



    def dfs_recursive(self, currnode: int, visited: list):

        visited.append(currnode)
        self.n_nodes_expanded += 1
        
        for edge_index in range(0, len(self.edges[currnode])):
            if self.edges[currnode][edge_index] == math.inf:
                # self.n_nodes_pruned += 1
                continue

            if edge_index == visited[0]: # Evaluating the path to the starting node
                if len(visited) == len(self.edges): # Complete path was found
                    # Check if the path is better than the current best
                    score = score_tour(visited, self.edges)
                    if score < self.BSSF:
                        self.BSSF = score

                        self.stats.append(SolutionStats(
                            tour=visited,
                            score=score,
                            time=self.timer.time(),
                            max_queue_size=1,
                            n_nodes_expanded=self.n_nodes_expanded,
                            n_nodes_pruned=self.n_nodes_pruned,
                            n_leaves_covered=self.cut_tree.n_leaves_cut(),
                            fraction_leaves_covered=self.cut_tree.fraction_leaves_covered()
                        ))
                        continue
                # self.n_nodes_pruned += 1
                continue
                
            if edge_index in visited:
                # self.n_nodes_pruned += 1
                continue

            self.dfs_recursive(edge_index, visited.copy())
    
    def branch_and_bound_recursive(self, 
                                currnode: int,
                                parent_rcm: list[list[float]],  
                                parent_lower_bound: float,          
                                visited: list[int]):

        visited.append(currnode)
        self.n_nodes_expanded += 1
        
        for edge_index in range(0, len(self.edges[currnode])):
            if self.timer.time_out():
                self.n_nodes_pruned += 1
                self.cut_tree.cut(visited)
                return False
            
            if self.edges[currnode][edge_index] == math.inf:
                self.n_nodes_pruned += 1
                self.cut_tree.cut(visited)
                continue

            if edge_index == visited[0]: # Evaluating the path to the starting node
                if len(visited) == len(self.edges): # Complete path was found
                    # Check if the path is better than the current best
                    score = score_tour(visited, self.edges)
                    if score < self.BSSF:
                        self.BSSF = score

                        print(f"Found a new best solution: {visited} with score {score}")

                        self.stats.append(SolutionStats(
                            tour=visited,
                            score=score,
                            time=self.timer.time(),
                            max_queue_size=1,
                            n_nodes_expanded=self.n_nodes_expanded,
                            n_nodes_pruned=self.n_nodes_pruned,
                            n_leaves_covered=self.cut_tree.n_leaves_cut(),
                            fraction_leaves_covered=self.cut_tree.fraction_leaves_covered()
                        ))
                        continue
                self.n_nodes_pruned += 1
                self.cut_tree.cut(visited)
                continue
                
            if edge_index in visited:
                self.n_nodes_pruned += 1
                self.cut_tree.cut(visited)
                continue



            curr_lower_bound, curr_rcm = self.calculate_reduced_cost_matrix(parent_rcm, parent_lower_bound, visited, edge_index)


            # if curr_lower_bound >= self.BSSF:
            if curr_lower_bound >= self.BSSF:
                self.n_nodes_pruned += 1
                self.cut_tree.cut(visited)
                continue

            self.branch_and_bound_recursive(edge_index, curr_rcm, curr_lower_bound, visited.copy())

    def generate_initial_reduced_cost_matrix(self) -> list[list[float]]:
        """
        Generates the initial reduced cost matrix from the edges.

        Returns:
            A list of lists representing the reduced cost matrix.
        """
        # starting_cost = 0
        n = len(self.edges)
        rcm = [[float(item) for item in inner_list] for inner_list in self.edges]

        # Set the diagonal to inf
        for i in range(n):
            rcm[i][i] = math.inf

        rcm, total_cost = self.reduce_cost_matrix(rcm)

        print("Initial Reduced Cost Matrix:")
        display_graph(rcm)
        print(f"Total Cost: {total_cost}")
        print()
        return rcm, total_cost

    def reduce_cost_matrix(self, matrix: list[list[float]]) -> list[list[float]]:

        n = len(self.edges)
        reduction_cost = 0

        # Make sure there is a 0 in every row
        for i in range(n):
            min_value = min(matrix[i])
            if min_value != 0 and min_value != math.inf:
                min_value = min_value
                reduction_cost += min_value
                for j in range(n):
                    if matrix[i][j] != math.inf:
                        matrix[i][j] -= min_value
        
        # Make sure there is a 0 in every column
        for j in range(n):
            min_value = min([row[j] for row in matrix])
            if min_value != 0 and min_value != math.inf:
                reduction_cost += min_value
                for i in range(n):
                    if matrix[i][j] != math.inf:
                        matrix[i][j] -= min_value
            
        return matrix, reduction_cost

    def calculate_reduced_cost_matrix(
            self, 
            parent_rcm: list[list[float]], 
            parent_lower_bound: float, 
            visited: list[int], 
            edge_index: int
        ) -> float:


        # exited_nodes = visited[:-1]  # Exclude current node
        # entered_nodes = visited[1:] + [edge_index] # Account for not entering the current node again

        curr_node = visited[-1]  # The last node in the visited list
        next_node = edge_index  # The node we are testing to enter

        # Make a copy of the parent matrix
        curr_rcm = copy_matrix(parent_rcm)

        
        self.nullify_row(curr_rcm, curr_node)  # Nullify the row of the current node

        self.nullify_column(curr_rcm, next_node)  # Nullify the column of the edge index

        # Nullify the column of the starting node to prevent premature loops
        # curr_rcm[next_node][visited[0]] = math.inf  

        curr_rcm[next_node][curr_node] = math.inf  # Nullify so it can't go back

        # # Infinity out the rows of the exited nodes
        # for i in range(len(rcm)):
        #     if i in exited_nodes:
        #         for j in range(len(rcm)):
        #             rcm[i][j] = math.inf
        
        # # Infinity out the columns of the entered nodes
        # for j in range(len(rcm)):
        #     if j in entered_nodes:
        #         for i in range(len(rcm)):
        #             rcm[i][j] = math.inf
        
        reduced_matrix, reduction_cost = self.reduce_cost_matrix(curr_rcm)

        # Add the cost of the reduced cost matrix + cost to parent + cost to current node
        # return self.rcm_initial_cost + reduction_cost + score_tour(visited, self.edges) + self.edges[visited[-1]][edge_index]
        edge_cost = parent_rcm[curr_node][next_node] # Gets the cost of the single, current, edge
        lower_bound = parent_lower_bound + edge_cost + reduction_cost

        # (Optional) Print debugging information if needed:
        print(f"Transition: {visited} -> {edge_index}")
        print(f"Original edge cost: {edge_cost}")
        print(f"Parent LB: {parent_lower_bound}")
        print(f"Reduction cost incurred: {reduction_cost}")
        print(f"New lower bound: {lower_bound}")
        print("New Reduced Matrix:")
        display_graph(reduced_matrix)
        print()

        

        return lower_bound, reduced_matrix
    
    def nullify_row(self, matrix: list[list[float]], row: int):
        for j in range(len(matrix)):
            matrix[row][j] = math.inf

    def nullify_column(self, matrix: list[list[float]], col: int):
        for i in range(len(matrix)):
            matrix[i][col] = math.inf

def dfs(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    dfs_solver = DFS_Solver(edges, timer)
    display_graph(edges)

    stats = dfs_solver.dfs_solve()

    pretty_print_solution_stats(stats)

    # return [stats[-1]]
    return stats

    # return dfs_solver.solve(edges)


def branch_and_bound(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    # set the diagonal to inf before reducing matrices
    # for i in range(len(edges)): edges[i][i] = inf  

    branch_and_bound_solver = DFS_Solver(edges, timer)
    display_graph(edges)
    stats = branch_and_bound_solver.branch_and_bound_solve()
    pretty_print_solution_stats(stats)

    return stats

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







def copy_matrix(matrix: list[list[float]]) -> list[list[float]]:
    """
    Creates a deep copy of a 2D list (matrix).

    Args:
        matrix: A 2D list representing the matrix to be copied.

    Returns:
        A deep copy of the input matrix.
    """
    return [row[:] for row in matrix]









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

def pretty_print_solution_stats(solution_stats_list: list[SolutionStats]):
    """
    Pretty prints a list of SolutionStats objects.

    Args:
        solution_stats_list: A list of SolutionStats namedtuples.
    """
    if not solution_stats_list:
        print("No solution statistics to display.")
        return

    for i, stats in enumerate(solution_stats_list):
        print(f"--- Solution {i + 1} ---")
        print(f"  Tour: {stats.tour}")
        print(f"  Score: {stats.score:.4f}")  # Format score to 4 decimal places
        print(f"  Time: {stats.time:.6f} seconds") # Format time to 6 decimal places
        print(f"  Max Queue Size: {stats.max_queue_size}")
        print(f"  Nodes Expanded: {stats.n_nodes_expanded}")
        print(f"  Nodes Pruned: {stats.n_nodes_pruned}")
        print(f"  Leaves Covered: {stats.n_leaves_covered}")
        print(f"  Fraction Leaves Covered: {stats.fraction_leaves_covered:.4f}") # Format fraction
        print()


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