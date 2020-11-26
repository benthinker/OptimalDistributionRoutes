import numpy as np
import networkx as nx
from typing import *

from framework import *
from .mda_problem import *
from .cached_air_distance_calculator import CachedAirDistanceCalculator


__all__ = ['MDAMaxAirDistHeuristic', 'MDASumAirDistHeuristic',
           'MDAMSTAirDistHeuristic', 'MDATestsTravelDistToNearestLabHeuristic']


class MDAMaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-Max-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDAMaxAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This method calculated a lower bound of the distance of the remaining path of the ambulance,
         by calculating the maximum distance within the group of air distances between each two
         junctions in the remaining ambulance path. We don't consider laboratories here because we
         do not know what laboratories would be visited in an optimal solution.

        TODO [Ex.21]:
            Calculate the `total_distance_lower_bound` by taking the maximum over the group
                {airDistanceBetween(j1,j2) | j1,j2 in CertainJunctionsInRemainingAmbulancePath s.t. j1 != j2}
            Notice: The problem is accessible via the `self.problem` field.
            Use `self.cached_air_distance_calculator.get_air_distance_between_junctions()` for air
                distance calculations.
            Use python's built-in `max()` function. Note that `max()` can receive an *ITERATOR*
                and return the item with the maximum value within this iterator.
            That is, you can simply write something like this:
        >>> max(<some expression using item1 & item2>
        >>>     for item1 in some_items_collection
        >>>     for item2 in some_items_collection
        >>>     if <some condition over item1 & item2>)
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)
        if len(all_certain_junctions_in_remaining_ambulance_path) < 2:
            return 0

        total_distance_lower_bound = max([self.cached_air_distance_calculator.get_air_distance_between_junctions(j_1, j_2)
                                       for j_1 in all_certain_junctions_in_remaining_ambulance_path
                                       for j_2 in all_certain_junctions_in_remaining_ambulance_path
                                       if j_1 != j_2])

        return total_distance_lower_bound


class MDASumAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-Sum-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDASumAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic evaluates the distance of the remaining ambulance route in the following way:
        It builds a path that starts in the current ambulance's location, and each next junction in
         the path is the (air-distance) nearest junction (to the previous one in the path) among
         all certain junctions (in `all_certain_junctions_in_remaining_ambulance_path`) that haven't
         been visited yet.
        The remaining distance estimation is the cost of this built path.
        Note that we ignore here the problem constraints (like enforcing the #matoshim and free
         space in the ambulance's fridge). We only make sure to visit all certain junctions in
         `all_certain_junctions_in_remaining_ambulance_path`.
        TODO [Ex.24]:
            Complete the implementation of this method.
            Use `self.cached_air_distance_calculator.get_air_distance_between_junctions()` for air
             distance calculations.
            For determinism, while building the path, when searching for the next nearest junction,
             use the junction's index as a secondary grading factor. So that if there are 2 different
             junctions with the same distance to the last junction of the so-far-built path, the
             junction to be chosen is the one with the minimal index.
            You might want to use python's tuples comparing to that end.
             Example: (a1, a2) < (b1, b2) iff a1 < b1 or (a1 == b1 and a2 < b2).
        """

        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        # while len(all_certain_junctions_in_remaining_ambulance_path) != 0:
        #     indexArr = np.argsort([locationIndex.index for locationIndex in all_certain_junctions_in_remaining_ambulance_path])
        #     disArr = [self.cached_air_distance_calculator.get_air_distance_between_junctions(state.current_location, locationDist) for locationDist in all_certain_junctions_in_remaining_ambulance_path]
        #     disArr = disArr[indexArr]
        #
        #     bla = np.argmin(disArr)
        #     minimalIndex = indexArr[bla]
        #     pathLength += disArr[bla]
        #
        #     all_certain_junctions_in_remaining_ambulance_path.pop(minimalIndex)
        #

        # i assume the current node is in all_certain_junctions_in_remaining_ambulance_path(this is back by the pdf)
        all_certain_junctions_in_remaining_ambulance_path = \
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state)
        current_location = state.current_location
        currMinJunction = current_location
        pathLength = 0
        while len(all_certain_junctions_in_remaining_ambulance_path) != 0:
            currMinDist = pow(2, 31)#this should be fixed
            for location in all_certain_junctions_in_remaining_ambulance_path:
                distance = self.cached_air_distance_calculator.get_air_distance_between_junctions(current_location, location)
                if (currMinDist, currMinJunction.index) > (distance, location.index):
                    currMinDist = distance
                    currMinJunction = location
            # all_certain_junctions_in_remaining_ambulance_path.pop(currMinJunction.index)
            all_certain_junctions_in_remaining_ambulance_path.remove(currMinJunction)
            pathLength += currMinDist
            current_location = currMinJunction

        return pathLength
        #
        # raise NotImplementedError  # TODO: remove this line and complete the missing part here!


class MDAMSTAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-MST-AirDist'

    def __init__(self, problem: GraphProblem):
        super(MDAMSTAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.Distance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound for the distance of the remaining route of the ambulance.
        Here this remaining distance is bounded (from below) by the weight of the minimum-spanning-tree
         of the graph, in-which the vertices are the junctions in the remaining ambulance route, and the
         edges weights (edge between each junctions pair) are the air-distances between the junctions.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        return self._calculate_junctions_mst_weight_using_air_distance(
            self.problem.get_all_certain_junctions_in_remaining_ambulance_path(state))

    def _calculate_junctions_mst_weight_using_air_distance(self, junctions: List[Junction]) -> float:
        """
        TODO [Ex.27]: Implement this method.
              Use `networkx` (nx) package (already imported in this file) to calculate the weight
               of the minimum-spanning-tree of the graph in which the vertices are the given junctions
               and there is an edge between each pair of distinct junctions (no self-loops) for which
               the weight is the air distance between these junctions.
              Use the method `self.cached_air_distance_calculator.get_air_distance_between_junctions()`
               to calculate the air distance between the two junctions.
              Google for how to use `networkx` package for this purpose.
              Use `nx.minimum_spanning_tree()` to get an MST. Calculate the MST size using the method
              `.size(weight='weight')`. Do not manually sum the edges' weights.
        """
        graph = nx.Graph()
        edges_weighted = {(j_1,j_2,self.cached_air_distance_calculator.get_air_distance_between_junctions(j_1, j_2))
                          for j_1 in junctions
                          for j_2 in junctions
                          if j_1 != j_2}

        for edge in edges_weighted:
            graph.add_edge(edge[0], edge[1], weight=edge[2])

        mst = nx.tree.minimum_spanning_tree(graph)
        return mst.size(weight='weight')
        # raise NotImplementedError  # TODO: remove this line!


class MDATestsTravelDistToNearestLabHeuristic(HeuristicFunction):
    heuristic_name = 'MDA-TimeObjectiveSumOfMinAirDistFromLab'

    def __init__(self, problem: GraphProblem):
        super(MDATestsTravelDistToNearestLabHeuristic, self).__init__(problem)
        assert isinstance(self.problem, MDAProblem)
        assert self.problem.optimization_objective == MDAOptimizationObjective.TestsTravelDistance
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound to the remained tests-travel-distance of the remained ambulance path.
        The main observation is that driving from a laboratory to a reported-apartment does not increase the
         tests-travel-distance cost. So the best case (lowest cost) is when we go to the closest laboratory right
         after visiting any reported-apartment.
        If the ambulance currently stores tests, this total remained cost includes the #tests_on_ambulance times
         the distance from the current ambulance location to the closest lab.
        The rest part of the total remained cost includes the distance between each non-visited reported-apartment
         and the closest lab (to this apartment) times the roommates in this apartment (as we take tests for all
         roommates).
        TODO [Ex.33]:
            Complete the implementation of this method.
            Use `self.problem.get_reported_apartments_waiting_to_visit(state)`.
        """
        assert isinstance(self.problem, MDAProblem)
        assert isinstance(state, MDAState)

        def air_dist_to_closest_lab(junction: Junction) -> float:
            """
            Returns the distance between `junction` and the laboratory that is closest to `junction`.
            """
            return min([junction.calc_air_distance_from(laboratory.location) for laboratory
                        in self.problem.problem_input.laboratories])

        closest_laboratory_to_state = air_dist_to_closest_lab(state.current_location)
        tests_on_ambulance = state.get_total_nr_tests_taken_and_stored_on_ambulance()

        total_cost = (tests_on_ambulance * closest_laboratory_to_state)
        total_cost += sum([air_dist_to_closest_lab(apartment.location) * apartment.nr_roommates for apartment in
                            self.problem.get_reported_apartments_waiting_to_visit(state)])
        return total_cost

        #raise NotImplementedError  # TODO: remove this line!
