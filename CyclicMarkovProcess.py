# import numpy as np
#
#
# class Node:
#     def __init__(self, intensity, probability=1):
#         self.start_node = self
#         self.prev_node = self
#         self.intensity = intensity
#         self.next = None
#         self.index = 0
#         self.probability = probability
#         self.is_end = True
#
#     def set_start_node(self, start_node):
#         self.start_node = start_node
#         if self.next is None:
#             self.next = start_node
#
#     def set_next(self, elements):
#         self.is_end = False
#         self.next = elements
#         elements.set_start_node(self.start_node)
#         elements.prev_node = self
#         self.start_node.prev_node = elements
#
#
# class Branching(Node):
#     def __init__(self, current_node, nodes, intensity, probability):
#         super().__init__(intensity, probability)
#         for next_index, node in enumerate(nodes):
#             node.set_start_node(current_node.start_node)
#             node.prev_node = current_node
#             node.index =
#
# class CyclicBranchingGraph:
#     def __init__(self, first_node):
#         self.first_node = first_node
#
#     def get_intensity_matrix:




