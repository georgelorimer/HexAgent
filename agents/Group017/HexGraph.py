import copy

from math import inf

class GraphNode():

    def __init__(self, id: int, colour: str, x: int, y: int):
        self.id = id
        self.colour = colour
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return str(self.x) + " " + str(self.y) + ":" + self.colour


class GraphEdge():

    def __init__(self, node_u: GraphNode, node_v: GraphNode):
        self.node_u = node_u
        self.node_v = node_v


class HGraph():

    def __init__(self, board_size=11, nodes=[], edges=[], components=[]):
        self.size = len(nodes)
        self.nodes = nodes
        self.edges = edges
        self.components = components
        self.board_size = board_size

        self.edges_r = 0
        self.edges_b = 0
        self.nodes_r = 0
        self.nodes_b = 0

        if self.size == 0:
            self.add_borders()

    def add_borders(self):
        # red 0, 1
        # red top to bottom
        red_a = GraphNode(0, "R.", -20, -20)
        red_b = GraphNode(1, "R.", 20, 20)

        # blue 2, 3
        # blue left to right
        blue_a = GraphNode(2, "B.", -30, -30)
        blue_b = GraphNode(3, "B.", 30, 30)

        self.nodes.extend([red_a, red_b, blue_a, blue_b])

        red_top = []
        red_bottom = []
        blue_left = []
        blue_right = []

        for i in range(0, self.board_size):
            # red top border
            node = GraphNode(4 + i, "R.", -1, i)
            red_top.append(node)
            self.edges.append(GraphEdge(node, red_a))

            # red bottom border
            node = GraphNode(self.board_size + 4 + i, "R.", self.board_size, i)
            red_bottom.append(node)
            self.edges.append(GraphEdge(node, red_b))
            
            # blue left border
            node = GraphNode(self.board_size*2 + 4 + i, "B.", i, -1)
            blue_left.append(node)
            self.edges.append(GraphEdge(node, blue_a))

            # blue right border
            node = GraphNode(self.board_size*3 + 4 + i, "B.", i, self.board_size)
            blue_right.append(node)
            self.edges.append(GraphEdge(node, blue_b))

        self.nodes.extend(red_top)
        self.nodes.extend(red_bottom)
        self.nodes.extend(blue_left)
        self.nodes.extend(blue_right)

        self.size = len(self.nodes)
        self.update_components(self.nodes, self.edges)
        

    def add_node(self, colour: str, x: int, y: int) -> None:

        # create a new node object with the position and colour passed as parameters
        node = GraphNode(self.size, colour, x, y)
        self.size += 1

        # append this node to the list of nodes
        self.nodes.append(node)

        # blank list to store new edges
        new_edges = []

        for neighbour in self.neighbours(x, y):
            # fetch the node at the neighbour position
            neighbour_node = self.get_node(neighbour[0], neighbour[1])
            
            if not neighbour_node:
                continue

            if neighbour_node.colour[0] != colour:
                continue

            # create an edge between the current neighbour and the node from the
            # main body of the function
            edge = GraphEdge(node, neighbour_node)

            new_edges.append(edge)

        self.edges.extend(new_edges)
        if colour == "R":
        	self.edges_r += len(new_edges)
        	self.nodes_r += 1
        else:
        	self.edges_b += len(new_edges)
        	self.nodes_b += 1

        self.update_components([node], new_edges)

    def neighbours(self, x: int, y: int) -> list:
        """
        Returns the list of neighbour squares on the board
        """

        # empty list to store positions
        neighbour_list = []

        # list of x and y offsets of the neighbours on the hex board
        neighbour_pos = [[-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1], [0, -1]]

        # iterate over offsets
        for pos in neighbour_pos:

            # calculate offsets
            diff_x = x + pos[0]
            diff_y = y + pos[1]

            neighbour_list.append([diff_x, diff_y])

        return neighbour_list


    def evaluate(self) -> int:
        # check win for red
        if Component.find(self.components, self.nodes[0]) == Component.find(self.components, self.nodes[1]):
        	return 3000

        # check win for blue
        if Component.find(self.components, self.nodes[2]) == Component.find(self.components, self.nodes[3]):
        	return -3000

        component_sizes = self.component_size()

        r_score = 1
        b_score = -1

        for comp in component_sizes[0]:
            r_score += comp

        for comp in component_sizes[1]:
            b_score -= comp

        r_score /= len(component_sizes[0])
        b_score /= len(component_sizes[1])

        if self.edges_r != 0 and self.nodes_r != 0:
       	    r_score /= (self.edges_r / self.nodes_r)
       	if self.edges_b != 0 and self.nodes_b != 0:
            b_score /= (self.edges_b / self.nodes_b)

        return int (r_score + b_score)

    def get_node(self, x: int, y: int) -> GraphNode:
        """
        Returns the node at a given x and y
        """
        for node in self.nodes:
            if node.x == x and node.y == y:
                return node

        return None

    def update_components(self, nodes: list, edges: list):
        # put the new node into a component containing just itsself
        for node in nodes:
            self.components.append(Component(node, 0))

        # iterate over the new edges
        for edge in edges:

            # if the nodes joined by the edge are not in the same component
            if Component.find(self.components, edge.node_u) != Component.find(self.components, edge.node_v):

                # connect the components containing the edges together
                Component.union(self.components, Component.find(self.components, edge.node_u), Component.find(self.components, edge.node_v))

    def component_size(self):
        # list holding two blank lists to store the connecte components for Red and Blue
        # players respectively
        component_sizes = [[], []]
        colour_dict = {"R": 0, "B": 1}

        # iterate over nodes
        for node in self.nodes:

            # if the parent of the current node in the component is itself
            if self.components[node.id].parent == node:

                # append its size to the list of sizes
                component_sizes[colour_dict[node.colour[0]]].append(self.components[node.id].size)

        return component_sizes

    def get_nodes(self, colour: str) -> list:

        nodes = []
        for node in self.nodes:
            if node.colour == colour and node.id > 47:
                nodes.append([node.x, node.y])

        return nodes

    def occupied_walls(self, colour: str, top_left: bool) -> list:
        wall_id = 0
        if colour == "B":
            wall_id = 2

        if not top_left:
            wall_id += 1

        parent = Component.find(self.components, self.nodes[wall_id])
        size = self.components[parent.id].size
        return size > 12

    def gcopy(self):
        """
        A function to return a copy of the current HexGraph
        """

        # empty lists to store nodes and edges
        nodes = []
        edges = []
        components = []

        # iterate over nodes and edges in the current object and add them
        # to the new lists
        for node in self.nodes:
            nodes.append(node)

        for edge in self.edges:
            edges.append(edge)

        # it is sufficien
        for component in self.components:
            components.append(copy.copy(component))

        # return a new HexGraph object with the new lists
        return HGraph(self.board_size, nodes, edges, components)

class Component:
    """
    A class which represents a connected component the graph, and contains
    functions for finding the connected components
    """

    def __init__(self, parent: GraphNode, rank: int):
        self.parent = parent
        self.rank = rank
        self.size = 1

    @classmethod
    def find(cls, components: list, node: GraphNode):
        """
        find function for union find operation, returns the parent node of the subset

        components: a list of Component objects
        node: the node whose parent is to be found

        """
        if components[node.id].parent != node:
            components[node.id].parent = cls.find(components, components[node.id].parent)
        return components[node.id].parent

    @classmethod
    def union(cls, components: list, node_u: GraphNode, node_v: GraphNode):
        """
        union function for union find operation, performs union by rank for efficiency
        """

        # compare ranks of components containing nodes u and v
        # if u has higher rank than v:
        if components[node_u.id].rank > components[node_v.id].rank:
            
            # get the size of v
            v_size = components[node_v.id].size

            # set the parent of node v to node u
            components[node_v.id].parent = node_u

            # update the size of the tree rooted at node u
            components[node_u.id].size += v_size

        # if v has higher rank than u:
        elif components[node_v.id].rank > components[node_u.id].rank:

            # get the size of u
            u_size = components[node_u.id].size

            # set the parent of node u to node v
            components[node_u.id].parent = node_v

            # update the size of the tree rooted at node v
            components[node_v.id].size += u_size

        # if the v and u are the same
        else:

            # get the size of v
            v_size = components[node_v.id].size

            # update the parent of one node to the other
            components[node_v.id].parent = node_u

            # update the size of the tree rooted at node u
            components[node_u.id].size += v_size

            # increase the rank of the new parent
            components[node_u.id].rank += 1

