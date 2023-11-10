import socket
import math
from copy import deepcopy, copy
from random import choice, randint

from HexGraph import HGraph, GraphNode

class BoardState():
    """
    A class to hold all the data relating to a specific state in the game, to
    improve code readability
    """

    def __init__(self) -> None:
        self.turn_count = 0

        self.board = []

        self.vulnerable = {"R":[], "B":[]}

        # stores pairs of positions that are vulnerable because they are inbetween a bridge
        self.prev_move = []

        self.graph = HGraph()

    def state_copy(self):
        new = BoardState()
        new.turn_count = self.turn_count

        new.board = deepcopy(self.board)
        new.vulnerable = deepcopy(self.vulnerable)

        new.prev_move = copy(self.prev_move)

        new.graph = self.graph.gcopy()

        return new


class Agent17():
    """
    This class describes Group 17's agent.

    R = MAX, B = MIN
    """

    HOST = "127.0.0.1"
    PORT = 1234

    def __init__(self, board_size=11) -> None:
        """Class initilization function

        Args:
            board_size (int): the size of the board, default is 11
        """
        self.s = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )

        self.s.connect((self.HOST, self.PORT))

        self.board_size = board_size
        self.swapped = False
        self.colour = ""
        
        self.state = BoardState()
       

    def run(self) -> None:
        """
        Reads data until it receives an END message or the socket closes.
        """
        while True:
            data = self.s.recv(1024)
            if not data:
                break

            # print(f"{self.colour} {data.decode('utf-8')}", end = "")
            if (self.interpret_data(data)):
                break

        # print(f"Agent17 {self.colour} terminated")

    def interpret_data(self, data) -> bool:
        """Checks the type of message and responds accordingly.

        Args:
            data (bytes): message from engine

        Returns:
            True: when the game ends
            False: otherwise
        """
        messages = data.decode("utf-8").strip().split("\n")
        messages = [x.split(";") for x in messages]

        for s in messages:
            # start message: e.g. [['START', '11', 'R']]
            print(s)

            if s[0] == "START":
                self.board_size = int(s[1])
                self.colour = s[2]
                self.state.board = [
                    [0] * self.board_size for i in range(self.board_size)
                ]

                if self.colour == "R":
                    self.make_move()

            elif s[0] == "END":
                return True

            elif s[0] == "CHANGE":

                if s[3] == "END":
                    return True

                elif s[1] == "SWAP":
                    self.colour = self.opp_colour()
                    self.swapped = True
                    if s[3] == self.colour:
                        self.make_move()

                elif s[3] == self.colour:
                    action = [int(x) for x in s[1].split(",")]

                    # add opponent's move to the board
                    self.move_on_state(self.state, action, self.opp_colour())

                    self.make_move()

        return False

    def available_moves(self, state: BoardState, board: list, colour: str) -> list:
        """
        Gathers the available spaces in the board passed as an argument, as
        opposed to the current board stored in 'self', this is so minimax can work properly.

        params: board list of integers
        Returns:
            2d list: each member is a board position that is empty
        """

        # check if a bridge has been attacked
        attack = self.check_for_bridge_attack(state, colour)
        if attack != None and state.board[attack[0]][attack[1]] == 0:
            return [attack]

        TOP_AMT = 5
        choices = ["P"] * TOP_AMT

        # picks the TOP_AMT best choices on the board
        for i in range(self.board_size):
            for j in range(self.board_size):
            
                #if randint(0, 2) == 1:
                    #continue

                if type(board[i][j]) != int:
                    continue

                for k in range(TOP_AMT):
                    if choices[k] == "P":
                        choices[k] = [i, j]
                        break

                    if colour == "R":
                        if board[i][j] > board[choices[k][0]][choices[k][1]]:
                            for l in range(TOP_AMT - 1, k, -1):
                                # shift along
                                choices[l] = choices[l - 1]
                            choices[k] = [i, j]
                            break

                    if colour == "B":
                        if board[i][j] < board[choices[k][0]][choices[k][1]]:
                            for l in range(TOP_AMT - 1, k, -1):
                                choices[l] = choices[l - 1]
                            choices[k] = [i, j]
                            break

        # remove any remaining placeholders
        while choices.count("P") != 0:
            choices.remove("P")

        return choices

    def minimax(self, node: BoardState, alpha: int, beta: int, depth: int, player: str) -> int:
        """Minimax with alpha-beta pruning to get the best move possible.

        Args:
            alpha (int): alpha passed from parent node
            beta (int): beta passed from parent node
            node (BoardState): the current state of the board
            depth (int): the search depth of the algorithm
            player (char): the player currently looking to minimise or maximise

        Returns:
            int: either maximum evaluation or minimum evaluation depending on player
        """

        ### R = MAX, B = MIN

        # base case (leaf/terminal node) pass the static evaluation back up the tree
        if (depth == 0):
            return self.leaf_evaluation(node)

        # maximising node
        if player == "R":
            # value of current node
            value = -math.inf

            # iterate over possible moves from current board position
            evaluated_node = self.perimeter_evaluation(node) # will chose moves appropriately
            for move in self.available_moves(node, evaluated_node, player):

                # get a new copy of the board and make the selected move
                daughter_node = node.state_copy()
                self.move_on_state(daughter_node, move, player)

                # evaluate children of this node
                value = max(value, self.minimax(daughter_node, alpha, beta, depth-1, "R"))
                alpha = max(alpha, value)

                # alpha/beta cutoff
                if value >= beta:
                    break

            return value

        # minimising node
        else:
            value = math.inf

            evaluated_node = self.perimeter_evaluation(node) # will chose moves appropriately

            for move in self.available_moves(node, evaluated_node, player):

                daughter_node = node.state_copy()
                self.move_on_state(daughter_node, move, player)

                value = min(value, self.minimax(daughter_node, alpha, beta, depth-1, "B"))
                beta = min(beta, value)

                if value <= alpha:
                    break

            return value

    def minimax_move(self) -> list:
        """
        Driver function for minimax that returns the best selected move as list [x, y]
        """

        # set best score based on whether we are minimising (B) or maximising (R) in
        # this match
        best = -math.inf  # R
        if self.colour == 'B':
            best = math.inf

        # initialise a variable to hold the index of the best move
        best_index = -1

        # get available moves
        perim_eval = self.perimeter_evaluation(self.state)
        moves = self.available_moves(self.state, perim_eval, self.colour)

        for i in range(self.board_size):
            for j in range(i):
                print("  ", end=" ")
            print(perim_eval[i])

        # iterate over moves
        for index, move in enumerate(moves):

            # duplicate board and 'execute' move on it
            daughter_node = self.state.state_copy()
            self.move_on_state(daughter_node, move, self.colour)

            # call minimax on current moves
            evaluation = self.minimax(daughter_node, -math.inf, math.inf, 3, self.colour)
            
            # determine if the current move was better than the last (or initial) move
            # based on whether the player is maximising or minimising in the match
            if self.colour == "R":
                if (evaluation > best):
                    best = evaluation
                    best_index = index
            else:
                if (evaluation < best):
                    best = evaluation
                    best_index = index

        # return the move with the best score
        return moves[best_index]

    def make_move(self):
        """
        Function to select and play a move
        """

        # if we are blue choose whether or not to swap
        if self.colour == "B" and self.state.turn_count == 0:
            # swap
            if self.swap_heuristic(self.state.board):
                self.swapped = True  # We swapped so changed member variable
                self.s.sendall(bytes("SWAP\n", "utf-8"))
                self.state.turn_count += 1
                return
            else:
                # Here we are making the second move as we did not swap and we know they are not anywhere near
                # the centre so we can take the centre
                move = [5, 5]
        # This is when we start as red, make the move at [8,2] and get that move swapped
        # so we can prepare for this and go to the centre
        elif self.swapped and self.colour == "B" and self.state.turn_count == 1:
            move = [5, 5]
        # If we are red we can make the first move so we take the spot [8,2]
        # as it is further from the centre so less likely to be swapped
        elif self.colour == "R" and self.state.turn_count == 0:  # Changed to elif...
            move = [8, 2]
        # This is the third move in the game where we have a piece at [8,2] and we do not know where their piece is.
        # potentials: 6,3 or 7,4
        elif self.colour == "R" and self.state.turn_count == 1 and not self.swapped:
            if self.state.board[6][3] == 0:
                move = [6, 3]
            elif self.state.board[7][4] == 0:
                move = [7, 4]
        # Anything other move then just use minimax function
        else:
            move = self.minimax_move()

        self.move_on_state(self.state, move, self.colour)
        self.s.sendall(bytes(f"{move[0]},{move[1]}\n", "utf-8"))

    #### UTILITY FUNCTIONS ####

    def move_on_state(self, state: BoardState, move: list, colour: str) -> None:
        state.board[move[0]][move[1]] = colour
        state.graph.add_node(colour, move[0], move[1])
        self.update_vulnerable_pos(state, move, colour)
        state.prev_move = move
        if colour == self.colour:
            state.turn_count += 1

    def neighbours(self, position: list, board: list) -> list:
        """
        Returns the list of empty neighbour squares on the board given a position
        """

        # empty list to store positions
        neighbour_list = []

        # list of x and y offsets of the neighbours on the hex board
        neighbour_pos = [[-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1], [0, -1]]

        # iterate over offsets
        for pos in neighbour_pos:

            # calculate offsets
            diff_x = position[0] + pos[0]
            diff_y = position[1] + pos[1]

            # check boundary conditions and coninue to next board position if the boundary
            # have been exceeded
            if diff_x < 0 or diff_y < 0 or diff_x >= self.board_size or diff_y >= self.board_size:
                continue

            # check if the position is empty on the board
            if board[diff_x][diff_y] != 0:
                continue

            neighbour_list.append([diff_x, diff_y])

        return neighbour_list

    def opp_colour(self):
        """
        Returns the char representation of the colour opposite to the current one.
        """
        if self.colour == "R":
            return "B"
        elif self.colour == "B":
            return "R"
        else:
            return "None"

    def rev_colour(self, colour: str) -> str:
        if colour == "R":
            return "B"
        else:
            return "R"

    def duplicate_board(self, board: list) -> list:
        """
        A function which returns a copy of the board argument passed to it
        """

        # create empty list to store the new board
  #      board_copy = []

        # iterate over orginal board and copy it's cells to the new list
   #     for i in range(self.board_size):
    #        row = []
     #       for j in range(self.board_size):
      #          row.append(board[i][j])
       #     board_copy.append(row)

        return deepcopy(board)

    #### HEURISTICS ####

    def leaf_evaluation(self, state: BoardState):
        """
        Combines all heuristics into one number
        """

        # less good towards end of game

        p_score = self.perim_score(state)

        # chains = better towards end of game
        chain_score = 10 * state.graph.evaluate()

        # diff score
        diff_score = 200 * self.diff_score(state, self.colour)

        evaluation = p_score + chain_score + diff_score

        return evaluation

    def diff_score(self, state: BoardState, colour: str) -> int:
        # red
        # smallest, largest y, smallest largest x
        diffs = [[100, -100], [100, -100]]
        for node in state.graph.get_nodes(colour):
            diffs[0][0] = min(diffs[0][0], node[0])
            diffs[0][1] = max(diffs[0][1], node[0])

            diffs[1][0] = min(diffs[1][0], node[1])
            diffs[1][1] = max(diffs[1][1], node[1])

        y_diff = abs(diffs[0][0] - diffs[0][1])
        x_diff = abs(diffs[1][0] - diffs[1][1])

        # red favours high y diff, low x_diff and vice versa
        if colour == "R":
            return y_diff - x_diff
        else:
            return x_diff - y_diff


    def perim_score(self, state: BoardState) -> int:
        perim_eval = self.perimeter_evaluation(state)

        score = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (type(perim_eval[i][j]) == int):
                    score += perim_eval[i][j]

        return score


    def perimeter_evaluation(self, state: BoardState):
        """
        A function which returns an integer evaluation of the state passed into it
        """

        # constants for easy adjusting of weights
        EDGE_WEIGHT = 30
        NEIGHBOUR_WEIGHT = 5
        BRIDGE_WEIGHT = 50

        if self.bridge_heuristic(state, self.colour) > 10:
            BRIDGE_WEIGHT = 1
            NEIGHBOUR_WEIGHT = 3

        board = self.duplicate_board(state.board)

        # add weights to the board edges, excluding corners when turn count is small
        for i in range(3, self.board_size - 3):
            # blue sides
            if type(board[i][0]) == int:
                if not state.graph.occupied_walls("B", True):
                    board[i][0] -= EDGE_WEIGHT
            if type(board[i][self.board_size - 1]) == int:
                if not state.graph.occupied_walls("B", False):
                    board[i][self.board_size - 1] -= EDGE_WEIGHT
            if type(board[0][i]) == int:
                if not state.graph.occupied_walls("R", True):
                    board[0][i] += EDGE_WEIGHT
            if type(board[self.board_size - 1][i]) == int:
                if not state.graph.occupied_walls("R", False):
                    board[self.board_size - 1][i] += EDGE_WEIGHT

        # get positions of pieces currently in the graph
        red = state.graph.get_nodes("R")
        blue = state.graph.get_nodes("B")

        # iterate over red's positions
        for pos in red:
            # increment neighbour positions by 10
            for n in self.neighbours(pos, state.board):
                board[n[0]][n[1]] = NEIGHBOUR_WEIGHT

            # increment bridge positions by 15
            for bridge in self.potential_db(pos, state.board):
                if (bridge[0] > 3 and bridge[0] < 8 and bridge[1] > 1 and bridge[1] < 9) and state.turn_count < 10:
                    board[bridge[0]][bridge[1]] = BRIDGE_WEIGHT * 2
                else:
                    board[bridge[0]][bridge[1]] = BRIDGE_WEIGHT

        # iterate over blues's positions
        for pos in blue:
            # decrement neighbour positions by 10
            for n in self.neighbours(pos, state.board):
                board[n[0]][n[1]] = -NEIGHBOUR_WEIGHT

            # decrement bridge positions by 15
            for bridge in self.potential_db(pos, state.board):
                if (bridge[0] > 3 and bridge[0] < 8 and bridge[1] > 1 and bridge[1] < 9) and state.turn_count < 10:
                    board[bridge[0]][bridge[1]] = BRIDGE_WEIGHT * -2
                else:
                    board[bridge[0]][bridge[1]] = BRIDGE_WEIGHT

        return board

    def bridge_heuristic(self, state, colour) -> float:
        """
        A function which calculates the number of double bridges the current player has
        """

        return self.count_vertical_bridges(state, colour) + self.count_diag_up_bridges(state, colour) + self.count_diag_down_bridges(state, colour)

    def count_vertical_bridges(self, state, colour) -> int:
        """
        Returns the 'double bridges' that run vertically through the board
        0 X 0
         0 0 0
          X 0 0
        """
        total = 0

        for pos in state.graph.get_nodes(colour):
            b = state.graph.get_node(pos[0] + 2, pos[1] - 1)
            if b:
                if b.colour == colour:
                    total += 1

        return total

    def count_diag_down_bridges(self, state: BoardState, colour) -> int:
        """
        Returns the 'double bridges' diagonally down through the board
        X 0 0
         0 X 0
        """
        total = 0

        for pos in state.graph.get_nodes(colour):
            b = state.graph.get_node(pos[0] + 1, pos[1] + 1)
            if b:
                if b.colour == colour:
                    total += 1

        # iterate from row 0 to board size - 1
       # for i in range(0, self.board_size - 1):#

            # iterate from square 0 to board size:
            #for j in range(0, self.board_size - 1):

                # check for vertical double bridge and increment counter
                #if board[i][j] == colour and board[i + 1][j + 1] == colour:
                 #   total = total + 1

        return total

    def count_diag_up_bridges(self, state, colour) -> int:
        """
        Returns the 'double bridges' that run diagonally up through the board

        0 0 X
         X 0 0
        """
        total = 0

        for pos in state.graph.get_nodes(colour):
            b = state.graph.get_node(pos[0] - 1, pos[1] + 2)
            if b:
                if b.colour == colour:
                    total += 1

        return total

    def swap_heuristic(self, board: list) -> bool:
        """
        A function to evaluate whether or not we should swap board positions

        Args:
            board (2d list): the current board state

        Returns:
            True for swap or False otherwise
        """
        swap_evaluation_board = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == "R" and swap_evaluation_board[i][j] == 1:
                    return True

        return False

    #### MOVE MAKING ####

    def update_vulnerable_pos(self, state: BoardState, move: list, colour: str):
        """
        checks for our vulnerable positions::: OUR UPDATE
        Params:
            state: the current board state
            move: the move that we would like to check for
            colour: the colour of the player looking to protect their hexes

        Move just made comes in (form [y,x])
        Checks if that move creates a double bridge
        """

        db_pos = [[[-2,1],[-1,0],[-1,1]], [[-1,2],[-1,1],[0,1]], [[1,1],[0,1],[1,0]], [[2,-1],[1,0],[1,-1]], [[1,-2],[1,-1],[0,-1]],[[-1,-1],[0,-1],[-1,0]]]

        for index, bridge in enumerate(db_pos):

            # calculate offsets
            diff_y = move[0] + bridge[0][0]
            diff_x = move[1] + bridge[0][1]

            # check boundary conditions and coninue to next board position if the boundary
            # have been exceeded
            if diff_x < 0 or diff_y < 0 or diff_x >= self.board_size or diff_y >= self.board_size:
                continue

            open1x = move[1] + bridge[1][1]
            open1y = move[0] + bridge[1][0]

            open2x = move[1] + bridge[2][1]
            open2y = move[0] + bridge[2][0]
            
            if state.board[diff_y][diff_x] == colour and state.board[open1y][open1x] == 0 and state.board[open2y][open2x] == 0:
                state.vulnerable[colour].append([[open1y, open1x], [open2y, open2x]])

    def check_for_bridge_attack(self, state: BoardState, colour: str):
        """
        check if the previous move was a bridge attack

        """
        # 2d list of pairs of vulnerable moves
        for bridge in state.vulnerable[colour]:
            if bridge[0] == state.prev_move:
                return bridge[1]
            elif bridge[1] == state.prev_move:
                return bridge[0]

        return None


    def potential_db(self, position: list, board: list) -> list:
        """
        A function to calculate the empty double bridge positions for given positon on the board

        Params:
            position: the position to calculate from
            board: the board to return the open double bridges from

        Returns:
            list of these positions
        """

        # empty list to store positions
        bridges = []

        # double bridge positions and their connecting cells stored in the form:
        # [bridge:[i, j], connecting pos 1:[i, j], connecting_pos_2[i, j]]
        bridge_pos = [[[-2,1],[-1,0],[-1,1]], [[-1,2],[-1,1],[0,1]], [[1,1],[0,1],[1,0]], [[2,-1],[1,0],[1,-1]], [[1,-2],[1,-1],[0,-1]],[[-1,-1],[0,-1],[-1,0]]]

        for bridge in bridge_pos:

            # calculate position of the bridge
            diff_y = position[0] + bridge[0][0]
            diff_x = position[1] + bridge[0][1]

            # check boundary conditions and continue to next board position if the boundary
            # have been exceeded
            if diff_x < 0 or diff_y < 0 or diff_x >= self.board_size or diff_y >= self.board_size:
                continue

            # calculate positions of the connecting cells
            open1y = position[0] + bridge[1][0]
            open1x = position[1] + bridge[1][1]

            open2y = position[0] + bridge[2][0]
            open2x = position[1] + bridge[2][1]

            # if all calculated positions open, add to the set
            if board[diff_y][diff_x] == 0 and board[open1y][open1x] == 0 and board[open2y][open2x] == 0:
                bridges.append([diff_y,diff_x])

        return bridges

    def triangle_check(self):
        """
        checks for opposition triangles and returns centre moves to block
        """
        triangle_moves = []
        opp_colour = self.opp_colour

        triangle_spots = [[-2,1], [-1,2], [1,1], [2,-1], [1,-2],[-1,-1]]

        triangle_neighbours = [[-1, 1], [0, 1], [1, 0], [1, -1], [0, -1],[-1, 0]]

        positions = self.graph.get_nodes(opp_colour)

        # flip each elemnt from x,y to y,x
        for i in range (len(positions)):
            positions[i] = positions[i].reverse()

        for i in range(len(positions)):
            for j in range(6):
                empty = [positions[i][0] + triangle_spots[j][0], positions[i][1] + triangle_spots[j][1]]

                if self.board[empty[0]][empty[1]] == 0:
                    if self.board[positions[i][0] + triangle_neighbours[j][0]][positions[i][1] + triangle_neighbours[j][1]] == opp_colour and self.board[positions[i][0] + triangle_neighbours[(j+1) % 6][0]][positions[i][1] + triangle_neighbours[(j+1) % 6][1]] == opp_colour:
                        triangle_moves.append(empty)


if (__name__ == "__main__"):
    agent = Agent17()
    agent.run()