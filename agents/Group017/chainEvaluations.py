board = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
board1 = [
    ["R", 0, 0, 0, 0, 0, 0, 0, 0, 0, "B"],
    [0, 0, "R", 0, 0, "R", 0, "R", 0, 0, "R"],
    [0, "B", 0, "R", "R", 0, "B", 0, "R", 0, 0],
    [0, 0, 0, 0, "B", 0, "R", 0, "B", 0, 0],
    [0, 0, "R", "B", "R", "R", 0, 0, "R", 0, 0],
    [0, 0, 0, "R", "B", "B", 0, 0, "R", 0, 0],
    [0, "B", "B", "B", 0, "B", "B", "R", 0, 0, 0],
    [0, 0, 0, "R", 0, 0, 0, 0, "B", 0, 0],
    [0, 0, 0, 0, 0, "B", 0, 0, 0, "B", 0, 0],
    [0, 0, 0, 0, 0, "B", 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]


def longestChain(pos, end):
    lenPos = []







def playerNeib(colour, target, board_size):

    positions = []
    endpoints = []

    for i in range (0, board_size):
        for j in range (0, board_size):
            if board1[i][j] == colour:
                position = []
                neighbours = 0
                position.append([i,j])

                # neighbour: x,y-1
                eval = neibEval(j, i-1, colour, target)
                if type(eval) == list:
                    neighbours += 1
                position.append(eval)

                # neighbour: x+1,y-1
                eval = neibEval(j+1, i-1, colour, target)
                if type(eval) == list:
                    neighbours += 1
                position.append(eval)

                # neighbour: x+1,y
                eval = neibEval(j+1, i, colour, target)
                if type(eval) == list:
                    neighbours += 1
                position.append(eval)

                # neighbour: x,y+1
                eval = neibEval(j, i+1, colour, target)
                if type(eval) == list:
                    neighbours += 1
                position.append(eval)

                # neighbour: x-1,y+1
                eval = neibEval(j-1, i+1, colour, target)
                if type(eval) == list:
                    neighbours += 1
                position.append(eval)

                # neighbour: x-1,y
                eval = neibEval(j-1, i, colour, target)
                if type(eval) == list:
                    neighbours += 1
                position.append(eval)

                positions.append(position)
                if neighbours == 1:
                    endpoints.append(position)
                board[i][j] = position
    
    return positions, endpoints


def neibEval(x,y, colour, target):
    # wall peice for winning wall 'W' or loosing wall 'L'
    if x < 0 or x > 10 or y < 0 or y > 10:
        if target == 'x':
            if x < 0 or x > 10:
                return 'W'
            if y < 0 or y > 10:
                return 'L'
        elif target == 'y':
            if x < 0 or x > 10:
                return 'L'
            if y < 0 or y > 10:
                return 'W'



    # neigb
    elif board1[y][x] == colour:
        return [x,y]
    
    # empty
    elif board1[y][x] == 0:
        return 0

    # opp
    else:
        return 'X'


# print("B=", playerNeib("B", 'x', 11))
# print("R=", playerNeib("R", 'y', 11))

pos, end = playerNeib("B", 'x', 11)
print(pos)
print(end)

# pos, end = playerNeib("R", 'y', 11)
# print(pos)
# print(end)

print(board)