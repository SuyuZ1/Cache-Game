from team_7k7k.rules import find_longest


# check if the game is over
def check_end(board, color):
    end = False
    # check blue player
    if (color == "blue"):
        for row in range(board.n):
            if board.grid[row][0] == color:
                end = find_path(color, board, (row, 0))
            
            if end:
                return True
    #check red player
    else:
        for col in range(board.n):
            if board.grid[0][col] == color:
                end = find_path(color, board, (0, col))
            if end:
                return True
    return end

# find the path of specify color through board
def find_path(color, board, start):
    visited = {}
    win = False
    # use dfs to explore path
    def dfs(node, visited, win):
        visited[node] = True
        childs = board.find_neighbor_list(node[0], node[1])
        for child in childs:
            if (win):
                return win
            if not child in visited:
                if (board.grid[child[0]][child[1]]==color):
                    if ((color == "red" and child[0] == board.n-1 ) or (color == "blue" and child[1] == board.n-1)):
                        return True
                    else:
                        win = dfs(child, visited, win)
        return win
    if start:
        win = dfs(start, visited, win)
    return win


# detect all possible acton, retrun result as a list
def find_possible_actions(board):
    actions = []
    for r in range(board.n):
        for q in range(board.n):
            # if this poisiton is empty, add in result
            if (board.grid[r][q] == 0):
                actions.append((r, q))
    return actions

# apply minmax to choose action
def choose_action(board, depth):
    value, next = minimax(board, depth, True, float('-inf'), float('inf'))

    return ('PLACE', next[0], next[1])


# return utility score
def evaluate(board):
    myLongest = find_longest(board, board.player)
    enemyLongest = find_longest(board, board.enemy)
    value = 0.4*(myLongest - enemyLongest) + 0.6*(len(board.action_list) - len(board.enemy_actions)) 
    return value


# put player in the speciy position
def reset_state(board, captured, player):
    if(captured):
        for node in captured:
            board.grid[node[0]][node[1]] = player



def minimax(board, depth, maxPlayer, alpha, beta, node=None):
    
    if check_end(board, board.player):
        return float('inf'), node
    elif check_end(board, board.enemy):
        return float('-inf'), node

    #leaf nodes
    if depth== 0:
        utility = evaluate(board)
        return utility, node

    actions = find_possible_actions(board)
    #print(actions)
    if (maxPlayer):
        value = float('-inf')
        if (not actions):
            return value, node
        elem = actions[0]
        for next in actions:
            r=next[0]
            q=next[1]
            original1 = board.action_list[:]
            original2 = board.enemy_actions[:]

            board.grid[r][q] = board.player
            board.action_list.append(next)
            #board.reset_weight(board.player)

            captured = board.update_captured(board.player, next)

            #board.update_weight(board.action_list, board.player)
            weight, move = minimax(board, depth-1, False, alpha, beta, next)
            prev = value

            value = max(value, weight)
            
            alpha = max(alpha, value)
            if (prev!=value):
                elem = next

            board.grid[r][q] = 0
            board.action_list = original1[:]
            board.enemy_actions = original2[:]
            reset_state(board, captured, board.enemy)
            if value >= beta:
                break
        return value, elem

        
    else:
        value = float('inf')
        if (not actions):
            return value, node
        elem = actions[0]
        for next in actions:
            r=next[0]
            q=next[1]
            original1 = board.action_list[:]
            original2 = board.enemy_actions[:]

            board.grid[r][q] = board.enemy
            board.enemy_actions.append(next)
            #board.reset_weight(board.enemy)

            captured = board.update_captured(board.enemy, next)

            #board.update_weight(board.enemy_actions, board.enemy)

            weight, move = minimax(board, depth-1, True, alpha, beta, next)
            value = min(value, weight)
            beta = min(beta, value)

            board.grid[r][q] = 0
            board.action_list = original1[:]
            board.enemy_actions = original2[:]            
            reset_state(board, captured, board.player)
            
            if value <= alpha:
                break
        return value, elem
    