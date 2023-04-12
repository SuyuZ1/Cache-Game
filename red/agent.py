from red.rules import find_longest

def check_end(board, color):
    end = False
    if (color == "blue"):
        for row in range(board.n):
            if board.grid[row][0] == color:
                end = find_path(color, board, (row, 0))
                
            if end:
                return True
    else:
        for col in range(board.n):
            if board.grid[0][col] == color:
                end = find_path(color, board, (0, col))
            if end:
                return True
    return end


def find_path(color, board, start):
    visited = {}
    win = False
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

     

def find_possible_actions(board):
    actions = []
    for r in range(board.n):
        for q in range(board.n):
            if (board.grid[r][q] == 0):
                actions.append((r, q))
    return actions



def choose_action(board, depth):
    value, next = minimax(board, depth, True, float('-inf'), float('inf'))

    return ('PLACE', next[0], next[1])



def evaluate(board, node):
    sumLongest = find_longest(board, board.player)
    
    enemySum = find_longest(board, board.enemy)
  
    #value = 0.2*(myLongest - enemyLongest) + 0.8*weight
    #value = 0.2*(myLongest - enemyLongest) + 0.4*(len(board.action_list) - len(board.enemy_actions)) + 0.3*weight + 0.1*(myMiddleCount-enemyMiddleCount)
    #value = 0.3*(myLongest - enemyLongest) + 0.6*(len(board.action_list) - len(board.enemy_actions)) + 0.1*(myMiddleCount-enemyMiddleCount)
    
    value = 0.3*(sumLongest - enemySum) 
    + 0.7*(len(board.action_list) - len(board.enemy_actions))
    
    return value



def reset_state(board, captured, player):
    if(captured):
        for node in captured:
            board.grid[node[0]][node[1]] = player
            """ if (player == board.player):
                board.action_list.append(node)
            else:
                board.enemy_actions.append(node)
        
 """

def minimax(board, depth, maxPlayer, alpha, beta, node=None):
    if check_end(board, board.player):
        return float('inf'), node
    elif check_end(board, board.enemy):
        return float('-inf'), node

     #leaf nodes
    if depth== 0:

        utility = evaluate(board, node)
       
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

            board.grid[r][q] = board.player

            original1 = board.action_list[:]
            original2 = board.enemy_actions[:]

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
            
            board.grid[r][q] = board.enemy

            original1 = board.action_list[:]
            original2 = board.enemy_actions[:]

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
    

    
""" function alphabeta(node, depth, α, β, maximizingPlayer) is
    if depth = 0 or node is a terminal node then
        return the heuristic value of node
    if maximizingPlayer then
        value := −∞
        for each child of node do
            value := max(value, alphabeta(child, depth − 1, α, β, FALSE))
            α := max(α, value)
            if value ≥ β then
                break (* β cutoff *)
        return value
    else
        value := +∞
        for each child of node do
            value := min(value, alphabeta(child, depth − 1, α, β, TRUE))
            β := min(β, value)
            if value ≤ α then
                break (* α cutoff *)
        return value """
    

