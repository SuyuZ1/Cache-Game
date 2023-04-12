"""
COMP30024 Artificial Intelligence, Semester 1, 2022
Project Part B: Playing the Game
The implementation of minimax

By Group_h1: Chengyi Huang, Yiya Zhuang
"""
from greedy.player import *
from greedy.astar import *

# How many steps to look forwards
MIN_STEP = 1
THRES_ACTIONS_4 = 15
THRES_ACTIONS_1 = 10000000000
THRES_ACTIONS_MIN = 10000000000
INFINITY = 9999999999

# Action tokens
ACT_S = "STEAL"
ACT_P = "PLACE"

# Color tokens
RED = "red"
BLUE = "blue"

class Minimax:
    """
    Minimax implementaion
    """
    class Node:
        def __init__(self, board2D, board, hexagon, color, depth, max_step=MIN_STEP):
            self.hexagon = hexagon
            self.father = None
            self.depth = depth
            self.c = color
            self.n = len(board2D)

            self.board = Board()
            for element in board.array:
                self.board.array.append(element)
            
            self.board2D = []
            for i in range(self.n):
                self.board2D.append([])
                for j in range(self.n):
                    self.board2D[i].append(board2D[i][j])
            
            self.value = -INFINITY
            if hexagon.s!=EMPTY_HEX:
                self.update_board(hexagon.s, (ACT_P, hexagon.r, hexagon.q))
            self.value = self.evaluate()
        
        def __eq__(self, other):
            if self.depth==other.depth and self.hexagon==other.hexagon and self.father==other.father:
                return True
            return False
        
        def update_board(self, color, action):
            # Place the piece
            if (len(action) >1):
                # This action is a placing action ("PLACE", r, q)
                r = action[1]
                q = action[2]
                self.board2D[r][q] = color
            else:
                # This action is a stealing action ("STEAL",)
                # Stealing the first move from another player
                r = self.board.array[0][1]
                q = self.board.array[0][2]
                self.board2D[r][q] = EMPTY_HEX
                self.board2D[q][r] = color
            self.board.update(color, action)
            self.diamond_detection(color, action)
        
        def evaluate(self):
            """
            Evaluate the value of a hexagon in the priority queue
            """
            heuristic = Heuristic(len(self.board2D), self.board, self.board2D, self.c)
            return -heuristic.min_h(self.c)+heuristic.color_score(self.c, self.board)
        
        def diamond_detection(self, color, action):
            """
            diamond_detection, given an action,
            remove any pieces on the board that is taken by this action with diamond capture.
            """
            if len(action)==1:
                # is a steal action, no need to detect for diamond captures
                return 
            
            # list of offsets, key is for returned hex,
            # value is for other components in diamond
            check_dict = {(2, -1): [(1, -1), (1, 0)], (-1, 0): [(0, -1), (-1, 1)],
                        (1, -1): [(0, -1), (1, 0)], (0, 1): [(1, 0), (-1, 1)],
                        (1, -2): [(0, -1), (1, -1)], (-1, -1): [(0, -1), (-1, 0)],
                        (-2, 1): [(-1, 1), (-1, 0)], (1, 0): [(0, 1), (1, -1)],
                        (-1, 1): [(0, 1), (-1, 0)], (0, -1): [(-1, 0), (1, -1)],
                        (-1, 2): [(0, 1), (-1, 1)], (1, 1): [(0, 1), (1, 0)]
                        }

            r = action[1]
            q = action[2]
            taken = []
            for key, value in check_dict.items():
                # check if target hex out of bound
                if self.out_bound(r + key[0], q + key[1]):
                    continue

                # check if target hex is the same color
                if self.board2D[r + key[0]][q + key[1]] != color:
                    continue

                # iterate other hex component in diamond
                is_diamond = True
                for cord in value:
                    

                    # out of bound check
                    if self.out_bound(r + cord[0], q + cord[1]):
                        is_diamond = False
                        break

                    cord_color = self.board2D[r + cord[0]][q + cord[1]]
                    
                    # empty check
                    if cord_color == EMPTY_HEX:
                        is_diamond = False
                        break

                    # check if other hex in diamond satisfy opposite color
                    if cord_color == color:
                        is_diamond = False
                        break

                if is_diamond:
                    taken_piece = (r + value[0][0], q + value[0][1])
                    taken.append(taken_piece)
                    taken_piece = (r + value[1][0], q + value[1][1])
                    taken.append(taken_piece)
                
            # remove the pieces that is taken by this action\
            for piece in taken :
                r = piece[0]
                q = piece[1]
                self.board2D[r][q] = EMPTY_HEX
                self.board.remove(r, q)
            
            return
        
        def out_bound(self, r, q):
            if r < 0 or r > self.n - 1 or q < 0 or q > self.n - 1 :
                return True
            return False
        
    def __init__(self, player):
        self.c = player.c
        self.n = player.n

        self.board = Board()
        for element in player.board.array:
            self.board.array.append(element)
        
        self.board2D = []
        for i in range(self.n):
            self.board2D.append([])
            for j in range(self.n):
                self.board2D[i].append(player.board2D[i][j])
    
    def max_value(self, board2D, board, hexagon, depth, max_step=MIN_STEP):
        action = (ACT_P, hexagon.r, hexagon.q)
        max_step = 1

        #print("node creation:")
        node = self.Node(board2D, board, hexagon, self.c, depth, max_step)

        # Traverse each possible action from this state
        largest = -INFINITY
        for i in range(self.n):
            for j in range(self.n):
                if node.board2D[i][j]!=EMPTY_HEX:
                    continue
                nextHex = Hexagon(i, j, self.c)
                nextNode = self.Node(node.board2D, node.board, nextHex, self.c, depth+1, max_step)
                nextNode.father = node
                value = nextNode.value
                # print(value, nextHex)
                if value>=largest:
                    largest = value
                    largestNode = nextNode
        
        action = (ACT_P, largestNode.hexagon.r, largestNode.hexagon.q)
        node.value = largest
        print(largest)
            
        return largest, action
    
    def check_win(self, color, board, board2D):
        visited = []
        
        for node in board.array:
            if (color == "red"):
                if (node not in visited) and (node[0] == color) and (node[1] == 0):
                    bool = self.dfs_win(board2D, visited, color, node)
                    if bool:
                        return True

            else:
                if (node not in visited) and (node[0] == color) and (node[2] == 0):
                    bool = self.dfs_win(board2D, visited, color, node)
                    if bool:
                        return True    
        return False

    def dfs_win(self, board2D, visited, color, node):
        visited.append(node)
        ##print("win_check: ", visited)

        if color == "red":
            if node[1] == self.n-1:
                ##print("winnnnnnnnnnnnnnnnnnnnnn")
                # print("node:",node)
                return True
        

            else:
                direction = [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, 1], [1, -1]]

                for dir in direction:
                    r = node[1] + dir[0]
                    q = node[2] + dir[1]
                    neighbor_node = [color, r, q]
                    if (neighbor_node not in visited) and (not self.out_bound(r, q)) and (board2D[r][q] == color):
                        bool = self.dfs_win(board2D, visited, color, neighbor_node)
                        # print("node:",neighbor_node)
                        if bool:
                            return True
        else:
            if node[2] == self.n - 1:
                ##print("winnnnnnnnnnnnnnnnnnnnnn")
                return True


            else:
                direction = [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, 1], [1, -1]]

                for dir in direction:
                    r = node[1] + dir[0]
                    q = node[2] + dir[1]
                    neighbor_node = [color, r, q]
                    if (neighbor_node not in visited) and (not self.out_bound(r, q)) and (board2D[r][q] == color):
                        bool = self.dfs_win(board2D, visited, color, neighbor_node)
                        if bool:
                            return True
        return False
         
    
    def out_bound(self, r, q):
            if r < 0 or r > self.n - 1 or q < 0 or q > self.n - 1 :
                return True
            return False


# Heuristic
class Heuristic:  
    def __init__(self, n, board, board2D, color):
        self.count = 0
        self.min_heuristic = INFINITY
        self.n = n
        self.startNode = None
        self.endNode = None
        self.vertical = True
        self.around_count = 0
        self.c = color

        self.board = Board()
        for element in board.array:
            self.board.array.append(element)
        
        self.board2D = []
        for i in range(len(board2D)):
            self.board2D.append([])
            for j in range(len(board2D)):
                self.board2D[i].append(board2D[i][j])

    def distance_win(self, node, color, direct):
        """
        Distance calculation function for axial board based on Manhattan distance
        """
        if color == RED:
            if direct == "up":
                return (abs(node[1] - (self.n-1)) + abs(node[1] - (self.n-1))) / 2
            if direct == "down":
                return node[1]

        else:
            if direct == "right":
                return (abs(node[2] - (self.n-1)) + abs(node[2] - (self.n-1))) / 2
            if direct == "left":
                return node[2]



    
    def out_bound(self, r, q):
        if r < 0 or r > self.n - 1 or q < 0 or q > self.n - 1 :
            return True
        return False


    def diamond_caption(self, hexagon):
        # list of offsets, key is for returned hex, value is for other components in
        # diamond
        check_dict = {(2, -1): [(1, -1), (1, 0)], (-1, 0): [(0, -1), (-1, 1)],
                    (1, -1): [(0, -1), (1, 0)], (0, 1): [(1, 0), (-1, 1)],
                    (1, -2): [(0, -1), (1, -1)], (-1, -1): [(0, -1), (-1, 0)],
                    (-2, 1): [(-1, 1), (-1, 0)], (1, 0): [(0, 1), (1, -1)],
                    (-1, 1): [(0, 1), (-1, 0)], (0, -1): [(-1, 0), (1, -1)],
                    (-1, 2): [(0, 1), (-1, 1)], (1, 1): [(0, 1), (1, 0)]
                    }

        available_list = []

        color = hexagon.s
        r = hexagon.r
        q = hexagon.q

        for key, value in check_dict.items():
            is_out = 0
            is_diamond = 1
            
            # check if target hex out of bound
            if self.out_bound(r + key[0], q + key[1]):
                continue

            # check if target hex empty
            if self.board2D[r + key[0]][q + key[1]] != "":
                continue

            for cord in value:
                # iterate other hex component in diamond
                if self.out_bound(r + cord[0], q + cord[1]):
                    is_out = 1
                    is_diamond = 0
                    break

                # check if other hex in diamond satisfy opposite color
                if color == BLUE:
                    if self.board2D[r + cord[0]][q + cord[1]] != RED:
                        is_diamond = 0
                else:
                    if self.board2D[r + cord[0]][q + cord[1]] != BLUE:
                        is_diamond = 0

            if is_diamond:
                available_list.append((r + key[0], q + key[1]))

        return available_list

    def dfs(self, visited, color, count, node):
        self.count += 1
        count += 1

        ##print("count is ", self.count)
        visited.append(node)
        ##print(visited)
        direction = [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, 1], [1, -1]]

        for dir in direction:
            r = node[1] + dir[0]
            q = node[2] + dir[1]
            neighbor_node = [color, r, q]
            if (neighbor_node not in visited) and (not self.out_bound(r, q)) and (self.board2D[r][q] == color):
                self.around(self.board2D, self.c, neighbor_node)
                if self.vertical:
                    if neighbor_node[1] < self.startNode[1]:
                        self.startNode = neighbor_node
                    elif neighbor_node[1] > self.endNode[1]:
                        self.endNode = neighbor_node
                else:
                    if neighbor_node[2] < self.startNode[2]:
                        self.startNode = neighbor_node
                    elif neighbor_node[2] > self.endNode[2]:
                        self.endNode = neighbor_node
                count = self.dfs(visited, color, count, neighbor_node)

        return count

    def dfs2(self, visited, color, node, direct):
        

        heuristic = self.distance_win(node, color, direct)
        
        if heuristic < self.min_heuristic:
            self.min_heuristic = heuristic


        ##print("count is ", self.count)
        visited.append(node)
        ##print(visited)
        direction = [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, 1], [1, -1]]

        for dir in direction:
            r = node[1] + dir[0]
            q = node[2] + dir[1]
            neighbor_node = [color, r, q]
            if (neighbor_node not in visited) and (not self.out_bound(r, q)) and (self.board2D[r][q] == color):
                self.dfs2(visited, color, neighbor_node, direct)



    def longest_path(self, color):
        visited = []
        max_len = 0
        max_around = 0
        self.startNode = None
        self.endNode = None
        startNode = None
        endNode = None

        if color == BLUE:
            self.vertical = False
        else:
            self.vertical = True

        for node in self.board.array:
            if (node not in visited) and node[0] == color:
                self.count = 1
                self.around_count = 0
                self.startNode = node
                self.endNode = node
                self.dfs(visited, color, self.count, node)
                if self.vertical:
                    self.count = self.endNode[1] - self.startNode[1]
                else:
                    self.count = self.endNode[2] - self.startNode[2]

                if self.count > max_len:
                    # Check if this path is win
                    if self.count==self.n-1:
                        max_len = INFINITY
                        max_around = self.around_count
                        break
                    
                    max_len = self.count
                    startNode = self.startNode
                    endNode = self.endNode
                    max_around = self.around_count

            ##print("visited length is ", len(visited))

        ##print("longest length is ", max_len)
        self.startNode = startNode
        self.endNode = endNode
        self.around_count = max_around
        return max_len

    def color_score(self, color, board):
        blue_count = 0
        red_count = 0
        for hex in board.array:
            if hex[0] == "blue":
                blue_count += 1
                # if hex[2]==0 or hex[2]==self.n-1:
                #     blue_count += 2
            elif hex[0] == "red":
                red_count += 1
                # if hex[1]==0 or hex[1]==self.n-1:
                #     red_count += 2

        if color == "red":
            return red_count - blue_count
        else:
            return blue_count - red_count

    def min_h(self, color):
        # assume player is red



        self.min_heuristic = INFINITY

        if (len(self.board.array) == 0):
            self.min_heuristic = self.n

        visited = []
        for node in self.board.array:
            if color == "red":
                if (node not in visited) and node[0] == color and (node[1] == 0 or node[1] == self.n-1):
                    if (node[1] == 0):
                        direct = "up"
                    else:
                        direct = "down"
                    self.dfs2(visited, color, node, direct)
            else:
                if (node not in visited) and node[0] == color and (node[2] == 0 or node[2] == self.n-1):
                    if (node[2] == 0):
                        direct = "right"
                    else:
                        direct = "left"

                    self.dfs2(visited, color, node, direct)

        if self.min_heuristic == INFINITY:
            self.min_heuristic = self.n
            #print("!!!!", self.min_heuristic)

        mini_h = self.min_heuristic
        return mini_h

        
        

        

        ##print("final minimum h is ", self.min_heuristic)

    def distance_to_win(self, color, from_top=True):
        visited = []
        for node in self.board.array:
            if (color == "red"):
                if (node not in visited) and (node[0] == color):
                    if node[1] == 0:
                        bool = self.dfs_distance_to_win(visited, color, node)
                        if bool:
                            return 1
                    elif node[1] == self.n-1:
                        bool = self.dfs_distance_to_win(visited, color, node, False)
                        if bool:
                            return 1

            else:
                if (node not in visited) and (node[0] == color):
                    if node[2] == 0:
                        bool = self.dfs_distance_to_win(visited, color, node)
                        if bool:
                            return 1  
                    elif node[2] == self.n-1:
                        bool = self.dfs_distance_to_win(visited, color, node, False)
                        if bool:
                            return 1  
                                  
        return 0
    
    def around(self, board2D, color, node):
            """
            Count how good the situation is around the given node for the given player
            """
            direction = [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, 1], [1, -1]]

            for dir in direction:
                r = node[1] + dir[0]
                q = node[2] + dir[1]
                if self.out_bound(r, q):
                    continue

                weight = 1
                if node[0] == RED:
                    if abs(dir[0])==1:
                        weight *= 10
                else:
                    if abs(dir[1])==1:
                        weight *= 10
                if board2D[r][q]==EMPTY_HEX:
                    self.around_count += 1 * weight
                elif board2D[r][q]==color:
                    self.around_count += 2 * weight
                else:
                    self.around_count -= 1 * weight
            
            return
