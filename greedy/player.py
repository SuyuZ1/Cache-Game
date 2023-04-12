"""
COMP30024 Artificial Intelligence, Semester 1, 2022
Project Part B: Playing the Game
The Player class

By Group_h1: Chengyi Huang, Yiya Zhuang
"""
from greedy.board import Board
from greedy.astar import *
from greedy.minimax import Minimax

# Action tokens
ACT_S = "STEAL"
ACT_P = "PLACE"

# Color tokens
RED = "red"
BLUE = "blue"




class Player:
    count = 0
    min_heuristic = 99999999
    
    def __init__(self, player, n):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        c: the color of the player
        n: the width of the board
        board: a list of occupied hexagons on the board [color, r, q]
        board2D: a 2D array version of the board, board2D[r][q]="red"/"blue"/""
        """
        self.c = player
        self.n = n
        self.can_steal = True
        self.board = Board()

        # Create an empty board in a form of 2D array
        self.board2D = []
        for i in range(self.n) :
            self.board2D.append([])
            for j in range(self.n) :
                self.board2D[i].append(EMPTY_HEX)

    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """
        # Perform steal if available
        if self.can_steal:
            self.can_steal = False
            if len(self.board.array)==1:
                if self.board.array[0][0] != self.c and \
                    (self.board.array[0][1]==0 or self.board.array[0][1]==self.n-1):
                    return (ACT_S,)

        minimax = Minimax(self)
        for i in range(self.n):
            for j in range(self.n):
                if self.board2D[i][j] == EMPTY_HEX:
                    hexagon = Hexagon(i, j)
                    break
        a, action = minimax.max_value(self.board2D, self.board, hexagon, 1)
        
        return action

    def turn(self, player, action):
        """
        Called at the end of each player's turn to inform this player of 
        their chosen action.
        """
        # Place the piece
        if (len(action) >1):
            # This action is a placing action ("PLACE", r, q)
            r = action[1]
            q = action[2]
            self.board2D[r][q] = player
        else:
            # This action is a stealing action ("STEAL",)
            # Stealing the first move from another player
            r = self.board.array[0][1]
            q = self.board.array[0][2]
            self.board2D[r][q] = EMPTY_HEX
            self.board2D[q][r] = player
        self.board.update(player, action)
        
        # Remove pieces taken by this move
        self.diamond_detection(player, action)

    def diamond_detection(self, color, action):
        """
        Diamond_detection, given an action,
        Remove any pieces on the board that is taken by this action with diamond capture.
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
