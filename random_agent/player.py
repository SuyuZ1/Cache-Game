"""
COMP30024 Artificial Intelligence, Semester 1, 2022
Project Part B: Playing the Game
The Player class

By Group_h1: Chengyi Huang, Yiya Zhuang
"""
from random_agent.board import Board
import random

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

        The parameter player is the string "red" if your player will
        play as Red, or the string "blue" if your player will play
        as Blue.
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
                self.board2D[i].append("")

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
        
        random.seed()
        r = random.randint(0, self.n-1)
        q = random.randint(0, self.n-1)
        while self.board.find(r, q)>=0:
            random.seed()
            r = random.randint(0, self.n-1)
            q = random.randint(0, self.n-1)
        
        return (ACT_P, r, q)


    def turn(self, player, action):
        """
        Called at the end of each player's turn to inform this player of 
        their chosen action. Update your internal representation of the 
        game state based on this. The parameter action is the chosen 
        action itself. 
        
        Note: At the end of your player's turn, the action parameter is
        the same as what your player returned from the action method
        above. However, the referee has validated it at this point.
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
            self.board2D[r][q] = ""
            self.board2D[q][r] = player
        self.board.update(player, action)
        
        # Remove pieces taken by this move
        self.diamond_detection(player, action)
    
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
                if cord_color == "":
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
            self.board2D[r][q] = ""
            self.board.remove(r, q)
        
        return
    
    def out_bound(self, r, q):
        if r < 0 or r > self.n - 1 or q < 0 or q > self.n - 1 :
            return True
        return False
