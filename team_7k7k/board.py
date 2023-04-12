from numpy import array, roll
import numpy as np

# -------------constants from referee provided from sekleton code

_ADD = lambda a, b: (a[0] + b[0], a[1] + b[1])

# Neighbour hex steps in clockwise order
_HEX_STEPS = array([(1, -1), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1)], 
    dtype="i,i")

# Pre-compute diamond capture patterns - each capture pattern is a 
# list of offset steps:
# [opposite offset, neighbour 1 offset, neighbour 2 offset]
#
# Note that the "opposite cell" offset is actually the sum of
# the two neighbouring cell offsets (for a given diamond formation)
#
# Formed diamond patterns are either "longways", in which case the
# neighbours are adjacent to each other (roll 1), OR "sideways", in
# which case the neighbours are spaced apart (roll 2). This means
# for a given cell, it is part of 6 + 6 possible diamonds.
_CAPTURE_PATTERNS = [[_ADD(n1, n2), n1, n2] 
    for n1, n2 in 
        list(zip(_HEX_STEPS, roll(_HEX_STEPS, 1))) + 
        list(zip(_HEX_STEPS, roll(_HEX_STEPS, 2)))]
# ------------------------------------------------------------------

class Board:
    """a board for our grame.
    it contains the width and height of the board which are the same.
    and a 2darray of the total hexgonal grid which has a value of either 1 or 0 
    to record if the current position is occuiped or not, with 1 = occuiped"""

    def __init__(self, n, player, enemy):
        self.n = n
        self.player = player
        self.grid = []
        self.grid = [[0 for r in range(n)]for q in range(n)]
    
        self.action_list = []
        self.enemy_actions = []
        self.enemy = enemy
        self.weight={}
        self.set_weight()


    # set weight for utility function
    def set_weight(self):
        n = self.n
        num = 1

        for i in range(1, n+1):
            self.weight[i] = num
            num *=n

  
    # update board information
    def update(self,player,action):

        # if action is place piece, place it in the board
        if (action[0]=="PLACE"):
            r = action[1]
            q = action[2]
            self.grid[r][q] = player
            # update action list for place action
            if (player == self.player):
                self.action_list.append((int(action[1]), int(action[2])))
            else:
                self.enemy_actions.append((int(action[1]), int(action[2])))

            self.update_captured(player, (r, q))



        # check grid, if there is one piece in the red piece,
        # then we steal that piece, and clear red piece
        elif(action[0]=="STEAL"):
            for r in range(self.n):
                for q in range(self.n):
                    if(self.grid[r][q]!=0):
                        self.grid[q][r]=player
                        # update action list for steal action
                        if (player == self.player):
                            self.action_list.append((q, r))
                        else:
                            self.enemy_actions.append((q, r))
                        if(r!=q):
                            self.grid[r][q]=0
                            #self.update_weight()
                        return 
    
    # update captured information, remove captured piece
    def update_captured(self, player, action):
        captured = self._apply_captures(player, action)
        
        for node in captured:
            self.grid[node[0]][node[1]] = 0
            if player == self.enemy :
                #remove self piece
                self.action_list.remove(node)
            else:
                #remove enemy piece
                self.enemy_actions.remove(node)
            
        return captured

        
    # find neighbour piece, return result as a list
    def find_neighbor_list(self,r,q):
        """ (r-1, q) (r-1, q+1) 
            (r, q-1) (r, q+1)
            (r+1, q-1) (r+1, q) 
        """ 

        # check poisiton is out of bound or not 
        result =[]
        if (r-1 >= 0):
            result.append((r-1,q))
                
        if (q+1 < self.n and r-1 >= 0):
            result.append((r-1,q+1))
                
        if (q-1 >= 0):
            result.append((r,q-1))
              
        if (q+1 < self.n):
            result.append((r,q+1))
                
        if (r+1 < self.n and q-1 >= 0):
            result.append((r+1,q-1))
                
        if (r+1 < self.n):
            result.append((r+1,q))
        return result


    def print_grid(self):
       print(np.matrix(self.grid))
              
    def __getitem__(self, item):
        return self.grid[item]
    
    #---------below functions from board.py from referee provided in skeleton code
    
    def inside_bounds(self, coord):
        """
        True iff coord inside board bounds.
        """
        r, q = coord
        return r >= 0 and r < self.n and q >= 0 and q < self.n

    
    def _apply_captures(self, player, action):
        """
        Check coord for diamond captures, and apply these to the board
        if they exist. Returns a list of captured token coordinates.
        """
        r = action[0]
        q = action[1]
        curr_type = self.grid[r][q]
        if (player == self.player):
            mid_type = self.enemy
        else:
            mid_type = self.player

        captured = set()

        # Check each capture pattern intersecting with coord
        for pattern in _CAPTURE_PATTERNS:
            coords = [_ADD((r,q), s) for s in pattern]
            # No point checking if any coord is outside the board!
            if all(map(self.inside_bounds, coords)):
                tokens = [self.grid[coord[0]][coord[1]] for coord in coords]
                if tokens == [curr_type, mid_type, mid_type]:
                    # Capturing has to be deferred in case of overlaps
                    # Both mid cell tokens should be captured
                    captured.update(coords[1:])
        return list(captured)
    
    #---------above functions from board.py from referee provided in skeleton code


        




    