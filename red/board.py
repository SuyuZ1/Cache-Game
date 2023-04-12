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
        self.weight_dict = {}
        self.action_list = []
        self.enemy_actions = []
        self.enemy = enemy
        self.weight = {}

        self.set_weight()


    def set_weight(self):
        n = self.n
        num = 1

        for i in range(1, n+1):
            self.weight[i] = num
            num *=n
        
        
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
    

    def update_captured(self, player, action):
        captured = self._apply_captures(player, action)
        
        for node in captured:
            self.grid[node[0]][node[1]] = 0
            if player == self.enemy :
                self.action_list.remove(node)
            else:
               
                self.enemy_actions.remove(node)
            
        return captured



    # node weight caculations and updates 
    def update_weight(self, actions, player):

        """ (r-1, q) (r-1, q+1) 
            (r, q-1) (r, q+1)
            (r+1, q-1) (r+1, q) """

        for action in actions:
            if (player == self.player):
                self.player_weight(player, action, 1)
            else:
                self.player_weight(player, action, -1)

        
    def player_weight(self, player, action, oppoTag):
        grid= self.grid
        r=action[0]
        q=action[1]
        weight = 10

        if (r-1 >= 0):
            if (grid[r-1][q] == 0):
                self.weight_dict[(r-1, q)] += (self.check_dif_color(r-1,q, player)*oppoTag)
                if (player == "red"):
                    self.weight_dict[(r-1, q)] += (weight*oppoTag)
            else:
                self.weight_dict[(r-1, q)] -= (self.check_dif_color(r-1,q, player)*oppoTag)

        if (q+1 < self.n and r-1 >= 0):
            if (grid[r-1][q+1] == 0):# and not self.check_dif_color(r-1,q+1, player)):
                self.weight_dict[(r-1, q+1)] += (self.check_dif_color(r-1,q, player)*oppoTag)
                self.weight_dict[(r-1, q+1)] += (weight*oppoTag)

                
            else:
                self.weight_dict[(r-1, q+1)] -= (self.check_dif_color(r-1,q, player)*oppoTag)

        if (q-1 >= 0):
            if (grid[r][q-1] == 0):# and not self.check_dif_color(r,q-1, player)):
                self.weight_dict[(r, q-1)] += (self.check_dif_color(r-1,q, player)*oppoTag)
                if (player == "blue"):
                    self.weight_dict[(r, q-1)] += (weight*oppoTag)

            else:
                self.weight_dict[(r, q-1)] -= (self.check_dif_color(r-1,q, player)*oppoTag)

        if (q+1 < self.n):
            if (grid[r][q+1] == 0):# and not self.check_dif_color(r,q+1, player)):
                self.weight_dict[(r, q+1)] += (self.check_dif_color(r-1,q, player)*oppoTag)
                if (player == "blue"):
                    self.weight_dict[(r, q+1)] += (weight*oppoTag)
            else:
                self.weight_dict[(r, q+1)] -= (self.check_dif_color(r-1,q, player)*oppoTag)
        
        if (r+1 < self.n and q-1 >= 0):
            if (grid[r+1][q-1] == 0): #and not self.check_dif_color(r+1,q-1, player)):
                self.weight_dict[(r+1, q-1)] += (self.check_dif_color(r-1,q, player)*oppoTag)
                self.weight_dict[(r+1, q-1)] += (weight*oppoTag)
                
            else:
                self.weight_dict[(r, q-1)] -= (self.check_dif_color(r-1,q, player)*oppoTag)
        if (r+1 < self.n):
            if (grid[r+1][q] == 0):#and not self.check_dif_color(r+1,q, player)):
                self.weight_dict[(r+1, q)] += (self.check_dif_color(r-1,q, player)*oppoTag)
                if (player == "red"):
                    self.weight_dict[(r+1, q)] += (weight*oppoTag)
            else:
                self.weight_dict[(r+1, q)] -= (self.check_dif_color(r-1,q, player)*oppoTag)


        """ 
        capture_list = self.check_capture(r,q, player)

        for node in capture_list:
            self.weight_dict[(node[0], node[1])] += (10*oppoTag)
        """



    def check_dif_color(self,r,q, color):
        grid = self.grid
        result = 0
        #color = self.player
        
        """ (r-1, q) (r-1, q+1) 
            (r, q-1) (r, q+1)
            (r+1, q-1) (r+1, q) """          
        if (r-1 >= 0):
            if (grid[r-1][q] == color):
                result+=2
                
        if (q+1 < self.n and r-1 >= 0):
            if (grid[r-1][q+1] ==color):
                result+=2
                
        if (q-1 >= 0):
            if (grid[r][q-1] ==color):
                result+=2
        
        if (q+1 < self.n):
            if (grid[r][q+1] ==color):
                result+=2
        
        if (r+1 < self.n and q-1 >= 0):
            if (grid[r+1][q-1] ==color):
                result+=2
        
        if (r+1 < self.n):
            if (grid[r+1][q] ==color):
                result+=2
        return result

    def check_capture(self,r,q, color):
        result = []
        #find neighbours for current node
        current_neigbor = self.find_neighbor_list(r,q)
        #find the neighbours if occupied by opponent player
        diff_color_neighbor = self.get_color_list(current_neigbor, color)
        
        for node in diff_color_neighbor:
            #find neighbours for each opponent 
            temp_neighbor_list = self.find_neighbor_list(node[0],node[1])
            #find same colour neighbours of this opponent 
            temp_neighbor_samecolor_list = self.get_color_list(temp_neighbor_list, color)

            #the intersection of neighbours of original node and neighbour(opponent player)
            #contains possibile diamond position 
            candidate = self.get_common_index(diff_color_neighbor,temp_neighbor_samecolor_list)
            for item in candidate:
                get_common_index = self.get_common_index(temp_neighbor_list, self.find_neighbor_list(item[0],item[1]))
                result.extend(get_common_index)
        
        return list(set(result))
        
    
    def find_neighbor_list(self,r,q):
        """ (r-1, q) (r-1, q+1) 
            (r, q-1) (r, q+1)
            (r+1, q-1) (r+1, q) 
        """ 

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
    

    def get_color_list(self,lst, color):
        result = []
        for index in lst:
            if(self.grid[index[0]][index[1]]!=color and self.grid[index[0]][index[1]]!=0):
                result.append(index)
        return result


    def get_common_index(self,lst1,lst2):
        result=[]
        for item1 in lst1:
            for item2 in lst2:
                if(item1==item2):
                    result.append(item1)
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


        




    