"""
COMP30024 Artificial Intelligence, Semester 1, 2022
Project Part B: Playing the Game
The Board Class

By Group_h1: Chengyi Huang, Yiya Zhuang
"""
from greedy.player import *

class Board:
    """
    Board class, represent an nxn board of the game Cachex
    Is a list of list, each represent an occupied hexagon on the board [color, r, q]
    """
    def __init__(self):
        """
        Initialize an empty board
        """
        self.array = []
    
    def update(self, player, action):
        """
        Update the board by an action. Not considering diamond caption
        """
        if (len(action) >1):
            # This action is a placing action ("PLACE", r, q)
            r = action[1]
            q = action[2]
            if self.find(r,q)>=0:
                print("error, placing occupied item", player,action)
                print(self.array)
                exit()
            self.array.append([player, r, q])
        else:
            # Steal the first move from another player
            r = self.array[0][1]
            q = self.array[0][2]
            self.remove(r, q)
            self.array.append([player, q, r])
        
    def find(self, r, q):
        """
        Find the index of the given coordinate
        """
        for i in range(len(self.array)):
            if self.array[i][1]==r:
                if self.array[i][2]==q:
                    return i
        return -1

    def remove(self, r, q) :
        """
        Remove the occupied piece on the given coordinate
        """
        index = self.find(r, q)
        if index >= 0:
            self.array.pop(index)
    
    def count(self, color):
        """
        Count the number of pieces on the board in the given color
        """
        count = 0
        for i in self.array:
            if i[0] == color:
                count += 1
        return count
    
    def count(self, start, end, row=True):
        """
        Count the number of pieces on the board in the range
        """
        count = 0
        for i in self.array:
            if row:
                r = i[1]
                if r>=start and r<=end:
                    count += 1
            else:
                q = i[2]
                if q>=start and q<=end:
                    count += 1
        return count
