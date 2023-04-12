
"""
COMP30024 Artificial Intelligence, Semester 1, 2022
Project Part B: Playing the Game

A helper to find the next action of the player.
This script contains the implementation of A* algorithm.
Modified from the code written for project A of our group "group_h1"
The main looping logic and structure are referenced from the blog:
https://blog.csdn.net/qq_39687901/article/details/80753433

By Group_h1: Chengyi Huang, Yiya Zhuang
"""

EMPTY_HEX = "" 
class Hexagon:
    """
    A hexagon in the board
    s: color of piece occupying this hex
    (r,q): axial coordinate of this hex
    """
    def __init__(self,r,q,s=""):
        self.s=s
        self.r=r
        self.q=q

    def __eq__(self, other):
        if self.r==other.r and self.q==other.q and self.s==other.s:
            return True
        return False
    def __str__(self):
        return "s:"+str(self.s)+",position:("+str(self.r)+","+str(self.q)+")"

def heuristic(point, goal) :
    """
    Distance calculation function for axial board based on Manhattan distance
    """
    return (abs(goal.r - point.r) + abs(goal.q - point.q + goal.r - point.r) + abs(goal.q - point.q))/2

class AStar:
    """
    A* implementation
    """
    class Node:
        def __init__(self, point, goal, g=0):
            self.point = point
            self.father = None
            self.g = g
            self.h = heuristic(point, goal)
 
    def __init__(self, board, size, start, goal, color, emptyHex=EMPTY_HEX):
        """
        Initialize Astar class
        :param board: 2D array represents the board
        :param start: the start hexagon
        :param goal: the goal hexagon
        :param color: the color of the player, the color of the path that the program is constructing
        :param emptyHex: represent a hexagon that is empty
        """
        self.exploring = []
        self.explored = []
        self.board = board
        self.size = size
        self.start = start
        self.goal = goal
        self.color = color
        self.emptyHex = emptyHex
    
    def outOfBoard(self, point) :
        if point.r < 0 or point.r > self.size - 1 or point.q < 0 or point.q > self.size - 1 :
            return True
        return False
 
    def getMinNode(self):
        """
        get the node with minimum f value from exploring (f=g+h)
        :return: Node
        """
        currentNode = self.exploring[0]
        for node in self.exploring:
            if node.g + node.h < currentNode.g + currentNode.h:
                currentNode = node
        return currentNode
 
    def pointInExplored(self, point):
        for node in self.explored:
            if node.point == point:
                return node
        return None
 
    def pointInExploring(self, point):
        for node in self.exploring:
            if node.point == point:
                return node
        return None
 
    def goalMet(self):
        goalNode = self.pointInExploring(self.goal)
        if goalNode == None :
            goalNode = self.pointInExplored(self.goal)
        return goalNode
 
    def searchNear(self, minF, offsetR, offsetQ):
        """
        explore the hexagon moved with the offset values from the minF node
        :param minF: the node with minimun f value (f=g+h)
        :param offsetR: change in r value
        :param offsetQ: change in q value
        :return: void
        """
        # Set the step cost from minF to currentPoint
        step = 1

        # Out of board checking
        currentPoint = Hexagon(minF.point.r + offsetR, minF.point.q + offsetQ)
        if self.outOfBoard(currentPoint) :
            return
        
        # Obstacle checking
        if self.board[minF.point.r + offsetR][minF.point.q + offsetQ] != self.emptyHex:
            if self.board[minF.point.r + offsetR][minF.point.q + offsetQ] != self.color:
                return
            else:
                step = 0
        
        # Check if currentPoint has already fully explored
        if self.pointInExplored(currentPoint):
            return
        
        

        # Add this currentNode to exploring if it is not yet in the list
        currentNode = self.pointInExploring(currentPoint)
        if not currentNode:
            currentNode = AStar.Node(currentPoint, self.goal, g=minF.g + step)
            currentNode.father = minF
            self.exploring.append(currentNode)
            return
        
        # if currentNode has already added in the exploring, update the node if less cost found
        if minF.g + step < currentNode.g:
            currentNode.g = minF.g + step
            currentNode.father = minF
 
    def start(self):
        """
        Start searching for path
        :return: None, solution will be printed
        """
        # Put the start node into 'exploring' list
        startNode = AStar.Node(self.start, self.goal)
        self.exploring.append(startNode)

        # Explore the board starting with the start node
        while True:
            # Find the node with least f value
            minF = self.getMinNode()

            # Explore the 6 neighbour hex in 6 directions from this hex
            self.searchNear(minF, 0, -1)
            self.searchNear(minF, 0, 1)
            self.searchNear(minF, -1, 0)
            self.searchNear(minF, 1, 0)
            self.searchNear(minF, -1, 1)
            self.searchNear(minF, 1, -1)

            # Mark minF as explored
            self.explored.append(minF)
            self.exploring.remove(minF)

            # Check if the goal is met
            goalNode = self.goalMet()
            if goalNode:
                # Check if there is possible less cost path
                if (self.getMinNode().h + self.getMinNode().g < goalNode.h + goalNode.g) :
                    continue

                # Backtracking the path from the goalNode
                current = goalNode
                pathList = []
                while True:
                    if current.father:
                        pathList.append(current.point)
                        current = current.father
                    else:
                        pathList.append(current.point)
                        break
                
                # Print the solution
                # print(len(pathList))
                pathList = list(reversed(pathList))
                emptyList = []
                occupiedList = []
                for point in pathList :
                    r = point.r
                    q = point.q
                    # print("("+str(r)+","+str(q)+')')
                    if self.board[r][q] == EMPTY_HEX :
                        emptyList.append((r,q))
                    else:
                        occupiedList.append((r,q))
                return (emptyList, occupiedList)
            
            # No solution if all nodes has been explored, and the goal is not met
            if len(self.exploring) == 0:
                # print(0)
                return [], []
