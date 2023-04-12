from red.board import Board
from red.rules import check_if_steal
from red.agent import choose_action



class Player:

    def __init__(self, player, n):
        """
        Called once at the beginning of a game to initialise this player.
        Set up an internal representation of the game state.

        The parameter player is the string "red" if your player will
        play as Red, or the string "blue" if your player will play
        as Blue.
        """
        # put your code here
        self.n = n
        self.player = player
        if player == "red":
            self.board = Board(n, player, "blue")
        else:
            self.board = Board(n, player, "red")
        self.round = 0

    def action(self):
        """
        Called at the beginning of your turn. Based on the current state
        of the game, select an action to play.
        """
        # put your code here

        if (self.player=='blue' and self.round==1):
            action = check_if_steal(self.board,self.player)
        elif (self.player=='red' and self.round == 0): 
            action = ('PLACE', (self.n//2) -1, self.n//2) 

        
        else:
            action = choose_action(self.board, 2)
            # if self.round<=40:
            #     action = choose_action(self.board, 2)
            # else:
            #     action = choose_action(self.board, 4)
            
             

            

        return action
        
    
    
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
        # put your code here
        self.board.update(player,action)
        #self.board.print_grid()
        self.round +=1
        #print("check");print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

        
        
