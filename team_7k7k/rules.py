def check_if_steal(board,player):
    grid = board.grid
    size = board.n
    center = size//2  #2
    # if size of board is even, then check poisiton
    """ (r-1, q) (r-1, q+1) 
            (r, q-1) (r, q+1)
            (r+1, q-1) (r+1, q) 
    """
    
    if(grid[center][center]!=0 or grid[center][center+1]!=0 
    or grid[center+1][center]!=0 or grid[center][center-1]!=0 
    or grid[center-1][center]!=0 or grid[center+1][center-1]!=0
    or grid[center-1][center+1]!=0):
        # steal(grid,center,player)  
        return ("STEAL", )
    return("PLACE",center,center)




def find_longest(board, player):
    max = 0
    count = {}
    if (player == "blue"):
        grid = board.grid
        for r in range(board.n):
            length = 0
            for q in range(board.n):
                if (grid[r][q]=="blue"):
                    length+=1
                    if (q==board.n -1 and length!=0):
                        if(length>max):
                            max = length
                            count[max] = 1
                        elif (length==max and length!=0):
                            count[max] +=1

                    continue
                else:
                    if(length>max):
                        max = length
                        count[max] = 1
                    elif (length==max and length!=0):
                        count[max] +=1
                    length = 0

    else:
        grid = board.grid
        for q in range(board.n):
            length = 0
            for r in range(board.n):
                if (grid[r][q]=="red"):
                    length+=1
                    if (r==board.n -1 and length!=0):
                        if(length>max):
                            max = length
                            count[max] = 1
                        elif (length==max and length!=0):
                            count[max] +=1
                    continue
                else:
                    if(length>max):
                        max = length
                        count[max] = 1
                    elif (length == max and length!=0):
                        count[max] +=1
                    length = 0
    if (max==0):
        return max
        
    result = count[max] * board.weight[max]
    #print(player, max, ":", count[max])
    return result
                
  