#!/usr/bin/python
import sys

import numpy as np
import itertools



def intersect(a, b):

    """
    Finds the intersection of two lists.

    Args:
    a (list): The first list.
    b (list): The second list.

    Returns:
    list: A list containing the elements common to both input lists.
    """

    return list(set(a) & set(b))

def make_util_matrix(num_players, strategy, util_list):

    """
    Constructs a utility matrix based on the number of players, their strategies, and a list of utilities.

    Args:
    num_players (int): The number of players.
    strategy (list): A list containing the number of strategies each player has.
    util_list (list): A list containing utilities for each combination of strategies.

    Returns:
    numpy.ndarray: A utility matrix representing the game.
    """

    tup_list = []
    for i in range(0, len(util_list), num_players):
        temp_list =  []
        for j in range(num_players):
            temp_list.append(util_list[i+j])

        tup_list.append(tuple(temp_list))  
    temp_str = "float"
    temp_str2 = ",float"*(num_players-1)
    string = temp_str+temp_str2
    dt = np.dtype(string)
    data = np.array(tup_list, dtype=dt)
    tup = tuple(x for x in strategy)
    util_matrix = data.reshape(tup[::-1])
    util_matrix = np.transpose(util_matrix)

    return util_matrix

def make_allpermut(num_players, strategy):

    """
    Args:
    num_players (int): The number of players.
    strategy (list): A list containing the number of strategies each player has.

    Returns:
    list: A list of all possible permutations of strategies.
    """

    somelists=[]
    temp=[[]  for i in range(num_players)] 
    for k in range(num_players):
        temp[k]=[i for i in range(strategy[k])]
        somelists.append(temp[k])

    allpermut=list(itertools.product(*somelists))#generates all the possible states
    for i in range(len(allpermut)):
        allpermut[i]=list(allpermut[i])

    return allpermut

def check(array, target, forstrat, num_players, strategy, util_matrix):
    #get values from allpermut, check for comparison in target, if this val itself is max return 1 ,else return 0

    """
    Checks if a given strategy is a best response to a target strategy for all players.

    Args:
    array (list): A list of indices representing strategies to be compared.
    target (int): The index of the target strategy.
    forstrat (list): The current strategy being evaluated.
    num_players (int): The number of players.
    strategy (list): A list containing the number of strategies each player has.
    util_matrix (numpy.ndarray): The utility matrix representing the game.

    Returns:
    int: 1 if the given strategy is a best response, otherwise 0.
    """

    allpermut=make_allpermut(num_players, strategy)

    tf=0
    getindex=[]
    for i in range(len(array)):
        getindex.append(allpermut[array[i]])
    for i in range(len(getindex)):
        if util_matrix[tuple(getindex[i])][target]<=util_matrix[tuple(forstrat)][target]:
            tf+=1
    if tf==len(array):        
        return 1
    else:
        return 0
    

def sel_index(player, args, multiplier, num_players):

    """
    Selects the index in the game data based on the player and their chosen strategies.

    Args:
    player (int): The player for whom the index is calculated.
    args (list): A list containing the chosen strategies of all players.
    multiplier (list): Cumulative products of the list with number of strategies.
    num_players (int): The number of players.

    Returns:
    int: The calculated index in the game data.
    """

    result = 0
    i = 0
    for arg in args:
        result = result + (arg * multiplier[i])
        i = i + 1
    result = result * num_players
    result += player
    return result


def find_strongly_dominant_eq(gamedata, playerno, totalplayer, topplayer, strategies,
                              multiplier, num_players, strategyarr = [], eqindex = -1):

    """
    Finds strongly dominant strategy equilibrium for a given player.

    Args:
    gamedata (list): The payoff matrix in Gambit nfg format 
    playerno (int): The index of the player for whom the equilibrium strategy is sought.
    totalplayer (list): A list of remaining players to consider.
    topplayer (int): The index of the top player in the list of remaining players.
    strategies (list): List containing the number of strategies of each player.
    multiplier (list): Cumulative product of the "strategies" list, used to find the index of
                       a player's utility within a strategy (using the sel_index function).
    num_players (int): The number of players.    
    strategyarr (list, optional): The list of strategies chosen so far. Defaults to [].
    eqindex (int, optional): The index of the equilibrium strategy found so far. Defaults to -1.

    Returns:
    int: The index of the equilibrium strategy or -sys.maxsize if none exists.
    """

    if len(totalplayer) >= 1:  # If there are remaining players to consider
        cur_player = totalplayer[0]  # Select the current player
        temp = 0
        totalplayer = totalplayer[1:]  # Remove the current player from the list of remaining players
        for strategy in range(strategies[cur_player]):  # Iterate over all possible strategies for the current player
            temparray = strategyarr[:]  # Create a copy of the current list of chosen strategies
            temparray.append(strategy)  # Append the current strategy to the list
            temp = find_strongly_dominant_eq(gamedata, playerno, totalplayer, topplayer, strategies, 
                                             multiplier, num_players, temparray, eqindex)  # Recursively call the function with updated parameters
            if temp == -sys.maxsize:  # If no strongly dominant equilibrium exists
                return temp  # Return -sys.maxsize
            else:
                eqindex = temp  # Update the equilibrium index with the returned value

        return temp  # Return the equilibrium index after evaluating all strategies for the current player
    

    else:                   # If there are no remaining players to consider
        max_payoff = -sys.maxsize  # Initialize the maximum payoff
        max_index = -1  # Initialize the index corresponding to the maximum payoff
        other_payoffs = []  # List to store payoffs of other strategies


        ##########        TASK 1           ###############


        ##########     Write Code          ###############
        # Calculate the payoffs of other strategies for comparison
        strategyarr.insert(playerno, 0)
        for strat in range(strategies[playerno]):  # Iterate over all strategies for the current player

            strategyarr[playerno]=strat
            other_strat_index = sel_index(playerno,strategyarr, multiplier, num_players)  # Calculate the index of the other strategy in the payoff matrix
            other_payoff = gamedata[other_strat_index]  # Get the payoff for the other strategy

            if other_payoff > max_payoff:
                max_payoff = other_payoff
                max_index = strat
            else:
                other_payoffs.append(other_payoff)  # Add the payoff to the list of other payoffs

            

        if max_payoff in other_payoffs:  # If the maximum payoff is found in the list of other payoffs
            return -sys.maxsize  # Return -sys.maxsize indicating no strongly dominant equilibrium
        if eqindex == -1:  # If no equilibrium index has been found yet
            eqindex = max_index # Update the equilibrium index with the index of the maximum payoff
        elif eqindex != max_index:  # If the equilibrium index does not match the index of the maximum payoff
            return -sys.maxsize  # Return -sys.maxsize indicating no strongly dominant equilibrium
        
        return eqindex


def find_weakly_dominant_eq(gamedata, playerno, totalplayer, topplayer, strategies, 
                            multiplier, num_players, eqindex, strategyarr = []):

    """
    Finds weakly dominant equilibrium strategy for a given player.

    Args:
    gamedata (list): The payoff matrix in Gambit nfg format 
    playerno (int): The index of the player for whom the equilibrium strategy is sought.
    totalplayer (list): A list of remaining players to consider.
    topplayer (int): The index of the top player in the list of remaining players.
    strategies (list): List containing the number of strategies of each player.
    multiplier (list): Cumulative product of the "strategies" list, used to find the index of
                       a player's utility within a strategy (using the sel_index function).
    num_players (int): The number of players.
    eqindex (int or list): The index/es of the equilibrium strategy found so far.
    strategyarr (list, optional): The list of strategies chosen so far. Defaults to [].

    Returns:
    tuple: The index of the equilibrium strategy and the updated equilibrium index list.
    """

    if len(totalplayer) >= 1:
        cur_player = totalplayer[0]
        temp = 0
        totalplayer = totalplayer[1:]
        for strategy in range(strategies[cur_player]):
            temparray = strategyarr[:]
            temparray.append(strategy)
            temp, eqindex = find_weakly_dominant_eq(gamedata, playerno, totalplayer, topplayer, strategies,
                                                    multiplier, num_players, eqindex, temparray)
            if temp == -sys.maxsize:
                return temp, eqindex

        return temp, eqindex

    else:
        
        ##########     TASK 2         ###############
        max_payoff = -sys.maxsize  # Initialize the maximum payoff
        new_eqindex = []  # List to store payoffs of other strategies

        strategyarr.insert(playerno, 0)
        for strat in range(strategies[playerno]):  # Iterate over all strategies for the current player
            
            strategyarr[playerno]=strat
            other_strat_index = sel_index(playerno,strategyarr, multiplier, num_players)  # Calculate the index of the other strategy in the payoff matrix
            other_payoff = gamedata[other_strat_index]  # Get the payoff for the other strategy

            if other_payoff > max_payoff:
                max_payoff = other_payoff

        for strat in range(strategies[playerno]):  # Iterate over all strategies for the current player

            strategyarr[playerno]=strat
            other_strat_index = sel_index(playerno,strategyarr, multiplier, num_players)  # Calculate the index of the other strategy in the payoff matrix
            other_payoff = gamedata[other_strat_index]  # Get the payoff for the other strategy

            if other_payoff == max_payoff:
                new_eqindex.append(strat)

        if eqindex == [-1]:
            eqindex = new_eqindex
        else:
            eqindex = intersect(eqindex, new_eqindex)

        ##########     Write Code          ###############


        if not eqindex:
            return -sys.maxsize, eqindex
        else:
            return eqindex[0], eqindex
        

def psne_gen(num_players, strategy, util_matrix):

    """
    Finds Pure Nash equilibrium strategies.

    Args:
    num_players (int): The number of players.
    strategy (list): A list containing the number of strategies each player has.
    util_matrix (numpy.ndarray): The utility matrix representing the game.

    Returns:
    list: A list of Pure Strategy Nash Equilibriums.
    """
    psnelist=[]


    ##########     TASK 3         ###############

    # Generate all possible strategy combinations
    def generate_combinations(strategy, current=[], index=0, result=[]):
        if index == len(strategy):
            result.append(current.copy())
            return

        for i in range(strategy[index]):
            current.append(i)
            generate_combinations(strategy, current, index + 1, result)
            current.pop()

        return result
            

    strategy_combinations = generate_combinations(strategy)

    for strat_comb in strategy_combinations:
        is_nash_equilibrium = True
        for player in range(num_players):
            player_strat = strat_comb[player]
            # Check if no player can increase their utility by unilaterally changing strategy
            for other_strat in range(strategy[player]):
                if(other_strat == player_strat):
                    continue

                if util_matrix[tuple(strat_comb)][player] < util_matrix[tuple(list(strat_comb[:player]) + [other_strat] + list(strat_comb[player+1:]))][player]:
                    is_nash_equilibrium = False
                    break
            if not is_nash_equilibrium:
                break
        if is_nash_equilibrium:
            psnelist.append(list(strat_comb))

    ##########     Write Code          ###############



      
    return psnelist 


def msne_gen(num_players, strategy, util_matrix):

    """
    Finds Mixed Nash equilibrium strategies.

    Args:
    num_players (int): The number of players.
    strategy (list): A list containing the number of strategies each player has.
    util_matrix (numpy.ndarray): The utility matrix representing the game.

    Returns:
    list: The MSNE in the form [[p_1*, p_2*], [q_1*, q_2*]].
    """
    msne=[]


    ##########     TASK 4         ###############
    
    s1 = (0,0); s2 = (0,1); s3 = (1,0); s4 = (1,1)
    a1 = util_matrix[s1][0]; a2 = util_matrix[s1][1]
    b1 = util_matrix[s2][0]; b2 = util_matrix[s2][1]
    c1 = util_matrix[s3][0]; c2 = util_matrix[s3][1]
    d1 = util_matrix[s4][0]; d2 = util_matrix[s4][1]

    p = (d2 - c2) / (a2 + d2 - b2 - c2) 
    q = (d1 - b1) / (a1 + d1 - b1 - c1)

    msne = [[p, 1-p], [q, 1- q]]
    ##########     Write Code          ###############
      
    return msne





if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please pass the name of the game file to be analyzed")
    f = open(sys.argv[1], "r")
    gameinfo = f.readline()
    data = f.readline().split(" ")
    data = data[data.index("{") + 1: data.index("}\n")]
    data = data[data.index("{") + 1:]
    num_players = len(data)
    strategies = list(map(int, data))
    # print(strategies)
    multiplier = []
    temp = 1
    for i in range(len(strategies)):
        multiplier.append(temp)
        temp = temp * strategies[i]

    f.readline()
    data = f.readline().split(" ")
    gamedata = list(map(int, data))
    # print(gamedata)



    ###############     Equilibrium     ###############
    playerslist = list(range(num_players))
    return_value = -1
    strong_eq = []
    for i in range(num_players):
        tempplayerlist = playerslist[:]
        tempplayerlist.remove(i)
        # Function find_strongly_dominant_eq(...) called.
        value = find_strongly_dominant_eq(gamedata, i, tempplayerlist, tempplayerlist[0], strategies, multiplier, num_players)
        # print("sdse ",value)
        if value == -sys.maxsize:
            print("No Strongly Dominant Strategy Equilibrium exists\n")
            return_value = 0
            break
        else:
            strong_eq.append(value)
    if return_value == -1:
        print(f"Strongly Dominant Strategy Equilibrium (in order of P1, P2, ... , Pn) is: {strong_eq}\n")
    else:
        min_eq_list = []
        for i in range(num_players):
            tempplayerlist = playerslist[:]
            tempplayerlist.remove(i)
            result_index = [-1]
            value, result_index = find_weakly_dominant_eq(gamedata, i, tempplayerlist, tempplayerlist[0], strategies, multiplier, num_players, result_index)
            if value == -sys.maxsize or len(result_index) == strategies[i]:
                print("No Weakly Dominant Strategy Equilibrium exists as well\n")
                return_value = -2
                break
            else:
                min_eq_list.append(result_index)

        if return_value != -2:
            print(f"Weakly Dominant Strategy Equilibrium(s) is (are): {min_eq_list}\n")

    util_matrix = make_util_matrix(num_players, strategies, gamedata)
    psnelist = psne_gen(num_players, strategies, util_matrix)
    if len(psnelist) == 0:
        print("No Pure Strategy Nash Equilibrium exists either")
    else:
        print(f"PSNEs: {psnelist}")

    if len(psnelist) % 2 == 0 and num_players == 2 and strategies[0] == 2 and strategies[1] == 2:
        msne = msne_gen(num_players, strategies, util_matrix)
        print(f"\nMSNE: {msne}")