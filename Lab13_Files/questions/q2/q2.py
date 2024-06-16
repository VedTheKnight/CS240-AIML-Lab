import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def Gale_Shapley(suitor_prefs, reviewer_prefs) -> Dict[str, str]:
    '''
        Gale-Shapley Algorithm for Stable Matching

        Parameters:

        suitor_prefs: dict - Dictionary of suitor preferences
        reviewer_prefs: dict - Dictionary of reviewer preferences

        Returns:

        matching: dict - Dictionary of suitor matching with reviewer
    '''

    matching = {suitor : ' ' for suitor in suitor_prefs.keys()}

    ## TODO: Implement the Gale-Shapley Algorithm
    suitor_list = suitor_prefs.copy()

    reviewer_best = {reviewer : ' ' for reviewer in reviewer_prefs.keys()}

    def checkvalid(s):
        for suitor in s.keys():
            if len(s[suitor])!=0:
                return True
        return False
    

    while checkvalid(suitor_list): #while there exist a possible reviewer for a suitor we continue matching
        counter = len(matching)
        for suitor in matching.keys():
            if(matching[suitor] != ' '):
                counter-=1
        
        if(counter == 0):
            break
        for suitor in suitor_list.keys():
            if(len(suitor_list[suitor]) == 0):
                continue
            if(matching[suitor] != ' '):
                continue
            reviewer_to_offer = suitor_list[suitor][0]
            suitor_list[suitor] = suitor_list[suitor][1:]
            if(reviewer_best[reviewer_to_offer] == ' '):
                reviewer_best[reviewer_to_offer] = suitor
                matching[suitor] = reviewer_to_offer
            else:
                #rto is engaged to m' and m' has higher pref then ignore
                if(reviewer_prefs[reviewer_to_offer].index(reviewer_best[reviewer_to_offer]) < reviewer_prefs[reviewer_to_offer].index(suitor)):
                    continue 
                else:
                    matching[reviewer_best[reviewer_to_offer]] = ' '
                    reviewer_best[reviewer_to_offer] = suitor
                    matching[suitor] = reviewer_to_offer   
            

    ## END TODO
    return matching

def avg_suitor_ranking(suitor_prefs: Dict[str, List[str]], matching: Dict[str, str]) -> float:
    '''
        Calculate the average ranking of suitor in the matching
        
        Parameters:
        
        suitor_prefs: dict - Dictionary of suitor preferences
        matching: dict - Dictionary of matching
        
        Returns:
        
        avg_suitor_ranking: float - Average ranking of suitor
    '''

    avg_suitor_ranking = 0

    ## TODO: Implement the average suitor ranking calculation
    for suitor in matching.keys():
        avg_suitor_ranking += suitor_prefs[suitor].index(matching[suitor]) + 1
    avg_suitor_ranking = avg_suitor_ranking / len(matching)
    ## END TODO

    
    assert type(avg_suitor_ranking) == float

    return avg_suitor_ranking

def avg_reviewer_ranking(reviewer_prefs: Dict[str, List[str]], matching: Dict[str, str]) -> float:
    '''
        Calculate the average ranking of reviewer in the matching
        
        Parameters:
        
        reviewer_prefs: dict - Dictionary of reviewer preferences
        matching: dict - Dictionary of matching
        
        Returns:
        
        avg_reviewer_ranking: float - Average ranking of reviewer
    '''

    avg_reviewer_ranking = 0

    ## TODO: Implement the average reviewer ranking calculation

    for suitor in matching.keys():
        avg_reviewer_ranking += reviewer_prefs[matching[suitor]].index(suitor) + 1
        
    avg_reviewer_ranking = avg_reviewer_ranking/ len(matching)
    ## END TODO

    assert type(avg_reviewer_ranking) == float

    return avg_reviewer_ranking

def get_preferences(file) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    '''
        Get the preferences from the file
        
        Parameters:
        
        file: file - File containing the preferences
        
        Returns:
        
        suitor_prefs: dict - Dictionary of suitor preferences
        reviewer_prefs: dict - Dictionary of reviewer preferences
    '''
    suitor_prefs = {}
    reviewer_prefs = {}

    for line in file:
        if line[0].islower():
            reviewer, prefs = line.strip().split(' : ')
            reviewer_prefs[reviewer] = prefs.split()

        else:
            suitor, prefs = line.strip().split(' : ')
            suitor_prefs[suitor] = prefs.split()
        
    return suitor_prefs, reviewer_prefs


if __name__ == '__main__':

    avg_suitor_ranking_list = []
    avg_reviewer_ranking_list = []

    for i in range(100):
        with open('data/data_'+str(i)+'.txt', 'r') as f:
            suitor_prefs, reviewer_prefs = get_preferences(f)

            # suitor_prefs = {
            #     'A': ['a', 'b', 'c'],
            #     'B': ['c', 'b', 'a'],
            #     'C': ['c', 'a', 'b']
            # }

            # reviewer_prefs = {
            #     'a': ['A', 'C', 'B'],
            #     'b': ['B', 'A', 'C'],
            #     'c': ['B', 'A', 'C']
            # }

            matching = Gale_Shapley(suitor_prefs, reviewer_prefs)

            avg_suitor_ranking_list.append(avg_suitor_ranking(suitor_prefs, matching))
            avg_reviewer_ranking_list.append(avg_reviewer_ranking(reviewer_prefs, matching))

    plt.hist(avg_suitor_ranking_list, bins=10, label='Suitor', alpha=0.5, color='r')
    plt.hist(avg_reviewer_ranking_list, bins=10, label='Reviewer', alpha=0.5, color='g')

    plt.xlabel('Average Ranking')
    plt.ylabel('Frequency')

    plt.legend()
    plt.savefig('q2.png')


    

