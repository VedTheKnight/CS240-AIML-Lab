import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt


''' Do not change anything in this function '''
def generate_random_profiles(num_voters, num_candidates):
    '''
        Generates a NumPy array where row i denotes the strict preference order of voter i
        The first value in row i denotes the candidate with the highest preference
        Result is a NumPy array of size (num_voters x num_candidates)
    '''
    return np.array([np.random.permutation(np.arange(1, num_candidates+1)) 
            for _ in range(num_voters)])


def find_winner(profiles, voting_rule):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        voting_rule is one of [plurality, borda, stv, copeland]
        In STV, if there is a tie amongst the candidates with minimum plurality score in a round, then eliminate the candidate with the lower index
        For Copeland rule, ties among pairwise competitions lead to half a point for both candidates in their Copeland score

        Return: Index of winning candidate (1-indexed) found using the given voting rule
        If there is a tie amongst the winners, then return the winner with a lower index
    '''

    winner_index = None
    num_voters, num_candidates = profiles.shape
    # TODO

    if voting_rule == 'plurality':
        votes = {(i+1) : 0 for i in range(num_candidates)}
        for voter_profile in profiles:
            votes[voter_profile[0]]+=1
        
        max_votes = 0
        best_candidate = -1
        for candidate in votes.keys():
            if(votes[candidate] > max_votes):
                max_votes = votes[candidate]
                best_candidate = candidate

        winner_index = best_candidate

    elif voting_rule == 'borda':
        candidate_borda = np.array([0]*num_candidates)
        for i in range(len(profiles)):
            voter_profile = profiles[i]
            voter_borda = [0]*num_candidates
            for i in range(num_candidates):
                voter_borda[voter_profile[i] - 1] = num_candidates - (i+1) #preference is index+1 and need to do candidate - 1 to index into voter borda
            voter_borda = np.array(voter_borda)
            candidate_borda = candidate_borda + voter_borda
        
        max_borda_score = 0
        for i in range(len(candidate_borda)):
            if(candidate_borda[i] > max_borda_score):
                max_borda_score = candidate_borda[i]
                winner_index = i+1
        

    elif voting_rule == 'stv':
        active_candidates = [i+1 for i in range(num_candidates)]
        for round in range(num_candidates - 1):
            votes = {i : 0 for i in active_candidates}
            for voter_profile in profiles:
                ptr = 0
                while(voter_profile[ptr] not in active_candidates):
                    ptr+=1
                votes[voter_profile[ptr]]+=1
            
            min_votes = float('inf')
            worst_canditate = -1
            for candidate in votes.keys():
                if(votes[candidate] < min_votes):
                    min_votes = votes[candidate]
                    worst_canditate = candidate

            active_candidates.remove(worst_canditate)
        winner_index = active_candidates[0]

    elif voting_rule == 'copeland':
        copeland_mapping = {(i+1) : 0 for i in range(num_candidates)}
        
        for i in range(num_candidates):
            for j in range(num_candidates):
                if(i == j):
                    continue
                c1 = i+1
                c2 = j+1

                c1_pref = 0
                c2_pref = 0

                for voter_profile in profiles:
                    voter_profile_list = voter_profile.tolist()
                    # print(voter_profile_list)
                    if(voter_profile_list.index(c1) < voter_profile_list.index(c2)):
                        c1_pref+=1
                    if(voter_profile_list.index(c2) < voter_profile_list.index(c1)):
                        c2_pref+=1

                if(c1_pref > c2_pref):
                    copeland_mapping[c1] += 1
                elif(c2_pref > c1_pref):
                    copeland_mapping[c2] += 1
                else:
                    copeland_mapping[c2] += 0.5
                    copeland_mapping[c1] += 0.5

        max_copeland_score = 0
        for candidate in copeland_mapping.keys():
            if(copeland_mapping[candidate] > max_copeland_score):
                max_copeland_score = copeland_mapping[candidate]
                winner_index = candidate
    # END TODO

    return winner_index


def find_winner_average_rank(profiles, winner):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        winner is the index of the winning candidate for some voting rule (1-indexed)

        Return: The average rank of the winning candidate (rank wrt a voter can be from 1 to num_candidates)
    '''

    average_rank = 0

    # TODO

    for voter_profile in profiles:
        voter_profile_list = voter_profile.tolist()
        average_rank += voter_profile_list.index(winner) + 1

    average_rank = average_rank / len(profiles)

    # END TODO

    return average_rank


def check_manipulable(profiles, voting_rule, find_winner):
    '''
        profiles is a NumPy array with each row denoting the strict preference order of a voter
        voting_rule is one of [plurality, borda, stv, copeland]
        find_winner is a function that takes profiles and voting_rule as input, and gives the winner index as the output
        It is guaranteed that there will be at most 8 candidates if checking manipulability of a voting rule

        Return: Boolean representing whether the voting rule is manipulable for the given preference profiles
    '''


    # TODO
    manipulable = False
    num_voters, num_candidates = profiles.shape
    # print(num_candidates)
    for voter_index in range(num_voters):
        # Generate all possible manipulations of voter's vote
        original_winner = find_winner(profiles, voting_rule)
        original_profile_list = profiles[voter_index].copy()
        original_profile_list = original_profile_list.tolist()

        for permutation in itertools.permutations(range(1, 1+num_candidates)):
            manipulated_profile = profiles.copy()

            permutation = np.array(list(permutation))
            permutation = permutation.reshape(profiles[voter_index].shape)

            manipulated_profile[voter_index] = permutation

            # Check if manipulation changes the winner
            
            manipulated_winner = find_winner(manipulated_profile, voting_rule)

            if original_profile_list.index(original_winner) > original_profile_list.index(manipulated_winner):
                manipulable = True
                break

        if manipulable:
            break
    

    # END TODO

    return manipulable


if __name__ == '__main__':
    np.random.seed(420)

    num_tests = 200
    voting_rules = ['plurality', 'borda', 'stv', 'copeland']

    average_ranks = [[] for _ in range(len(voting_rules))]
    manipulable = [[] for _ in range(len(voting_rules))]
    for _ in tqdm(range(num_tests)):
        # Check average ranks of winner
        num_voters = np.random.choice(np.arange(80, 150))
        num_candidates = np.random.choice(np.arange(10, 80))
        profiles = generate_random_profiles(num_voters, num_candidates)

        for idx, rule in enumerate(voting_rules):
            winner = find_winner(profiles, rule)
            avg_rank = find_winner_average_rank(profiles, winner)
            average_ranks[idx].append(avg_rank / num_candidates)

        # Check if profile is manipulable or not
        num_voters = np.random.choice(np.arange(10, 20))
        num_candidates = np.random.choice(np.arange(4, 8))
        profiles = generate_random_profiles(num_voters, num_candidates)
        
        for idx, rule in enumerate(voting_rules):
            manipulable[idx].append(check_manipulable(profiles, rule, find_winner))


    # Plot average ranks as a histogram
    for idx, rule in enumerate(voting_rules):
        plt.hist(average_ranks[idx], alpha=0.8, label=rule)

    plt.legend()
    plt.xlabel('Fractional average rank of winner')
    plt.ylabel('Frequency')
    plt.savefig('average_ranks.jpg')
    
    # Plot bar chart for fraction of manipulable profiles
    manipulable = np.sum(np.array(manipulable), axis=1)
    manipulable = np.divide(manipulable, num_tests)
    plt.clf()
    plt.bar(voting_rules, manipulable)
    plt.ylabel('Manipulability fraction')
    plt.savefig('manipulable.jpg')