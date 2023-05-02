# counts frequency of each data point
def count_frequency(data: list) -> dict :
    freq_data = dict()

    # goes through each bucket in data list
    for bucket in data :
        # go through each item in bucket
        for item in bucket :
            # add 1 to item count
            if freq_data.get(item) != None:
                freq_data[item] += 1
            else :
                freq_data[item] = 1
    return freq_data

# filters frequency based on support threshold
def filter_freq(s: int, data: dict) -> dict :
    # save only items with value > support threshold
    return dict(filter(lambda x: x[1] >= s, data.items()))

# creates all pairs from a list
def create_pairs(data: list) -> list :
    cand_pairs = []
    
    # go through each element
    for i in range(0,len(data)) :
        # go through every element after first
        for j in range(i+1,len(data)) :
            cand_pairs.append((data[i], data[j]))

    return cand_pairs

# creates all triples from a list
def create_triples(data: list) -> list :
    cand_triples = []

    # go through each element
    for i in range(0,len(data)) :
        # go through each element after first
        for j in range(i+1,len(data)) :
            # go through each element after second
            for k in range(j+1, len(data)) :
                cand_triples.append((data[i], data[j], data[k]))

    return cand_triples

# reads from file, saves market data to list
def read_file(path: str) -> list :
    # opens file
    file = open(path, 'r')
    line = file.readline()
    
    data = []

    # goes through each line
    while line :
        # splits current line into each item, except for last \n
        items = line.split(' ')[:-1]
        
        # save data, go to next line
        data.append(items)
        line = file.readline()

    file.close()
    return data

# finds frequent pairs in the data
def find_cand_pairs(data: list) -> dict :
    freq_pairs = dict()

    # goes through each bucket
    for bucket in data :
        # creates possible pairs from the bucket
        pairs = create_pairs(bucket)

        # goes through each pair, counts them
        for pair in pairs :
            # checks to make sure (i,j) and (j,i) count to same pair
            if freq_pairs.get((pair[0], pair[1])) != None:
                freq_pairs[(pair[0], pair[1])] += 1
            elif freq_pairs.get((pair[1], pair[0])) != None:
                freq_pairs[(pair[1], pair[0])] += 1
            else :
                if pair[0] > pair[1] :
                    freq_pairs[(pair[1], pair[0])] = 1
                else :
                    freq_pairs[(pair[0], pair[1])] = 1

    return freq_pairs

# filters market based on frequent singles
def filter_data_singles(data: list, filtered_singles: dict) -> list :
    # goes through each bucket
    for bucket in data :
        index = 0

        # goes through each item in bucket
        while index < len(bucket) :
            # removes item if not in frequent list
            if filtered_singles.get(bucket[index]) == None :
                del bucket[index]
            else :
                index += 1
    return data

# checks if triple is a possible candidate
def check_cand_triple(freq_pairs: dict, triple: tuple) -> bool :
    # checks (X,Y,Z) has (X,Y), (X,Z), and (Y,Z), also checks for pair swaps
    if freq_pairs.get((triple[0], triple[1])) != None or freq_pairs.get((triple[1], triple[0])) != None :
        if freq_pairs.get((triple[0], triple[2])) != None or freq_pairs.get((triple[2], triple[0])) != None :
            if freq_pairs.get((triple[2], triple[1])) != None or freq_pairs.get((triple[1], triple[2])) != None :
                return True
    return False

# filters out the data based on frequent pairs
def filter_data_pairs(data: list, freq_pairs: dict) -> list :
    # goes through all lines in data
    for bucket in data :
        # creates the possible pairs for the bucket
        pairs = create_pairs(bucket)

        # the items which are in frequent pairs so it can remove
        save_list = []

        # goes through all pairs
        for pair in pairs :
            if freq_pairs.get((pair[0], pair[1])) != None or freq_pairs.get((pair[1], pair[0])) != None :
                if pair[0] not in save_list :
                    save_list.append(pair[0])
                if pair[1] not in save_list :
                    save_list.append(pair[1])
        
        index = 0
        while index < len(bucket) :
            if bucket[index] not in save_list :
                del bucket[index]
            else :
                index += 1
        
        if bucket == [] :
            del bucket
    return data

# gets cand triples
def find_cand_triples(data: list) -> dict :
    cand_triples = dict()

    # goes through each bucket
    for bucket in data :
        # gets all possible triples from current bucket
        triples = create_triples(bucket)

        # goes through each triple
        for triple in triples :
            # sorts triple
            triple = tuple(sorted(triple))

            # saves triple count
            if cand_triples.get(triple) != None :
                cand_triples[triple] += 1
            else :
                cand_triples[triple] = 1
                  
    return cand_triples

# gets the confidence for each pair
def triple_confidence(freq_triples: dict, freq_pairs: dict) -> dict :
    confidence = dict()

    # goes through each pair
    for triple in freq_triples.keys() :
        # gets (X,Y) -> Z, (X,Z) -> Y, (Y,Z) -> X
        pair1 = (triple[0], triple[1])
        pair2 = (triple[0], triple[2])
        pair3 = (triple[1], triple[2])
        amount1 = freq_pairs.get(pair1)
        amount2 = freq_pairs.get(pair2)
        amount3 = freq_pairs.get(pair3)

        # checks if pairs need to be swapped
        if amount1 == None :
            pair1 = (triple[1], triple[0])
            amount1 = freq_pairs.get(pair1)

        if amount2 == None :
            pair2 = (triple[2], triple[0])
            amount2 = freq_pairs.get(pair2)

        if amount3 == None :
            pair3 = (triple[2], triple[1])
            amount3 = freq_pairs.get(pair3)

        triple_amount = freq_triples.get(triple)

        # saves confidence numbers for each pair
        confidence[(pair1, triple[2])] = triple_amount / amount1
        confidence[(pair2, triple[1])] = triple_amount / amount2
        confidence[(pair3, triple[0])] = triple_amount / amount3
    
    # sorts ascending
    sort_second = sorted(confidence.items(), key = lambda x : x[0][0][1])
    sort_first = sorted(sort_second, key = lambda x : x[0][0][0])
    sort_third = sorted(sort_first, key = lambda x : x[0][1])

    # sorts descending
    sorted_trips = sorted(sort_third, key = lambda x : x[1], reverse = True)  

    return sorted_trips

# gets the confidence for each pair
def double_confidence(freq_pairs: dict, freq_singles: dict) -> dict :
    confidence = dict()

    # goes through each pair
    for pair in freq_pairs.keys() :
        # gets X -> Y and Y -> X
        amount1 = freq_singles.get(pair[0])
        amount2 = freq_singles.get(pair[1])

        pair_amount = freq_pairs[pair]

        # saves confidence numbers
        confidence[(pair[0], pair[1])] = pair_amount / amount1
        confidence[(pair[1], pair[0])] = pair_amount / amount2
    
    # sorts ascending
    sort_second = sorted(confidence.items(), key = lambda x : x[0][0])
    sort_first = sorted(sort_second, key = lambda x : x[0][1])
    
    # sorts descending
    sorted_pairs = sorted(sort_first, key = lambda x : x[1], reverse = True)  

    return sorted_pairs

# prints output to file
def print_output(path, triple_conf, double_conf) :
    file = open(path, 'w')
    index = 0

    file.write('OUTPUT A\n')

    # goes through top 5 doubles in double confidences, prints
    for values in double_conf:
        if index == 5 :
            break
        
        output = values[0][0] + ' ' + values[0][1] + ' ' + str(values[1]) + '\n'
        file.write(output)

        index += 1
    
    file.write('OUTPUT B\n')

    index = 0

    # goes through top 5 triples in triple confidences, prints
    for values in triple_conf :
        if index == 5 :
            break

        output = values[0][0][0] + ' ' + values[0][0][1] + ' ' + values[0][1] + ' ' +  str(values[1]) + '\n'
        file.write(output)

        index += 1

    file.close()

def main() :
    # reads data
    data = read_file('browsing-data.txt')
    s = 100

    # gets count of each item, then filters to get frequent items
    freq_data = count_frequency(data)
    filtered_singles = filter_freq(s, freq_data)

    # removes all non frequent items from dataset
    filtered_data = filter_data_singles(data, filtered_singles)

    # gets candidate pairs by counting, then filters them to get the truly frequent pairs
    cand_pairs = find_cand_pairs(filtered_data)
    freq_pairs = filter_freq(s, cand_pairs)
    # removes all non frequent pairs from dataset
    filtered_data = filter_data_pairs(filtered_data, freq_pairs)

    # gets the candidate triples by counting, then filters to get frequent triples
    trips = find_cand_triples(filtered_data)
    frequent_trips = filter_freq(s, trips)

    # gets the confidence for pairs and triples
    triple_conf = triple_confidence(frequent_trips, freq_pairs)
    double_conf = double_confidence(freq_pairs, filtered_singles)

    print_output('output.txt', triple_conf, double_conf)

#runs program
if __name__ == '__main__' :
    main()