import numpy as np

# reads stop words into hash table
def read_stop_words(path) :
    # opens file
    file = open(path, 'r')

    line = file.readline()

    stop_words = dict()

    # goes through each line in file
    while line :
        # read name into hash table
        stop_words[line[:-1]] = 1

        line = file.readline()

    file.close()

    return stop_words

# read data filtered by stop words
def read_data(path, stop_words) :
    # opens file
    file = open(path, 'r')

    data = []

    line = file.readline()

    # goes through each line in file
    while line :
        # reads line, splits into values
        words = line[:-1].split(' ')

        # filters out stop words
        words = list(filter(lambda x: stop_words.get(x) == None, words))
        data.append(words)

        line = file.readline()
    
    return data

# reads all labels from file
def read_labels(path) :
    # opens file
    file = open(path, 'r')

    labels = []

    line = file.readline()

    # goes through each line in file
    while line :
        # reads line
        labels.append(int(line[:-1]))

        line = file.readline()
    
    return np.array(labels)

# read ocr data, data + label
def read_ocr(path) :
    # opens file
    file = open(path, 'r')

    line = file.readline()

    data = []
    labels = []

    # goes through each line in file
    while line :
        # if empty line
        if line.startswith('\t\t') :
            line = file.readline()
            continue
        
        # get start of sequence
        start_index = line.index('m')

        # get data 
        inputs = list(map(int, line[start_index + 1:start_index+ 129]))

        # get label
        label = line[start_index + 130]

        # checks if vowel (0) or consonant (1)
        if label in 'aeiou' :
            label = 0
        else :
            label = 1

        data.append(inputs)
        labels.append(label)

        line = file.readline()

    return (np.array(data), np.array(labels))

# forms vocab from training data
def form_vocab(data) :
    vocab = dict()

    # index of word
    index = 0

    # goes through each word in data
    for line in data :
        for word in line :
            # adds word if havent found, go to next index
            if vocab.get(word) == None :
                vocab[word] = index
                index += 1
    
    return vocab

# converts word data to feature data
def convert_data(data, vocab) :
    new_data = []
    
    # goes through each line in data
    for line in data :
        new_line = [0] * len(vocab)

        # goes through each word in line
        for word in line :
            # makes sure word exists in vocab, then sets value at index of word to 1
            if vocab.get(word) != None :
                new_line[vocab[word]] = 1
        
        new_data.append(new_line)

    return np.array(new_data)

# predicts using sign of value
def predict(x,w):
    return 1 if np.dot(x,w) >= 0 else 0

# calculates accuracy of given model from given test values
def accuracy(x, w, y) :
    correct = 0
    total = x.shape[0]

    # goes through each data point
    for i in range(x.shape[0]) :
        # checks if prediction matches actual output
        if predict(x[i], w) == y[i] :
            correct += 1
    
    # returns % correct
    return correct / total

# trains model from given train and test data
def train_model(train, test, epochs=100, learning_rate = 0.001) :
    # unpacks data and labels
    train_data, train_labels = train
    test_data, test_labels = test

    # weights
    w = np.zeros(shape=train_data.shape[1])

    mistakes_list = []
    accuracy_train_list = []
    accuracy_test_list = []
    w_list = []
    c_list = []
    # goes through each epoch
    for _ in range(epochs) :
        mistakes = 0
        c = 1
        # goes through each data point in train set
        for i in range(train_data.shape[0]) :
            x = train_data[i]
            y = train_labels[i]

            # predicts from x
            y_pred = predict(x, w)
            
            if y != y_pred :
                # updates weights based on prediction
                w = w + (learning_rate * (y-y_pred) * x)

                mistakes += 1

                if _ == epochs - 1 and i != train_data.shape[0] - 1:
                    w_list.append(w)
                    c_list.append(c)
                    c = 1
            else :
                c += 1

        if _ == epochs - 1 :
            w_list.append(w)
            c_list.append(c)
        # saves number of mistakes, accuracy of train and test sets
        mistakes_list.append(mistakes)
        accuracy_train_list.append(accuracy(train_data, w, train_labels))
        accuracy_test_list.append(accuracy(test_data, w, test_labels))
    voted = (w_list, c_list)
    return w, mistakes_list, accuracy_train_list, accuracy_test_list, voted

# gets accuracy using average voted method
def avg_voted(voted, x, y) :
    w_list, c_list = voted
    w = w_list[0] * c_list[0]

    # adds up weights
    for i in range(1, len(w_list)) :
        w += w_list[i] * c_list[i]

    # averages weights
    w /= len(w_list)

    # return accuracy using new weights
    return accuracy(x, w, y)

# prints output of model
def print_output(train, test, output_file, mistakes_list, accuracy_train_list, accuracy_test_list, voted) :
    # print mistakes
    for i, mistake in enumerate(mistakes_list) :
        output_file.write("iteration-" + str(i+1) + " " + str(mistake) + "\n")
    
    output_file.write('\n')

    # print accuracy
    for i in range(len(accuracy_train_list)) :
        output_file.write('iteration-' + str(i+1) + ' training-accuracy ' + str(accuracy_train_list[i]) + ' testing-accuracy ' + str(accuracy_test_list[i]) + '\n')

    output_file.write('\n')

    # get train and test average voted accuracies
    average_voted_train = avg_voted(voted, train[0], train[1])
    average_voted_test = avg_voted(voted, test[0], test[1])

    # print overall testing and training accuracies
    output_file.write('training-accuracy- standard-perceptron ' + str(accuracy_train_list[-1]) + ' averaged-perceptron ' + str(average_voted_train) + '\n')

    output_file.write('\n')

    output_file.write('testing-accuracy- standard-perceptron ' + str(accuracy_test_list[-1]) + ' averaged-perceptron ' + str(average_voted_test) + '\n')

# runs program
def main() :
    # fortune cookie folder
    folder = './fortune-cookie-data/'
    
    # reads stop words
    stop_words = read_stop_words(folder + 'stoplist.txt')

    # reads training data and labels
    train_data = read_data(folder + 'traindata.txt', stop_words)
    train_labels = read_labels(folder + 'trainlabels.txt')

    # reads test data and labels
    test_data = read_data(folder + 'testdata.txt', stop_words)
    test_labels = read_labels(folder + 'testlabels.txt')

    # gets vocab
    vocab = form_vocab(train_data)

    # converts data into features using bag of words
    train_data = convert_data(train_data, vocab)
    test_data = convert_data(test_data, vocab)

    # combines data and labels
    train = (train_data, train_labels)
    test = (test_data, test_labels)

    # trains model for fortune cookie
    w, mistakes_list, accuracy_train_list, accuracy_test_list, voted = train_model(train, test, epochs=20, learning_rate=1)

    output_file = open('output.txt', 'w')

    # print output for first model
    print_output(train, test, output_file, mistakes_list, accuracy_train_list, accuracy_test_list, voted)

    output_file.write('\n')
    output_file.write('\n')

    # read in ocr data
    ocr_folder = './OCR-data/'

    ocr_train = read_ocr(ocr_folder + 'ocr_train.txt')
    ocr_test = read_ocr(ocr_folder + 'ocr_test.txt')

    # train ocr model
    w2, mistakes_list2, accuracy_train_list2, accuracy_test_list2, voted2 = train_model(ocr_train, ocr_test, epochs=20, learning_rate=1)

    # print output for second model
    print_output(ocr_train, ocr_test, output_file, mistakes_list2, accuracy_train_list2, accuracy_test_list2, voted2)

    output_file.close()

# runs program
if __name__ == '__main__' :
    main()