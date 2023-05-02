import csv
from sklearn import feature_extraction
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from textatistic import Textatistic
import textatistic
import string
import random
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
# reads data and labels
def read_data(path) :
    file = open(path, 'r')

    reader = csv.reader(file)

    data = []
    labels = []

    for i, row in enumerate(reader) :
        if i == 0 : continue
        line = ''
        for words in row[1:] :
            line += words

        data.append(line.lower())
        labels.append(0 if row[0] == 'ham' else 1)

    return data, labels

# reads stop words from list
def read_stop_words(path) :
    file = open(path, 'r')

    line = file.readline()

    data = []

    while line != '' :
        data.append(line[:-1])

        line = file.readline()
    
    return data


# trains data
def train(x,y) :
    train = (x[0:len(x)//2], y[0:len(x)//2])
    test = (x[len(x)//2:], y[len(x)//2:])
    train_x = np.array(train[0])
    train_y = np.array(train[1])
    test_x = np.array(test[0])
    test_y = np.array(test[1])

    mlp = MLPClassifier(activation='tanh', solver='adam', epsilon=1e-3, batch_size=10, max_iter=1000, hidden_layer_sizes=(240,120),learning_rate='adaptive',verbose=1)
    mlp = mlp.fit(train_x, train_y)
    #print_curve(mlp, train_x, train_y)
    # uncomment for sklearn feature selection

    #lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_x, train_y)
    #model = SelectFromModel(lsvc, prefit=True)
    #X_new = model.transform(train_x)
    #test_x_new = model.transform(test_x)
    #mlp_new = MLPClassifier(activation='tanh', solver='adam', epsilon=1e-11, batch_size=40, max_iter=10000, hidden_layer_sizes=(240,120))
    #mlp_new = mlp_new.fit(X_new, train_y)
    #preds2 = mlp_new.predict(test_x_new)
    #print(confusion_matrix(test_y, preds2))

    # prints scores
    print(mlp.score(test_x, test_y))
    print(mlp.score(train_x, train_y))
    preds = mlp.predict(test_x)
    
    c = confusion_matrix(test_y, preds)
    print(confusion_matrix(test_y, preds))
    

    return mlp, c
from io import StringIO
import sys
def print_curve(mlp, x, y) :
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    mlp.fit(x,y)
    sys.stdout = old_stdout
    loss_history = mystdout.getvalue()
    loss_list = []
    for line in loss_history.split('\n'):
        if 'Stopping' in line :
            break
        loss_list.append(float(line[line.index('=') + 2:]))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    #plt.savefig("warmstart_plots/pure_LogRes:"+".png")
    plt.xlabel("Time in epochs")
    plt.ylabel("Loss")
    plt.show()
    x=1
    

# used to remove punctuations
def preprocess(data) :
    for i in range(len(data)) :
        line = list(filter(lambda x: x not in string.punctuation, data[i]))
        line = "".join(line)
        data[i] = line

num_features = 6

def main() :
    path = './spam.csv'
    stop_path = './stoplist.txt'
    data, labels = read_data(path)
    #preprocess(data)
    stop_words = read_stop_words(stop_path)

    # feature vectors
    vectorizer = feature_extraction.text.CountVectorizer(stop_words='english')
    tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english')

    vectorizer.fit(data[0:len(data)//2])
    tfidf.fit(data[0:len(data)//2])

    print(len(data))

    vector = vectorizer.transform(data)
    a = vector.toarray()
    b = tfidf.transform(data).toarray()
    
    # used to find best model params, features
    #find_model_params(a,b,data,labels)

    # train best params
    train(a, labels)
    
# finds best features + hypers
def find_model_params(a, b, data, labels) :
    all_data = []

    # add readability features
    for i, sentence in enumerate(data) :
        readability = 0
        try :
            readability = Textatistic(sentence)
        except :
            while sentence.endswith(" etc") : # breaks textastistic
                sentence = sentence[:-4]
            sentence += '.'
            try :
                readability = Textatistic(sentence)
            except :
                readability = Textatistic('time.') # add random junk word if empty

        dale = readability.dalechall_score
        smog = readability.smog_score
        flesch = readability.flesch_score
        fleshk = readability.fleschkincaid_score
        fog = readability.gunningfog_score
        notchale = readability.notdalechall_count
        all_data.append([dale, smog, flesch, fleshk, fog, notchale])
    
    
    for i in range(0,len(data)) :
        all_data[i] += a[i].tolist()

    # was used to find best params
    temp = list(zip(all_data, labels))
    #random.shuffle(temp)
    all_data, labels = zip(*temp)
    validation, all_data = all_data[:len(all_data)//8], all_data[len(all_data)//8:]
    val_labels, all_labels = labels[:len(labels)//8], labels[len(labels)//8:]
    find_best_model(validation, val_labels)
    #train_rnn(a, labels)
    train(all_data, labels)

# finds best features, hypers
def find_best_model(x, y) :
    x = np.array(x)
    y = np.array(y)
    best_fp = 10000
    best_index = 0

    # goes through, removes each feature, trains model
    for feature in range(num_features) :
        cur_y = y
        cur_x = np.delete(x, feature, axis=1)
        train = (cur_x[0:len(cur_x)//2], cur_y[0:len(cur_y)//2])
        test = (cur_x[len(cur_x)//2:], cur_y[len(cur_y)//2:])
        train_x = np.array(train[0])
        train_y = np.array(train[1])
        test_x = np.array(test[0])
        test_y = np.array(test[1])
        mlp = MLPClassifier(activation='tanh', solver='adam', epsilon=1e-11, batch_size=40, max_iter=10000, hidden_layer_sizes=(240,120))
        
        mlp.fit(train_x, train_y)

        print(mlp.score(test_x, test_y))
        preds = mlp.predict(test_x)
        tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()

        # gets best false positive 
        if fp < best_fp :
            best_fp = fp
            best_index = feature

    # train with no feature removal
    train = (x[0:len(x)//2], y[0:len(y)//2])
    test = (x[len(x)//2:], y[len(y)//2:])
    train_x = np.array(train[0])
    train_y = np.array(train[1])
    test_x = np.array(test[0])
    test_y = np.array(test[1])

    
    mlp = MLPClassifier(activation='tanh', solver='adam', epsilon=1e-11, batch_size=40, max_iter=10000, hidden_layer_sizes=(240,120))
    mlp.fit(train_x, train_y)
    print(mlp.score(test_x, test_y))
    preds = mlp.predict(test_x)
    tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()

    # compare best removal vs no removal
    if fp < best_fp :
        best_fp = fp
        best_index = -1
    else :
        train_x = np.delete(train_x, feature, axis=1)
        test_x = np.delete(test_x, feature, axis=1)
    print(best_fp, best_index)

    rates = ['adaptive', 'invscaling', 'constant']
    epsilons = [1e-6,1e-7,1e-8]
    batches = [10, 50, 100]

    max = [0,0,0]
    maxscore = 0

    # try each hyperparam, save best
    for rate in rates :
        for epsilon in epsilons :
            for batch in batches :
                mlp = MLPClassifier(activation='tanh', solver='adam', epsilon=epsilon, batch_size=batch, max_iter=5000, hidden_layer_sizes=(240,120), learning_rate=rate)
                mlp.fit(train_x, train_y)
                score = mlp.score(test_x, test_y)
                if score > maxscore :
                    maxscore = score
                    max = [rate, epsilon, batch]


    return best_index, max


if __name__ == '__main__' :
    main()