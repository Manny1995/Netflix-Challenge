import numpy as np
from sklearn.metrics import (
    mean_absolute_error
)

from algos import (
    cosine_filtering,
    pearson_filtering,
    jaccard_filtering,
    centered_jaccard,
    combined_filtering,
    centered_cosine,
    adjusted_cosine_filtering,
    blended_model
)

is_amp = 0
is_iuf = 0

def train_users(users, train_file):
    training = open(train_file, 'r')
    training = training.read().strip().split('\n')
    for i, line in enumerate(training):
        users[i] = [int(x) for x in line.split()]

def test_dataset(users, datasetName):
    dataset = open(datasetName, 'r').read().strip().split('\n')
    dataset = [data.split() for data in dataset]
    dataset = [[int(e) for e in data] for data in dataset]
    
    curUser = dataset[0][0]
    curPredict = []
    curFrame = [0] * 1000

    batch_sol = []

    for userId, movieId, rating in dataset:
        userId-=1
        movieId-=1
        if userId != curUser:
            processUser(users, curFrame, curPredict, userId, batch_sol)
            curUser = userId
            curFrame = [0] * 1000
            curPredict = []
        
        if rating == 0:
            curPredict.append(movieId)
        else:
            curFrame[movieId] = rating


    processUser(users, curFrame, curPredict, userId, batch_sol)

    return batch_sol

def test_dataset_item(users, datasetName):

    movies = np.transpose(users)

    dataset = open(datasetName, 'r').read().strip().split('\n')
    dataset = [data.split() for data in dataset]
    dataset = [[int(e) for e in data] for data in dataset]
    
    curUser = dataset[0][0]
    curFrame = [0] * 1000
    movie_ids = []

    batch_sol = []

    for userId, movieId, rating in dataset:
        userId-=1
        movieId-=1
        if userId != curUser:
            processUser_item(users, movies, curFrame, movie_ids, userId, movieId, batch_sol)
            curUser = userId
            curFrame = [0] * 1000
            movie_ids = []
        
        if rating == 0:
            movie_ids.append(movieId)
        else:
            curFrame[movieId] = rating


    processUser_item(users, movies, curFrame, movie_ids, userId, movieId, batch_sol)

    return batch_sol


def processUser(users, activeUser, predictList, userId, batch_sol):
    global is_amp
    global is_iuf

    print("processingUser %s", userId+1);
    ratingList = []
    for i in range(0, len(predictList)):

       # sol = cosine_filtering(users, activeUser, predictList[i], is_amp) 
        sol = jaccard_filtering(users, activeUser, predictList[i], 0)

        #sol = cosine_filtering(users, activeUser, predictList[i], 0)
        activeUser[predictList[i]] = sol
        ratingList.append(sol)

        new_entry = []
        new_entry.append(userId + 1)
        new_entry.append(predictList[i] + 1)
        new_entry.append(sol)

        batch_sol.append(new_entry)



def processUser_item(users, movies, activeUser, predictList, userId, itemId, batch_sol):
    print("processingUser item %s", userId+1);
    ratingList = []
    for i in range(0, len(predictList)):
        sol = adjacent_cosine_filtering(users, movies, activeUser, predictList[i], 1)
        activeUser[predictList[i]] = sol
        ratingList.append(sol)

        new_entry = []
        new_entry.append(userId + 1)
        new_entry.append(predictList[i] + 1)
        new_entry.append(sol)

        batch_sol.append(new_entry)

def write_to_file(batch_sol, dataset_name):
    print "writing to file"
    dataset_name = dataset_name.replace(".txt", "_results.txt")
    target = open(dataset_name, "w")
    for userId, movieId, rating in batch_sol:
        line = str(userId) + "\t" + str(movieId) + "\t" + str(rating) + "\n"
        target.write(line)

def cos_tests(users):
    results = test_dataset(users, 'tiny5.txt')
    write_to_file(results, 'tiny5_cosmooth.txt')
    results = test_dataset(users, 'test5.txt')
    write_to_file(results, 'test5_cossmooth.txt')
    results = test_dataset(users, 'test10.txt')
    write_to_file(results, 'test10_cossmooth.txt')    
    results = test_dataset(users, 'test20.txt')
    write_to_file(results, 'test20_cossmooth.txt')  

def centered_cos_tests(users):
    results = test_dataset(users, 'tiny5.txt')
    write_to_file(results, 'tiny5_centered_cos.txt')
    results = test_dataset(users, 'test5.txt')
    write_to_file(results, 'test5_centered_cos.txt')
    results = test_dataset(users, 'test10.txt')
    write_to_file(results, 'test10_centered_cos.txt')    
    results = test_dataset(users, 'test20.txt')
    write_to_file(results, 'test20_centered_cos.txt')  


def blend_tests(users):
    results = test_dataset(users, 'tiny5.txt')
    write_to_file(results, 'tiny5_blend_64.txt')
    results = test_dataset(users, 'test5.txt')
    write_to_file(results, 'test5_blend_64.txt')
    results = test_dataset(users, 'test10.txt')
    write_to_file(results, 'test10_blend_64.txt')    
    results = test_dataset(users, 'test20.txt')
    write_to_file(results, 'test20_blend_64.txt')

def item_iuf(users):
    results = test_dataset_item(users, 'tiny5.txt')
    write_to_file(results, 'tiny5_iufitem.txt')
    results = test_dataset_item(users, 'test5.txt')
    write_to_file(results, 'test5_iufitem.txt')
    results = test_dataset_item(users, 'test10.txt')
    write_to_file(results, 'test10_iufitem.txt')    
    results = test_dataset_item(users, 'test20.txt')
    write_to_file(results, 'test20_iufitem.txt')  

def item_amp_tests(users):

    results = test_dataset_item(users, 'tiny5.txt')
    write_to_file(results, 'tiny5_item_amp.txt')
    results = test_dataset_item(users, 'test5.txt')
    write_to_file(results, 'test5_item_amp.txt')
    results = test_dataset_item(users, 'test10.txt')
    write_to_file(results, 'test10_item_amp.txt')    
    results = test_dataset_item(users, 'test20.txt')
    write_to_file(results, 'test20_item_amp.txt')  


def cos_tests(users):
    results = test_dataset(users, 'tiny5.txt')
    write_to_file(results, 'tiny5_jaccarddirichlet.txt')
    results = test_dataset(users, 'test5.txt')
    write_to_file(results, 'test5_jaccarddirichlet.txt')
    results = test_dataset(users, 'test10.txt')
    write_to_file(results, 'test10_jaccarddirichlet.txt')    
    results = test_dataset(users, 'test20.txt')
    write_to_file(results, 'test20_jaccarddirichlet.txt')  

def cross_val(users, k):

    k_size = len(users) / k
    
    k_list = [users[i:i + k] for i in range(0, len(users), k)]
    lowest_err = 10000
    lowest_users = []

    for i in range(0, len(k_list)):
        test = k_list[i]

        train = []
        for j in range(0, len(k_list)):
            if j != i:
                # train.append(k_list[i])
                # np.concatenate(train, k_list[i])
                for f in range(0, len(k_list[i])):
                    train.append(k_list[i][f])
            else:
                # zero_arr = [[0] * 1000] * k_size
                # # train.append(zero_arr)
                # np.concatenate((train, zero_arr), axis=0)
                for f in range(0, len(k_list[i])):
                    train.append([0] * 1000)

        # train = [test[j] for j in range(0, len(k_list)) if j != i]
        print "length of train"
        print len(train)
        offset = i * k_size
        test_formatted = convert_to_form(test)
        a, b = strip_test(test_formatted)
        err, fuckshit = cross_val_train(train, a, b)
        print err
        if (err < lowest_err):
            lowest_err = err
            lowest_users = fuckshit

    print lowest_err
    print lowest_users
    write_train(lowest_users, 'train_best.txt')
    return lowest_users

def write_train(users, dataset_name):
    print "writing to file"
    target = open(dataset_name, "w")
    for u in users:
        line = "\t".join([str(x) for x in u])
        line+="\n"
        target.write(line)



def convert_to_form(messy_form):
    sol = []
    for i in range(0, len(messy_form)):
        user = messy_form[i]
        for j in range(0, len(user)):
            if (user[j] != 0):
                arr = [i, j, user[j]]
                sol.append(arr)
                # print sol
    return sol


def cross_val_train(users, query_key, answer_key):

    batch_sol = []
    curFrame = [0] * 1000
    curUser = query_key[0][0]
    curPredict = []
    for userId, movieId in query_key:
        userId-=1
        movieId-=1
        if userId != curUser:
            print userId
            processUser(users, curFrame, curPredict, userId, batch_sol)
            curUser = userId
            curFrame = [0] * 1000
            curPredict = []
        
        curPredict.append(movieId)


    processUser(users, curFrame, curPredict, userId, batch_sol)

    x, y = strip_test(batch_sol)

    return np.sqrt(mean_absolute_error(answer_key, y)), users

def strip_test(full_test):

    answer_key = []
    query_key = []
    for user, movieId, rating in full_test:
        answer_key.append(rating)
        query_key.append([user, movieId])



    return query_key, answer_key


def pearson_steroid_tests(users):

    global is_iuf
    global is_amp

    is_iuf = 0
    is_amp = 1


    print is_iuf
    print is_amp 

    results = test_dataset(users, 'tiny5.txt')
    write_to_file(results, 'tiny5_adjamp.txt')
    results = test_dataset(users, 'test5.txt')
    write_to_file(results, 'test5_adjamp.txt')
    results = test_dataset(users, 'test10.txt')
    write_to_file(results, 'test10_adjamp.txt')    
    results = test_dataset(users, 'test20.txt')
    write_to_file(results, 'test20_adjamp.txt')  


def main():  
    num_users = 200
    num_movies = 1000
    users = [[0] * num_movies] * num_users
    train_users(users, 'train.txt')    
#    users = cross_val(users, 5)
    centered_cos_tests(users)
    # item_amp_tests(users)
 #   pearson_steroid_tests(users)
 #   blend_tests(users)
    cos_tests(users)

main()