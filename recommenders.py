import numpy as np

        


    
    
def sumOfWeights(users, a):
    denominator = 0
    for userPos in range(1, len(users)):
        denominator = denominator + cosine_sim(a, users[userPos])
        
    return denominator
        
# Takes in vector for a and position for i
def weightedAverage(a, i):
    weight = cosine_sim(a, users[i])
    numerator = 0
    for moviePos in range(1, len(users[1])):
        numerator = numerator + (weight * users[i][moviePos])

    return numerator
    
def cosMethod(users, key):
    weightSum = sumOfWeights(users, key)
    weightAverage = 0
    
    for i in range(1, len(users)):
        weightAverage = weightAverage + weightedAverage(key, i)

    print weightAverage
    print weightSum

    rating = weightAverage / weightSum
    return rating

def getAverage(user):
    return sum(user) / len(user)

def getDeviation(i)
    avg = getAverage(i)
    

def userDeviation(user):
    avg = getAverage(user)
    wd = 0
    for i in range(1, len(user)):
        wd = wd + (user[i] - avg)
        
    return wd    
    
def weightedDeviation(users, a):
    for i in range(1, len(users))
    weight = cosine_sim(a, )


def pearson_filtering(users, key):   
    userAverage = getAverage(key)
    
    



def train_users(users):
    training = open('train.txt', 'r')
    training = training.read().strip().split('\r')
    print len(training)
    for i, line in enumerate(training):
        users[i] = [int(x) for x in line.split()]