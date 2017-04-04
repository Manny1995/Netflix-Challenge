import numpy as np
from sklearn.metrics import jaccard_similarity_score
from scipy.stats import pearsonr

from helpers import (
    filter_common,
    filter_single,
    clean_rating,
    cosine_sim,
    pearson_correlation,
    adjusted_cosine_sim,
    set_clean_rating
)

from steroids import (
    iuf,
    case_amplification
)
    
cachedMovies = []

# return_clean_rating
moviesMean = []
totalClicks = 0

iuf_weights = []



def jaccard_filtering(users, activeUser, itemIndex, case_amp):
    numerator = 0
    denominator = 0
    weights = [jaccard_similarity_score(activeUser, u) for u in users]

    for userPos in range(0, len(users)):
        weight = weights[userPos]
        if (users[userPos][itemIndex] == 0):
            continue

        if (case_amp == 1):
            weight = case_amplification(weight)

        denominator += weight
        temp = weight * users[userPos][itemIndex]
        numerator += temp

    rating = 3
    if denominator != 0:
        rating = numerator / denominator

    global moviesMean
    global totalClicks
    if (len(moviesMean) == 0):
        addOne(users)

    # rating = active_avg + temp_rating
    rating *= (moviesMean[itemIndex] + 0) / (moviesMean[itemIndex] + 1)

    return clean_rating(rating)


def centered_jaccard(users, activeUser, movieIndex, case_amp):
    denominator = 0
    numerator = 0
    weights = [jaccard_similarity_score(users[movieIndex], u) for u in users]

    #200
    for i in range(0, len(users)):
        weight = weights[i]

        if (case_amp == 1):
            weight = case_amplification(weight)

        aRating = activeUser[i]

        if (activeUser[i] == 0):
            continue

        denominator += weight
        numerator+= (weight * aRating)


    rating = 3
    if denominator != 0:
        rating = numerator / denominator

    return clean_rating(rating)




def getMean(users):
    movies = np.transpose(users)
    global moviesMean 
    moviesMean = []
    for i in range(0, len(movies)):
        moviesMean.append(np.mean(movies[i]))



def addOne(users):
    global totalClicks 
    totalClicks = 0

    totalMovies = [0] * 1000
    movies = np.transpose(users)
    temp_mov = [0] * 1000
    for i in range(0, len(users)):
        for j in range(0, 1000):
            if (users[i][j] > 0):
                totalMovies[j]+=1
                totalClicks+=1

    global moviesMean
    moviesMean = temp_mov



def cosine_filtering(users, activeUser, itemIndex, case_amp):

    numerator = 0
    denominator = 0
    weights = [cosine_sim(activeUser, u) for u in users]

    common_movie = 0
    for userPos in range(0, len(users)):
        weight = weights[userPos]

        if (users[userPos][itemIndex] == 0):
            continue

        if (case_amp == 1):
            weight = case_amplification(weight)

        denominator += weight
        temp = weight * users[userPos][itemIndex]
        numerator += temp

    rating = 3
    if denominator != 0:
        rating = numerator / denominator

    global moviesMean
    global totalClicks
    if (len(moviesMean) == 0):
        addOne(users)

    rating *= (moviesMean[itemIndex] + 0) / (moviesMean[itemIndex] + 1)

    return clean_rating(rating)

def avgUserRating(a):
    a_single = filter_single(a)
    if (len(a_single) <= 0):
        return 3

    res = np.mean(a_single)
    return res

def adjusted_cosine_filtering(users, movies, activeUser, movieIndex, case_amp):

    num = 0
    denom = 0

    for i in range(0, len(movies)):

        active_j = activeUser[i]

        if (active_j == 0):
            continue

        weight = adjusted_cosine_sim(users, movies, movieIndex, i)

        if (case_amp == 1):
            weight = case_amplification(weight)

        num+=(weight * active_j)
        denom+=weight

    rating = 3
    if denom != 0:
        rating = num / denom


    global moviesMean
    global totalClicks
    if (len(moviesMean) == 0):
        addOne(users)
    rating *= (moviesMean[itemIndex] + 0) / (moviesMean[itemIndex] + 1)


    return clean_rating(rating)




def pearson_filtering(users, activeUser, movieIndex, case_amp, is_iuf):
    denom = 0
    num = 0

    active_avg = 0

    filter_active = filter_single(activeUser)

    if len(filter_active) == 0:
        active_avg = 0
    else:
        active_avg = np.mean(filter_active)

    global iuf_weights

    weights = []
    
    if (is_iuf == 1):
        steroidUsers = iuf(users)
        weights = [pearsonr(activeUser, u) for u in steroidUsers]
    else:
        weights = [pearsonr(activeUser, u) for u in users]


    for i in range(0, len(users)):
        pearson = weights[i][0]

        if (users[i][movieIndex] == 0):
            continue

        if (case_amp == 1):
            pearson = case_amplification(pearson)            


        denom = denom + abs(pearson)

        user_dev = users[i][movieIndex] - np.mean(filter_single(users[i]))
        num = num + (pearson * user_dev)


    temp_rating = 0
    if denom != 0:
        temp_rating = num/denom


    global moviesMean
    global totalClicks
    if (len(moviesMean) == 0):
        addOne(users)

    rating = active_avg + temp_rating
    rating *= (moviesMean[itemIndex] + 0) / (moviesMean[itemIndex] + 1)


    return clean_rating(rating)


def combined_filtering(users, movies, activeUser, movieIndex, case_amp):
    denom = 0
    num = 0

    active_avg = 0

    filter_active = filter_single(activeUser)
    if len(filter_active) == 0:
        active_avg = 0
    else:
        active_avg = np.mean(filter_active)

    weights = [pearsonr(activeUser, u) for u in users]
    jaccard_weights = [jaccard_similarity_score(movies[movieIndex], mov) for mov in movies]

    for i in range(0, len(users)):
        pearson = weights[i][0]

        if (users[i][movieIndex] == 0):
            continue

        if (case_amp == 1):
            pearson = case_amplification(pearson)

        pearson *= jaccard_weights[i]

        denom = denom + abs(pearson)

        user_dev = users[i][movieIndex] - np.mean(filter_single(users[i]))
        num = num + (pearson * user_dev)


    temp_rating = 0
    if denom != 0:
        temp_rating = num/denom

    return clean_rating(active_avg + temp_rating)

    

def centered_cosine(users, activeUser, movieIndex, case_amp):
    denom = 0
    num = 0

    active_avg = 0

    filter_active = filter_single(activeUser)
    if len(filter_active) == 0:
        active_avg = 0
    else:
        active_avg = np.mean(filter_active)

    weights = [cosine_sim(activeUser, u) for u in users]

    for i in range(0, len(users)):
        pearson = weights[i]

        if (users[i][movieIndex] == 0):
            continue

        if (case_amp == 1):
            pearson = case_amplification(pearson)


        denom = denom + abs(pearson)

        user_dev = users[i][movieIndex] - np.mean(filter_single(users[i]))
        num = num + (pearson * user_dev)


    temp_rating = 0
    if denom != 0:
        temp_rating = num/denom

    global moviesMean
    global totalClicks
    if (len(moviesMean) == 0):
        addOne(users)

    rating = active_avg + temp_rating
    rating *= (moviesMean[itemIndex] + 0) / (moviesMean[itemIndex] + 1)

    return clean_rating(rating)


def blended_model(users, activeUser, movieIndex, case_amp):
    
    #These weights are based on the MAE of each individual model
    weights = [(1/0.788663602035791), (1/0.79724183221146), (1/0.81993925463799)]
    # cache movies, so we do not have to recalculate
    global cachedMovies 
    if len(cachedMovies) <= 0:
        cachedMovies = np.transpose(users)


    r1 = weights[0] * cosine_filtering(users, activeUser, movieIndex, 0)
    r2 = weights[1] * pearson_filtering(users, activeUser, movieIndex, 0, 0)
    r3 = weights[2] * adjusted_cosine_filtering(users, cachedMovies, activeUser, movieIndex, 0)

    return clean_rating(((r1+r2+r3) / np.sum(weights)))

    