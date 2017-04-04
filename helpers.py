import numpy as np

cache = {}

return_clean_rating = 1

def set_clean_rating(i):
    global return_clean_rating
    return_clean_rating = i

def filter_common(v1, v2):
    v1_new = []
    v2_new = []
    for i in range(0, len(v1)):
        if (v1[i] > 0 and v2[i] > 0):
            v1_new.append(v1[i])
            v2_new.append(v2[i])
    
    return v1_new, v2_new
    
def filter_single(a):
    new_vec = []
    for i in range(0, len(a)):
        if (a[i] > 0):
            new_vec.append(a[i])

    if (len(new_vec) == 0):
        new_vec.append(0)

    return new_vec

def vector_magnitude(v1):
    return np.sqrt(np.sum(np.square(v1)))

def cosine_sim(a, b):

    a_new, b_new = filter_common(a, b)

    sim = np.dot(a_new, b_new)

    norm_a = vector_magnitude(a_new)
    norm_b = vector_magnitude(b_new)
    if norm_a != 0 and norm_b != 0:
        sim /= (norm_a * norm_b)
    else:
        sim = 0

    if sim > 1:
        sim = 1
    elif sim < -1:
        sim = -1

    return sim


def adjusted_cosine_sim(users, movies, i_index, j_index):

    num = 0
    d1 = 0
    d2 = 0

    # cache the average of the movies so we do not have to repeat calculations
    global cache

    for u in range(0, 200):

        user_rating_i = users[u][i_index]
        user_rating_j = users[u][j_index]

        user_avg = 0
        if (cache.has_key(u)):
            user_avg = cache[u]
        else:
            user_avg = np.mean(filter_single(users[u]))
            cache[u] = user_avg
 

        t1 = user_rating_i - user_avg
        t2 = user_rating_j - user_avg

        num+=(t1 * t2)
        d1+=np.power(t1, 2)
        d2+=np.power(t2, 2)

    if (d1 == 0 or d2 == 0):
        return 0

    sim = num / (np.sqrt(d1) * np.sqrt(d2))

    if sim > 1:
        sim = 1
    elif sim < -1:
        sim = -1

    return sim

def pearson_correlation(a, b):

    a_new, b_new = filter_common(a, b)
    a_mean = np.mean(a_new)
    b_mean = np.mean(b_new)
    a_adj = np.subtract(a_new, a_mean[np.newaxis])
    b_adj = np.subtract(b_new, b_mean[np.newaxis])
    num = np.dot(a_adj, b_adj)
    sum_sq_a = np.dot(a_adj, a_adj)
    sum_sq_b = np.dot(b_adj, b_adj)
    denom = np.sqrt(sum_sq_a * sum_sq_b)
    if denom == 0:
        return 0
    return num/denom


def clean_rating(rating):
    global return_clean_rating
    # print rating

    if (return_clean_rating == 0):
        # print "unclean"
        return rating

    if (rating > 5):
        rating = 5
    elif (rating < 1):
        rating = 1
    else:
        rating = int(round(rating))

    return rating
