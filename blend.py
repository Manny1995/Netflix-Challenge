from challenge import (
	write_to_file
)

def weighted_results(results_file, weight):
	dataset = open(results_file, 'r').read().strip().split('\n')
	dataset = [data.split() for data in dataset]
	dataset = [[int(e) for e in data] for data in dataset]
	
	results = []

	for userId, movieId, rating in dataset:
		results.append(rating)

	return [res * weights for res in results]

def users_and_movies(results_file):
	dataset = open(results_file, 'r').read().strip().split('\n')
	dataset = [data.split() for data in dataset]
	dataset = [[int(e) for e in data] for data in dataset]

	users = []
	movies = []

	for userId, movieId, rating in dataset:
		users.append(userId)
    	movies.append(movieId)

	return users, movies

def init_weights():
	weights = {}
	weights['pearson'] = 0.79724183221146
	weights['cosine'] = 0.788663602035791
	weights['item'] = 0.81993925463799


def main():
	weights = init_weights();

	users, movies = users_and_movies("results/pearson/test5_pearson_results.txt")
	total = zip(weighted_results("results/pearson/test5_pearson_results.txt", weights['pearson']), weighted_results("results/test5_cos_results.txt", weights['cosine']), weighted_results("results/item/test5_adj_results.txt", weights['item']))

	# total = [sum(t) for t in total]
	# w_s = sum(weights)
	# total = [t / w_s for t in total]

	# write_to_file(zip(users, movies, total), "test5_blend.txt")

main()