import csv
import numpy as np
import datetime
from datetime import datetime

# movie profile class, contains movie data
class MovieProfile :
    def __init__(self, id, title, genres, rating: float) -> None:
        self.id = id
        self.title = title
        self.genres = genres.split('|')[-1]
        self.rating = rating
        self.tag = ""

    # checks equality by id
    def __eq__(self, __o: object) -> bool:
        if __o is MovieProfile :
            return self.id == __o.id
        return False

# user class, contains the movies the user rated
class User :
    def __init__(self, id) -> None:
        self.movies = dict()
        self.id = id

    # adds movie to user list
    def add_movie(self, movie_data, rating) :
        movie = MovieProfile(movie_data[0], movie_data[1], movie_data[2], float(rating))

        self.movies[movie_data[0]] = movie

    # changes movie tag in user list
    def update_tag(self, movie_id, tag) :
        if self.movies.get(movie_id) == None :
            return
        self.movies[movie_id].tag = tag

    # checks if user has movie
    def has_movie(self, id) :
        return self.movies.get(id) != None
    
    # gets movie from user, only call when know it exists
    def get_movie(self, id) :
        return self.movies[id]

# go through whole table, get averages
def get_all_ratings(movies:list, users) :
    out = []

    i = 0

    # goes through every movie
    while i < len(movies) :
        # gets ratings of movie
        x = get_adjusted_rating(movies[i], users)

        # checks if movie has enough ratings to be recommended
        if x == None :
            del movies[i]
        else :
            out.append(x)
            i += 1
    
    return np.array(out)

# gets ratings of movie
def get_adjusted_rating(movie_id, users:list) :
    average = 0
    count = 0
    same = True
    last = None

    # goes through each user, gets av
    for user in users :
        # if user has movie
        if user.has_movie(movie_id) :
            # add to average
            average += user.get_movie(movie_id).rating
            count += 1

            # checks if all values are same
            if last == None :
                last = user.get_movie(movie_id).rating
            else :
                if last != user.get_movie(movie_id).rating :
                    same = False
                else :
                    last = user.get_movie(movie_id).rating

    # if no user rated
    if count == 0 or same :
        return None
    
    average /= count

    values = [0] * len(users)

    # goes through each user, saves adjusted user rating for cos sim
    for i in range(len(users)) :
        if users[i].has_movie(movie_id) :
            values[i] = users[i].get_movie(movie_id).rating - average
        else :
            values[i] = 0

    return values

# reads user reviews
def read_reviews_and_tags(rating_path, tag_path, movies:dict) :
    file = open(rating_path, 'r', encoding="utf-8")
    users = dict()

    skip = True 
    matrix = np.zeros(shape=(len(movies), 671))
    # read user views
    for line in csv.reader(file) :
        # skip first line
        if skip :
            skip = False
            continue

        # new user
        if users.get(line[0]) == None :
            users[line[0]] = User(line[0])
        
        #add movie
        if movies.get(line[1]) != None :
            users[line[0]].add_movie(movies[line[1]], line[2])
    file.close()

    file = open(tag_path, 'r', encoding="utf-8")

    # read tags
    for line in csv.reader(file) :
        if users.get(line[0]) == None :
            continue

        # gets tag and movie it belongs to
        movie_id = line[1]
        tag = line[2]

        # updates user movie with user tag
        users[line[0]].update_tag(movie_id, tag)
    file.close()

    return users

# reads movies from file
def read_movies(path) :
    file = open(path, 'r', encoding="utf-8")
    data = dict()

    skip = True

    # goes through each movie
    for line in csv.reader(file) :
        # skip first line
        if skip :
            skip = False
            continue
        # save movie details to movie id
        data[line[0]] = line
    file.close()
    return data

# calculates cosine similarity from a given matrix
def calculate_cosine_similarity(data) :
    # computes dot product for each element
    similarity = np.dot(data, data.T)

    # squared magnitude
    square_mag = np.diag(similarity)
    inv_square_mag = 1 / square_mag

    # fixes 1/0 issues
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # computes magnitude of matrix elements
    inv_mag = np.sqrt(inv_square_mag)
        
    # cosine similarity (dot product times 1/(mag1*mag2))
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag

    # replaces 1's with -10 in order to ignore in calculations
    np.fill_diagonal(cosine, -10)

    return cosine

# computes neighborhoods of size n
def compute_neighborhoods(similarities) :
    # computes neighborhood of size 5 for each movie
    neighborhoods = np.empty(shape=(similarities.shape[0], 5), dtype=tuple)
    for i in range(0,similarities.shape[0]) :
        neighborhoods[i] = top_five(similarities[i])

    return neighborhoods

# gets all user recommendations
def get_user_recommendations(data, users,movies) :
    # gets similarity matrix
    similarities = calculate_cosine_similarity(data)

    # dictionary of user recommendations
    user_recs = dict()

    # gets neighborhoods of each movie
    neighborhoods = compute_neighborhoods(similarities)

    # goes through each user
    for user in users.values() :
        # goes through each movie
        for movie_index in range(len(movies)) :
            movie = movies[movie_index]

            # if user doesnt have movie, get rating
            if not user.has_movie(movie) :
                estimate_rating(user, movie, neighborhoods[movie_index], user_recs, movies)

    return user_recs

# gets top five elements in list
def top_five(list) :
    # list of top (moviename, similarity) tuples
    top_values = [("",-10)] * 5

    # goes through every element in list
    for i in range(len(list)) :
        # if new value is more than smallest value in the top values
        if list[i] > top_values[0][1] :
            # swaps first element with new value
            temp = top_values[0]
            top_values[0] = (i, list[i])

            # swaps remaining top values
            for j in range(0,4) :
                if top_values[j][1] > top_values[j+1][1] :
                    temp = top_values[j]
                    top_values[j] = top_values[j+1]
                    top_values[j+1] = temp
                else :
                    break
    
    return top_values

# estimates user rating
def estimate_rating(user:User, movie, neighborhood, user_recs, movies) :
    top = 0
    bottom = 0

    # goes through all 5 values 
    for sim in neighborhood :
        # checks if user has movie
        if user.has_movie(movies[sim[0]]) :
            top += (user.get_movie(movies[sim[0]]).rating * sim[1])
            bottom += sim[1]
     
    # if no ratings to go off of
    if bottom == 0 : 
        return

    rxi = (top / bottom)

    # if user doesnt have recommended items yet
    if user_recs.get(user.id) == None :
        user_recs[user.id] = dict()
    user_recs[user.id][movie] = rxi

# sorts dictionary by key then value
def sort_dict(elements):
    sort = list(sorted(elements.items(), key=lambda x: x[0]))
    return list(sorted(sort, key=lambda x: x[1],reverse=True))

# runs program
def main() :
    folder = './movie-lens-data/'

    # reads movies and ratings and tags, saves
    all_movies = read_movies(folder + 'movies.csv')
    users = read_reviews_and_tags(folder + 'ratings.csv', folder + 'tags.csv', all_movies)

    # converts dictionary of movieid-movieprofiles to list of movie ids
    movies = list(all_movies.keys())

    # gets users in list form
    user_items = list(users.values())

    # gets utility matrix
    data = get_all_ratings(movies, user_items)

    # gets user recommendations
    recommendations = get_user_recommendations(data, users, movies)

    # sorts all user item estimations
    outputs = dict()
    for id, value in recommendations.items() :
        outputs[id] = sort_dict(value)

    # outputs user movie recommendations
    file = open('output.txt', 'w')
    for id, estimates in outputs.items() :
        # print user id
        file.write(id)

        # print top 5 movie ids
        for i in range(0,min(5,len(estimates))) :
            file.write(' ' + str(estimates[i][0]))
        
        file.write('\n')

    file.close()

if __name__ == '__main__' :
    main()