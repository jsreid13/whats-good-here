from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from time import time

import multiprocessing as mp
import logging
import pymysql
import os

# Uncomment this for debugging information to print to console
#  logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Send sql query to get information
def sql_query(query: str, connection):
    with connection.cursor() as cursor:
        cursor.execute(query)
        return cursor.fetchall()


def sql_insert(_restaurant_id: str, _meal: str, _average_stars: float, _reviews: str, connection):
    with connection.cursor() as cursor:
        # Insert values into table of your choice and commit them
        sql = "INSERT INTO `meal_scores` (`business_id`, `meal`,\
               `average_stars`, `reviews`) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (_restaurant_id, _meal, _average_stars, _reviews))
    connection.commit()


def average_meal_rating(meal_dict):
    """
    Average the list of all ratings for each meal and return a dict

    :param list meal_dict: Dictionary where keys are meals and value is a list
    of dicts
    :returns: Input dictionary with average rating appended
    """
    for meal, review_dict in meal_dict.items():
        # List of all the star ratings for that meal to average later
        stars = list(review_dict.values())
        # Find the average of the stars for each review of this meal item
        avgValue = sum(stars)/len(stars)
        # Add the average stars to the final dictionary to be output for each
        # meal in the list for the restaurant
        meal_dict[meal]['average_stars'] = avgValue
    return meal_dict


def find_meal_words(review: str, _meal_set: set, word_lemmatizer):
    """
    Use POS tags in the reviews after being tokenized to extract all the nouns
    and then return the nouns that are present in the list of meal words
    Meals should be a set to allow for much faster finding of whether a noun
    present in a review is a meal word

    :param str review: The body text of a review
    :param set meals: Set containing all the potential valid meals
    :returns: list of found words within the review that are also meal words
    """
    possible_meal_names = []
    multiword_name = ''
    # Use split instead of nltk.tokenize because this is significantly faster
    # and we dont care about tokenizing punctuation or numbers when looking
    # for meal words
    tokenized = review.split()

    for (word, pos) in pos_tag(tokenized):
        # This means the word is a noun
        if pos[:2].startswith(('J', 'N')):
            lemmatized_word = word_lemmatizer.lemmatize(word.lower())
            # Also append the base noun itself in case it is also a meal word
            # or this chain is not a compound meal word
            possible_meal_names.append(lemmatized_word.strip())
            if possible_meal_names:
                # Noun is a string of consisting of groups of nouns found in the
                # review back-to-back, once a non-noun is found the string is reset
                # This appends the initial noun to the list to search for as
                # potential meal words, and then appends that inital word plus the
                # next noun if present and so on until a non noun is hit
                multiword_name = ' '.join([multiword_name, lemmatized_word])
                possible_meal_names.append(multiword_name.strip().lower())
            else:
                # This is the first word, so just start the noun chain but dont
                # write it to the nouns list or it will double up
                multiword_name = lemmatized_word
        else:
            multiword_name = ''
    # Convert to set to remove duplicates
    possible_meal_names_set = set(possible_meal_names)
    # Get list of words that are within the set of valid meal words
    output = [word for word in possible_meal_names_set if word in _meal_set]
    return output


def build_meal_set(word_lemmatizer):
    """
    Gather food item names from the Food101 dataset, webscraping and from the
    NLTK food synset, merge them together into a set to allow for faster search

    :param stem word_lemmatizer: A NLTK word lemmatizer to use make words consistent
    :returns: Set of meal words
    """
    food = wordnet.synset('food.n.02')
    # Words in these databases are separated by _ instead of spaces, this
    # converts them back to spaces and puts the word groups into a tuple
    food_list = [tuple(w.lower().split('_')) for s in food.closure(lambda s: s.hyponyms())
                 for w in s.lemma_names()]
    food_list = []
    [food_list.append(tuple(line.rstrip().split('_'))) for line in
        open(os.getcwd() + '/data/meal_names/category_names.txt',
             'r').read().lower().split('\n')]
    [food_list.append(tuple(line.rstrip().split(' '))) for line in
        open(os.getcwd() + '/data/meal_names/meals.txt',
             'r').read().lower().split('\n')]
    [food_list.append(tuple(line.rstrip().split(' '))) for line in
        open(os.getcwd() + '/data/meal_names/vegetables.txt',
             'r').read().lower().split('\n')]
    [food_list.append(tuple(line.rstrip().split(' '))) for line in
        open(os.getcwd() + '/data/meal_names/fruits.txt',
             'r').read().lower().split('\n')]
    stemmed_foods = []
    # If food is a tuple then that food consists of multiple words, if not then
    # it is just a string containing a single word
    for food in food_list:
        if isinstance(food, tuple):
            food_str = ''
            for word in food:
                food_str = food_str + word_lemmatizer.lemmatize(word) + ' '
            stemmed_foods.append(food_str.rstrip())
        else:
            stemmed_foods.append(word_lemmatizer.lemmatize(food))
    # Return the final list as a set for faster lookup
    return set(stemmed_foods)


def write_to_database(business_id, meal_scores_dict):
    """
    Send the average score and review id's for each meal found in a review
    to the SQL database

    :business_id: 22 long Hex ID for that business
    :meal_scores_dict: Dictionary containing meal_word:dict of review_ids and
    their score that mention that meal
    :returns: None
    """
    for meal_word, review_dict in meal_scores_dict.items():
        try:
            average_stars = review_dict.pop('average_stars')
        # If no meal word is detected with any review for that restaurant then
        # there will be no average stars and this will throw an error
        except KeyError:
            sql_insert(business_id, meal_word, 'NULL', 'NULL', cnx)
            continue
        # What is left in review_dict is just the remaining review_ids with
        # the star rating as value
        sql_insert(business_id, meal_word, average_stars, str(review_dict), cnx)
    return


def meal_score_finder(_startRow: int, _endRow: int, _final_dict: dict, review_count: int,
                      review_with_meal_count: int, _meal_set: set, connection):
    """
    Main function, made into a separate function to allow for multiprocessing

    :param int _startRow: The row number to start finding at, used for multiprocessing
    :param int _endRow: The row number to end finding at, used for multiprocessing
    :param dict final_dict: The dictionary to append the found meal score to
    :param set meals: Set containing valid meal names
    :returns: Dict of found meals scores for each restaurant in range
    """
    for restaurant_id in restaurants[_startRow:_endRow]:
        logging.info(restaurant_id)
        response = sql_query("select business_id, text, stars, id from review WHERE \
                business_id = '%s';" % restaurant_id, connection)
        meals_with_stars = {}
        for result in response:
            review_count.value += 1
            # Retrieve the important parts of what's returned from SQL
            review = result[1]
            stars = result[2]
            reviewID = result[3]
            review_meals = find_meal_words(review, _meal_set, lemma)
            # Count the number of reviews that found a meal word
            if review_meals:
                review_with_meal_count.value += 1
            for meal in review_meals:
                try:
                    meals_with_stars[meal][reviewID] = int(stars)
                except KeyError as e:
                    meals_with_stars[meal] = {reviewID: int(stars)}
        meals_with_avg_stars = average_meal_rating(meals_with_stars)

        # Write values to database and a dict for any further use in this script
        _final_dict[restaurant_id] = meals_with_avg_stars
        write_to_database(restaurant_id, meals_with_avg_stars)
        logging.info(meals_with_avg_stars)


# Function to easily find the time different parts of the code
def timer(t1, msg):
    t2 = time()
    print(msg + str(t2-t1))
    return t2


t1 = time()

# Use the slower lemmatizer for the final word from the review saved as a meal
# word as this will be very few words, and this will allow for more words
# to be grouped together
lemma = WordNetLemmatizer()

# The number of threads to run
num_workers = mp.cpu_count()

# Initialize the final output dict
final_dict = {}

# Create a set of the meals contained in mealsFile
meal_set = build_meal_set(lemma)

# Start connection to SQL database through pymysql
# TODO: Fill in with login info for your database
user_name = ''
database_name = ''
password = ''
cnx = pymysql.connect(user=user_name, database=database_name, password=password)

# Enclose all SQL work within a try...finally block so closure of the
# connection is guarunteed
try:
    # Ensure SQL table has business_id indexed so the for loop runs quickly
    try:
        sql_query("CREATE INDEX inx_bizid on review(business_id);", cnx)
    except pymysql.err.InternalError:
        logging.info("business_id already indexed")

    # Create the table in SQL to write these meal scores to and insert them all
    sql_query("DROP TABLE IF EXISTS `meal_scores`", cnx)
    # Made reviews column mediumtext to match attributes.value column from database
    # Might overflow if there is a lot of reviews with a lot of keywords
    sql_query("CREATE TABLE `meal_scores`(`business_id` varchar(22), `meal`\
              varchar(255), `average_stars` FLOAT(3), `reviews` MEDIUMTEXT,\
              PRIMARY KEY (`business_id`, `meal`))", cnx)

    # Parse the reviews and create a dict containing the average rating
    # of each meal and open the output file to write to
    restaurants = sql_query(
        "SELECT DISTINCT id FROM business JOIN (SELECT business_id FROM category\
            WHERE category='Restaurants') AS a ON business.id=a.business_id;", cnx)
    num_restaurants = len(restaurants)
    print(len(restaurants))

    manager = mp.Manager()  # For multiprocessing so create a global for each process
    final_dict = manager.dict()  # Create a global dict for each process to write to
    rev_count = manager.Value('i', 0)
    rev_with_meal_count = manager.Value('i', 0)

    i_ary = []  # Index to break up all the restaurants across each thread
    processes = []
    processChunks = [i for i in range(0, num_restaurants, int(num_restaurants/num_workers))]
    for i in processChunks:
        i_ary.append(i)
        if len(i_ary) == 1:
            if num_restaurants % num_workers != 0:
                # have to use new connections to SQL for each process or else they conflict
                p = mp.Process(target=meal_score_finder
                               , args=(processChunks[-1]
                                       , num_restaurants
                                       , final_dict
                                       , rev_count
                                       , rev_with_meal_count
                                       , meal_set
                                       , pymysql.connect(user=user_name,
                                                         database=database_name,
                                                         password=password)
                                       ))
                processes.append(p)
                p.start()
            continue
        p = mp.Process(target=meal_score_finder
                       , args=(i_ary[-2]
                               , i_ary[-1]
                               , final_dict
                               , rev_count
                               , rev_with_meal_count
                               , meal_set
                               , pymysql.connect(user=user_name,
                                                 database=database_name,
                                                 password=password)
                               ))
        processes.append(p)
        p.start()

    [x.join() for x in processes]
    timer(t1, "Finished in ")
    print("Percentage of reviews containing a meal word: " +
          str(rev_with_meal_count.value * 100 / rev_count.value))

finally:
    cnx.close()
