import scipy.sparse as sp
import sys

def load_data(rating_file, testRatio=0.1, binary=True):
    user_count = item_count = 0
    ratings = []
    for line in open(rating_file):
        arr = line.strip().split()
        user_id = int(arr[0])
        item_id = int(arr[1])
        score = float(arr[2])
        timestamp = int(arr[3])
        ratings.append((user_id, item_id, score, timestamp))
        user_count = max(user_count, user_id)
        item_count = max(item_count, item_id)
    user_count += 1
    item_count += 1

    ratings = sorted(ratings, key=lambda x: x[3])

    test_count = int(len(ratings) * testRatio)
    count = 0
    train_matrix = sp.lil_matrix((user_count, item_count))
    test_ratings = []
    for rating in ratings:
        if count < len(ratings) - test_count:
            train_matrix[rating[0], rating[1]] = 1 if binary else rating[2]
        else:
            test_ratings.append(rating)
        count += 1

    new_users = set([])
    new_ratings = 0

    for u in range(user_count):
        if train_matrix.getrowview(u).sum() == 0:
            new_users.add(u)
    for rating in ratings:
        if rating[0] in new_users:
            new_ratings += 1

    sys.stderr.write("Data\t{}\n".format(rating_file))
    sys.stderr.write("#Users\t{}, #newUser: {}\n".format(user_count, len(new_users)))
    sys.stderr.write("#Items\t{}\n".format(item_count))
    sys.stderr.write(
        "#Ratings\t {} (train), {}(test), {}(#newTestRatings)\n".format(
            train_matrix.sum(), len(test_ratings), new_ratings))

    return train_matrix, test_ratings

def SortMapByValue(d):
    return sorted(d.items(), key=lambda x: -x[1])

def TopKeysByValue(map_item_score, topK, ignore_keys):
    if ignore_keys is None:
        ignore_set = set()
    else:
        ignore_set = set(ignore_keys)

    # Another implementation that first sorting.
    top_entities = SortMapByValue(map_item_score)
    top_keys = []
    for entry in top_entities:
        if len(top_keys) >= topK:
            break
        if not entry[0] in ignore_set:
            top_keys.append(entry[0])
    return top_keys
