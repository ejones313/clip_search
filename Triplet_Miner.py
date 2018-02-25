import numpy as np
import random

def triplet_loss(A, P, N, margin=0.0):
    pos_dist = np.linalg.norm(A-P)
    neg_dist = np.linalg.norm(A-N)
    return pos_dist - neg_dist + margin

def mine_triplets_all(embedding_tuples):
    triplets_caption = []
    triplets_clips = []
    for tuple in embedding_tuples:
        anchor = tuple[0]
        positive = tuple[1]
        for tuple in embedding_tuples:
            negative = tuple[0]
            if triplet_loss(anchor, positive, negative) > 0:
                triplets_caption.append((anchor, positive, negative))

            temp = anchor
            anchor = positive
            positive = temp

            negative = tuple[1]
            if triplet_loss(anchor, positive, negative) > 0:
                triplets_clips.append((anchor, positive, negative))
    return triplets_caption, triplets_clips

def mine_triplets_random(embedding_tuples):
    triplets_caption = []
    triplets_clips = []
    m = len(embedding_tuples)
    for tuple in embedding_tuples:
        anchor = tuple[0]
        positive = tuple[1]
        index = random.randint(0, m - 1)
        while triplet_loss(anchor, positive, embedding_tuples[index][0]) <= 0:
            index = random.randint(0, m - 1)
        triplets_caption.append((anchor, positive, embedding_tuples[index][0]))

        temp = anchor
        anchor = positive
        positive = temp

        while triplet_loss(anchor, positive, embedding_tuples[index][1]) <= 0:
            index = random.randint(0, m - 1)
        triplets_clips.append((anchor, positive, embedding_tuples[index][1]))