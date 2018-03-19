import json
import gensim
import numpy as np

with open("./captions/train_vec1.json", "r") as f:
    videos = json.load(f)

model = gensim.models.KeyedVectors.load_word2vec_format('./wiki/wiki.en.vec', binary=False)
print('Model Loaded.')

for id in videos:
    video = videos[id]
    captions_vec = video['vectors']
    for caption_vec in captions_vec:
        for word_vec in caption_vec:
            vec = np.array(word_vec, dtype=np.float32)
            print(model.similar_by_vector(vec, topn=3))
