import gensim
import json
import string

def preprocess(word):
    translator = str.maketrans('', '', string.punctuation + string.digits)
    processed = word
    processed = processed.translate(translator)
    return processed

def replace_punctuation(sentence):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    processed = sentence.lower()
    processed = processed.translate(translator)
    return processed

# Load Facebook's Fasttext model.
model = gensim.models.KeyedVectors.load_word2vec_format('./wiki/wiki.en.vec', binary=False)
print('Model Loaded.')

with open("./captions/train.json", "r") as f:
    videos = json.load(f)

for id in videos:
    video = videos[id]
    captions = video['sentences']
    caption_vec = []
    for caption in captions:
        processed_caption = replace_punctuation(caption)
        sentence = processed_caption.strip().split()
        sentence_vec = []
        for word in sentence:
            key = preprocess(word)
            if key in model:
                vector_word = model[key]
                sentence_vec.append(vector_word.tolist())
            else:
                print(word)
                print(key)
        caption_vec.append(sentence_vec)
    video['vectors'] = caption_vec

print('Conversions Complete')

all_keys = list(videos.keys())
i = 1
for i in range(10):
    print(i)
    keys = all_keys[i*1000:(i+1)*1000]
    print(len(keys))
    vid_partial = { key:value for key,value in videos.items() if key in keys }
    with open("./captions/train_vec" + str(i+1) + ".json", "w") as f:
        json.dump(vid_partial, f)
    i += 1

del model
