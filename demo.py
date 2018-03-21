import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import webbrowser
import gensim
import json
import string
import data_prep
import utils
import video_save
from datetime import datetime

 

#NEED: Words to embeddings, vid_model, word_model, original_video_embeddings, vid_to_url_map
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

def get_caption_vector(word_embedding_model, caption_input):
    processed_caption = replace_punctuation(caption_input)
    sentence = processed_caption.strip().split()
    sentence_vec = []
    for word in sentence:
        key = preprocess(word)
        if key in word_embedding_model:
            vector_word = word_embedding_model[key]
            sentence_vec.append(vector_word.tolist())
        else:
            print("Unfortunately, we can't come up with a word embedding for {}. Please enter a new caption".format(key))
            return None
    sentence_array = np.array(sentence_vec)
    print("Shape of outputted array from caption vector: ", sentence_array.shape)
    return Variable((torch.from_numpy(sentence_array)).unsqueeze(1).float())

def unpack_and_unscramble(output, indices, batch_size = 5000):
    output, lengths = nn.utils.rnn.pad_packed_sequence(output)

    # Unscramble output
    order = torch.zeros(indices.shape).long()
    for i in range(indices.size()[0]):
        order[indices[i]] = i
    video_unscrambled = utils.unscramble(output, lengths, order, batch_size, False)
    return video_unscrambled

def get_best_index(vid_embeddings, caption_embedding, num_best = 1):
    #Stack vid as rows, mess with shape of caption embedding so row vec
    best_indices = np.argsort(np.linalg.norm(caption_embedding - vid_embeddings, axis = 1))
    return best_indices[0:num_best]

def to_time(float_time):
    int_time = int(float_time)
    num_minutes = int_time // 60
    num_seconds = int_time - 60 * num_minutes
    str_time = str(num_minutes) + ":" + str(num_seconds)
    if num_seconds < 10:
        str_time = str(num_minutes) + ":0" + str(num_seconds)
    return str_time


def main(preprocessed = True):
    start = datetime.now()
    print("Initializing the Demo:")
    demo_dir = '/Users/erikjones/Documents/Stanford 2017:18/Winter CS230/Demo/'
    data_dir = '/Users/erikjones/Documents/Stanford 2017:18/Winter CS230/Full_Data/'
    print("Loading word embeddings. Time passed: ", str(datetime.now() - start))
    word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(demo_dir + 'wiki.en/wiki.en.vec', binary=False)
    print("Loading models. Time passed: ", str(datetime.now() - start))
    word_model = torch.load(demo_dir + 'word_best.pt', map_location=lambda storage, loc: storage)
    vid_model = torch.load(demo_dir + 'vid_best.pt', map_location=lambda storage, loc: storage)
    vid_embeddings, vid_ids, timestamps = None, None, None
    if preprocessed:
        vid_embeddings, vid_ids, timestamps = video_save.reconstruct_files()
    else:
        print("Making dataset. Time passed: ", str(datetime.now() - start))
        filenames = [demo_dir + 'demo_' + str(i) + '.pkl' for i in range(5)]
        dataSet = data_prep.Dataset(filename = filenames)
        print("Getting pairs. Time passed: ", str(datetime.now() - start))
        datasets, indices = dataSet.get_pairs(0, 5000, store_names = True)
        vid_dataset, vid_indices = datasets[1], indices[1]
        vid_ids = dataSet.vid_ids
        timestamps = dataSet.time_stamps
        #Get indexing working
        print("Calculating video embeddings. Time passed: ", str(datetime.now() - start))
        vid_outputs_packed = vid_model(vid_dataset)
        vid_embeddings = unpack_and_unscramble(vid_outputs_packed, vid_indices)
        vid_embeddings = vid_embeddings.data.numpy()
    print("Ready! Total time: ", str(datetime.now() - start))
    caption_input = input("Enter a caption (q to quit): ")
    same_caption_count = 1
    while caption_input.lower() != "q":
        caption_vec = get_caption_vector(word_embedding_model, caption_input)
        if caption_vec is not None:
            #Test if unsqueeze is necessary
            caption_embedding = word_model(caption_vec)
            caption_embedding = caption_embedding.data.numpy()[-1, :, :]
            start = datetime.now()
            best_video_indices = get_best_index(vid_embeddings, caption_embedding, num_best = same_caption_count)
            #Takes last one, which is the only new one?
            vid_index = best_video_indices[-1]
            print("TIME: ", str(datetime.now() - start))
            vid_id = vid_ids[vid_index]
            timestamp = timestamps[vid_index]
            #Python trash to convert video id to a URL.
            list_vid_id = list(vid_id)
            list_vid_id[1] = "="
            vid_id = "".join(list_vid_id)
            url = "https://www.youtube.com/watch?" + vid_id
            print("URL: ", url)
            print("Start: {}, End: {}".format(to_time(timestamp[0]), to_time(timestamp[1])))
            watch = input("Want to load the URL? (y/n): ")
            while watch.lower()[0] not in ['y', 'n']:
                print("Invalid response.")
                watch = input("Want to load the URL? (y/n): ")
            if watch.lower()[0] == 'y':
                webbrowser.open(url)
            prev_input = caption_input
        #invalid word entered to get_caption_vector
        else:
            same_caption_count = 1
            prev_input = None
        caption_input = input("Enter a caption (q to quit, t to try next video): ")
        if caption_input.lower() == 't' and prev_input is not None:
            caption_input = prev_input
            same_caption_count += 1
        else:
            same_caption_count = 1

if __name__ == '__main__':
    main()
