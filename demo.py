import torch
import torch.nn as nn
import numpy as np
import webbrowser
import gensim
import json
import string
import Word2Vec
import data_prep
import utils

#NEED: Words to embeddings, vid_model, word_model, original_video_embeddings, vid_to_url_map

def get_caption_vector(word_embedding_model, caption_input):
    processed_caption = Word2Vec.replace_punctuation(caption_input)
    sentence = processed_caption.strip().split()
    sentence_vec = []
    for word in sentence:
        key = Word2Vec.preprocess(word)
        if key in model:
            vector_word = model[key]
            print("Output from model: ",vector_word.tolist())
            sentence_vec.append(vector_word.tolist())
        else:
            print("Unfortunately, we can't come up with a word embedding for {}. Please enter a new caption".format(key))
            return None
    sentence_array = np.array(sentence_vec)
    print("Shape of outputted array from caption vector: ", sentence_array)
    return sentence_array

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


def main():
	print("Initializing the Demo...")
	demo_dir = '/Users/erikjones/Documents/Stanford 2017:18/Winter CS230/Demo/'
	word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(demo_dir + 'wiki.en/wiki.en.vec', binary=False)
	word_model = torch.load(demo_dir + 'word_best.pt')
	vid_model = torch.load(demo_dir + 'vid_best.pt')
	filenames = [demo_dir + 'demo_' + str(i) + '.pkl' for i in range(5)]
	dataSet = data_prep.Dataset(filename = filenames)
	datasets, indices = dataSet.get_pairs(0, 5000, store_names = True)
	vid_dataset, vid_indices = datasets[1], indices[1]
	vid_ids = dataSet.vid_ids
	timestamps = dataSet.time_stamps
	#Get indexing working
	vid_outputs_packed = vid_model(vid_dataset)
	vid_embeddings = unpack_and_unscramble(vid_outputs_packed, vid_indices)
	caption_input = ""
	print("Ready!")
	while caption_input.lower() != "q":
		caption_input = input("Enter a caption (q to quit): ")
		caption_vec = get_caption_vector(word_embedding_model, caption_input)
		if caption_vec is not None:
			#Test if unsqueeze is necessary
			caption_embedding = word_model(caption_vec)
			best_video_indices = get_best_index(vid_embeddings, caption_embedding)
			best_vid_ids = []
			best_timestamps = []
			for i in best_video_indices:
				best_vid_ids.append(vid_ids[i])
				best_timestamps.append(timestamps[i])

		#Format urls and stuff









if __name__ == '__main__':
	main()
