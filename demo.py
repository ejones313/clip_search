import torch
import numpy as np
import webbrowser

#NEED: Words to embeddings, vid_model, word_model, original_video_embeddings, vid_to_url_map

def main():
	original_video_embeddings = 
	word_model = 
	vid_model = 
	vid_to_url_map = 

	#Get indexing working
	vid_outputs = vid_model(original_video_embeddings)

	caption_input = ""
	while caption_input.lower() != "q":
		caption_input = input("Enter a caption (q to quit): ")
		#preprocess input
		words = caption_input.split('')
		






if __name__ == '__main__':
	main()