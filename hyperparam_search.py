import numpy as np
import train
import utils
import os
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def search():
    args = parser.parse_args()
    args.model_dir = '.'
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.num_epochs = 30

    learning_rates = [0.1]
    margins = [0.3]
    hidden_dims = [256]
    reg_strengths = [0.000008]#generate_values(3, -6, -2)

    best_val_prctile = float("-inf")
    best_val_dist_diff = -1
    opt_rate = 0
    opt_margin = 0
    opt_hidden_dim = 0
    opt_reg_strength = 0
    for rate in learning_rates:
        for margin in margins:
            for hidden_dim in hidden_dims:
                for reg_strength in reg_strengths:
                    params.margin = float(margin)
                    params.learning_rate = float(rate)
                    params.hidden_dim = hidden_dim
                    params.reg_strength = float(reg_strength)

                    avg_prctile, avg_dist_diff, word_model, vid_model, dist_matrix, phrases, ids = train.main(params, args)
                    print("LR: {}, Margin: {}, Hidden Dim: {}, Reg Strength: {}, Avg Percentile: {}, Avg P - N dist: {}".format(rate, margin, hidden_dim, reg_strength, avg_prctile, avg_dist_diff))
                    if avg_prctile > best_val_prctile:
                        opt_rate = rate
                        opt_margin = margin
                        opt_hidden_dim = hidden_dim
                        opt_reg_strength = reg_strength
                        best_val_prctile = avg_prctile
                        best_val_dist_diff = avg_dist_diff
                        torch.save(word_model, 'word_best.pt')
                        torch.save(vid_model, 'vid_best.pt')
                        np.save('best_dist_matrix.npy', dist_matrix)
                        

    print('Optimal Learning Rate: %f\nOptimal Margin: %f\nOptimal hidden_dim: %d\nOptimal Regularization: %f' % (opt_rate, opt_margin, opt_hidden_dim, opt_reg_strength))

def generate_values(num, low, high):
    log_vals = np.random.rand(num)
    log_vals = low + (log_vals * (high - low))
    return np.power(10, log_vals)

search()