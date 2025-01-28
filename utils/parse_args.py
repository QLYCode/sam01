import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='./data/ACDC', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='ACDC_pCE', help='experiment_name')
    parser.add_argument('--fold', type=str,
                        default='fold1', help='cross validation')
    parser.add_argument('--sup_type', type=str,
                        default='scribble', help='supervision type')
    # progressmix unet
    parser.add_argument('--model', type=str,
                        default='efficientumamba', help='model_name')

    parser.add_argument('--num_classes', type=int,  default=4,
                        help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.03,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[256, 256],
                        help='patch size of network input')
    parser.add_argument('--seed', type=int,  default=2022, help='random seed')
    args = parser.parse_args()

            
    return args