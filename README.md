## Train the models: train_weakly_supervised_segmentation_pCE_ours_proposed.py
model: dynamicmix cyclemix  progressmix
parser.add_argument('--model', type=str,
                    default='dynamicmix', help='model_name')
## Test the models: test_2D_fully.py

