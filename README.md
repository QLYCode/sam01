## Train the models: train_weakly_supervised_segmentation_pCE_ours_proposed.py
model: dynamicmix cyclemix  progressmix
parser.add_argument('--model', type=str,
                    default='dynamicmix', help='model_name')
## Test the models: test_2D_fully.py

## Train other semi- and weakly- methods
[train_weakly_supervised_ustm_2D.py](train_weakly_supervised_ustm_2D.py)
[train_ACDC_scribblevc.py](train_ACDC_scribblevc.py)
[train_deep_adversarial_network_2D.py](train_deep_adversarial_network_2D.py)
[train_entropy_minimization_2D.py](train_entropy_minimization_2D.py)
[train_fully_supervised_2D.py](train_fully_supervised_2D.py)
[train_mean_teacher_2D.py](train_mean_teacher_2D.py)
[train_MSCMR_scribblevc.py](train_MSCMR_scribblevc.py)
[train_partially_fully_supervised.py](train_partially_fully_supervised.py)
[train_s2l.py](train_s2l.py)
[train_uncertainty_aware_mean_teacher_2D.py](train_uncertainty_aware_mean_teacher_2D.py)
[train_weakly_supervised_pCE_2D.py](train_weakly_supervised_pCE_2D.py)
[train_weakly_supervised_pCE_Entropy_Mini_2D.py](train_weakly_supervised_pCE_Entropy_Mini_2D.py)
[train_weakly_supervised_pCE_GatedCRFLoss_2D.py](train_weakly_supervised_pCE_GatedCRFLoss_2D.py)
[train_weakly_supervised_pCE_Inter&Intra_Class_2D.py](train_weakly_supervised_pCE_Inter%26Intra_Class_2D.py)
[train_weakly_supervised_pCE_MumfordShah_Loss_2D.py](train_weakly_supervised_pCE_MumfordShah_Loss_2D.py)
[train_weakly_supervised_pCE_random_walker_2D.py](train_weakly_supervised_pCE_random_walker_2D.py)
[train_weakly_supervised_pCE_TV_2D.py](train_weakly_supervised_pCE_TV_2D.py)