import argparse, os


#######################################
def basic_training_parameters(parser):
    ##### Dataset-related Parameters
    parser.add_argument('--dataset',              default='online_products',   type=str,   help='Dataset to use. Currently supported: online_products.')
    parser.add_argument('--use_tv_split',         action='store_true',            help='Flag. If set, split the training set into a training/validation set.')
    parser.add_argument('--tv_split_by_samples',  action='store_true',            help='Flag. If set, create the validation set by taking a percentage of samples PER class. \
                                                                                        Otherwise, the validation set is create by taking a percentage of classes.')
    parser.add_argument('--tv_split_perc',        default=0.8,      type=float, help='Percentage with which the training dataset is split into training/validation.')
    parser.add_argument('--augmentation',         default='base',   type=str,   help='Type of preprocessing/augmentation to use on the data.  \
                                                                                      Available: base (standard), adv (with color/brightness changes), big (Images of size 256x256), red (No RandomResizedCrop).')


    ### General Training Parameters
    parser.add_argument('--lr',                default=0.00001,  type=float,        help='Learning Rate for network parameters.')
    parser.add_argument('--fc_lr',             default=-1,       type=float,        help='Optional. If not -1, sets the learning rate for the final linear embedding layer.')
    parser.add_argument('--decay',             default=0.0004,   type=float,        help='Weight decay placed on network weights.')
    parser.add_argument('--n_epochs',          default=150,      type=int,          help='Number of training epochs.')
    parser.add_argument('--kernels',           default=6,        type=int,          help='Number of workers for pytorch dataloader.')
    parser.add_argument('--bs',                default=112 ,     type=int,          help='Mini-Batchsize to use.')
    parser.add_argument('--seed',              default=1,        type=int,          help='Random seed for reproducibility.')
    parser.add_argument('--scheduler',         default='step',   type=str,          help='Type of learning rate scheduling. Currently supported: step')
    parser.add_argument('--gamma',             default=0.3,      type=float,        help='Learning rate reduction after tau epochs.')
    parser.add_argument('--tau',               default=[120,220,250,280], nargs='+',type=int , help='Stepsize before reducing learning rate.')

    ##### Loss-specific Settings
    parser.add_argument('--optim',           default='adam',        type=str,   help='Optimization method to use. Currently supported: adam & sgd.')
    parser.add_argument('--loss',            default='margin',      type=str,   help='Training criteria: For supported methods, please check criteria/__init__.py')

    ##### Network-related Flags
    parser.add_argument('--embed_dim',        default=128,         type=int,                    help='Embedding dimensionality of the network. Note: dim = 64, 128 or 512 is used in most papers, depending on the architecture.')
    parser.add_argument('--not_pretrained',   action='store_true',                              help='Flag. If set, no ImageNet pretraining is used to initialize the network.')
    parser.add_argument('--arch',             default='resnet50_frozen_normalize',  type=str,   help='Underlying network architecture. Frozen denotes that \
                                                                                                  exisiting pretrained batchnorm layers are frozen, and normalize denotes normalization of the output embedding.')

    ##### Evaluation Parameters
    parser.add_argument('--no_train_metrics', action='store_true',   help='Flag. If set, evaluation metrics are not computed for the training data. Saves a forward pass over the full training dataset.')
    parser.add_argument('--evaluate_on_gpu',  action='store_true',   help='Flag. If set, all metrics, when possible, are computed on the GPU (requires Faiss-GPU).')
    parser.add_argument('--evaluation_metrics', nargs='+', default=['e_recall@1', 'e_recall@10', 'e_recall@100', 'e_recall@1000', 'e_precision@10', 'e_precision@50', 'e_precision@100','nmi', 'f1', 'mAP_1000', 'mAP_lim', 'mAP_c', \
                                                                    'c_recall@1', 'c_recall@2', 'c_recall@4','c_precision@10', 'c_precision@50', 'c_precision@100',\
                                                                    'dists@intra', 'dists@inter', 'dists@intra_over_inter', 'rho_spectrum@0', \
                                                                    'rho_spectrum@-1', 'rho_spectrum@1', 'rho_spectrum@2', 'rho_spectrum@10'], type=str, help='Metrics to evaluate performance by.')

    parser.add_argument('--storage_metrics',    nargs='+', default=['e_recall@1'],     type=str, help='Improvement in these metrics on a dataset trigger checkpointing.')
    parser.add_argument('--evaltypes',          nargs='+', default=['discriminative'], type=str, help='The network may produce multiple embeddings (ModuleDict, relevant for e.g. DiVA). If the key is listed here, the entry will be evaluated on the evaluation metrics.\
                                                                                                       Note: One may use Combined_embed1_embed2_..._embedn-w1-w1-...-wn to compute evaluation metrics on weighted (normalized) combinations.')


    ##### Setup Parameters
    parser.add_argument('--gpu',          default=[0], nargs='+',                  type=int, help='Gpu to use.')
    parser.add_argument('--savename',     default='group_plus_seed',               type=str, help='Run savename - if default, the savename will comprise the project and group name (see wandb_parameters()).')
    parser.add_argument('--source_path',  default=os.getcwd()+'/../../Datasets',   type=str, help='Path to training data.')
    parser.add_argument('--save_path',    default=os.getcwd()+'/Training_Results', type=str, help='Where to save everything.')
    parser.add_argument('--multi-gpu',    action='store_true', help='multi gpu?')
    parser.add_argument('--resume',    default=None, type=str, help='Where to load checkpoint')


    return parser



#######################################
def wandb_parameters(parser):
    ### Online Logging/Wandb Log Arguments
    parser.add_argument('--log_online',      action='store_true',            help='Flag. If set, run metrics are stored online in addition to offline logging. Should generally be set.')
    parser.add_argument('--wandb_key',       default='<your_api_key_here>',  type=str,   help='API key for W&B.')
    parser.add_argument('--project',         default='Sample_Project',       type=str,   help='Name of the project - relates to W&B project names. In --savename default setting part of the savename.')
    parser.add_argument('--group',           default='Sample_Group',         type=str,   help='Name of the group - relates to W&B group names - all runs with same setup but different seeds are logged into one group. \
                                                                                               In --savename default setting part of the savename.')
    return parser



#######################################
def loss_specific_parameters(parser):
    ### Contrastive Loss
    parser.add_argument('--anneal',             default=0.01,  type=float, help='Learning rate on class proxies.')

    ##variant
    parser.add_argument('--variant',             default='PNP-O',  type=str, help='variant type.')

    ##normalization size
    parser.add_argument('--alpha',             default=  1,  type=int, help='alpha ')


    parser.add_argument('--b',             default=  2,  type=float, help='b')
  
    parser.add_argument('--margin',             default=0.5,  type=float, help='pnp triplet margin')

    return parser



#######################################
def batchmining_specific_parameters(parser):
    ### Distance-based Batchminer
    parser.add_argument('--miner_distance_lower_cutoff', default=0.5, type=float, help='Lower cutoff on distances - values below are sampled with equal prob.')
    parser.add_argument('--miner_distance_upper_cutoff', default=1.4, type=float, help='Upper cutoff on distances - values above are IGNORED.')
    ### Spectrum-Regularized Miner (as proposed in our paper) - utilizes a distance-based sampler that is regularized.
    parser.add_argument('--miner_rho_distance_lower_cutoff', default=0.5, type=float, help='Lower cutoff on distances - values below are sampled with equal prob.')
    parser.add_argument('--miner_rho_distance_upper_cutoff', default=1.4, type=float, help='Upper cutoff on distances - values above are IGNORED.')
    parser.add_argument('--miner_rho_distance_cp',           default=0.2, type=float, help='Probability to replace a negative with a positive.')
    return parser


#######################################
def batch_creation_parameters(parser):
    parser.add_argument('--data_sampler',              default='class_random', type=str, help='How the batch is created. Available options: See datasampler/__init__.py.')
    parser.add_argument('--samples_per_class',         default=2,              type=int, help='Number of samples in one class drawn before choosing the next class. Set to >1 for tuple-based loss.')
    ### Batch-Sample Flags - Have no relevance to default SPC-N sampling
    parser.add_argument('--data_batchmatch_bigbs',     default=512,            type=int, help='Size of batch to be summarized into a smaller batch. For distillation/coreset-based methods.')
    parser.add_argument('--data_batchmatch_ncomps',    default=10,             type=int, help='Number of batch candidates that are evaluated, from which the best one is chosen.')
    parser.add_argument('--data_storage_no_update',    action='store_true',              help='Flag for methods that need a sample storage. If set, storage entries are NOT updated.')
    parser.add_argument('--data_d2_coreset_lambda',    default=1, type=float,            help='Regularisation for D2-coreset.')
    parser.add_argument('--data_gc_coreset_lim',       default=1e-9, type=float,         help='D2-coreset value limit.')
    parser.add_argument('--data_sampler_lowproj_dim',  default=-1, type=int,             help='Optionally project embeddings into a lower dimension to ensure that greedy coreset works better. Only makes a difference for large embedding dims.')
    parser.add_argument('--data_sim_measure',          default='euclidean', type=str,    help='Distance measure to use for batch selection.')
    parser.add_argument('--data_gc_softened',          action='store_true', help='Flag. If set, use a soft version of greedy coreset.')
    parser.add_argument('--data_idx_full_prec',        action='store_true', help='Deprecated.')
    parser.add_argument('--data_mb_mom',               default=-1, type=float, help='For memory-bank based samplers - momentum term on storage entry updates.')
    parser.add_argument('--data_mb_lr',                default=1,  type=float, help='Deprecated.')

    return parser
