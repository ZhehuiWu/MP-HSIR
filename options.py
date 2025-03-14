import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=32,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')
parser.add_argument('--init', type=str, default='xu',
                    help='which init scheme to choose.', choices=['kn', 'ku', 'xn', 'xu'])

parser.add_argument('--mode', type=int, default=0, help='Degraded Mode.')
parser.add_argument('--natural_scene_single_de_type', nargs='+', default=['gaussianN', 'complexN', 'blur', 'sr', 'inpaint', 'bandmiss'],
                    help='which type of single degradation is training and testing for.')
parser.add_argument('--remote_sensing_single_de_type', nargs='+', default=['gaussianN', 'complexN', 'blur', 'sr', 'inpaint', 'haze', 'bandmiss'],#'gaussianN', 'complexN', 'blur', 'sr', 'inpaint', 'haze', 'bandmiss'
                    help='which type of single degradation is training and testing for.')

parser.add_argument('--patch_size', type=int, default=64, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

parser.add_argument('--data_type',type=str,default= "remote_sensing",help = "Types of data used for training.")
parser.add_argument('--classifier',type=bool,default= False,help = "")

parser.add_argument('--db_path', type=str, default='',
                    help='where clean HSIs of remote_sensing saves.')

parser.add_argument('--classifier_path', type=str, default='',
                    help='')

parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default= None, help='checkpoint save path')

parser.add_argument("--ckpt_dir",type=str,default='',help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=list,default= [0],help = "Number of GPUs to use for training")
parser.add_argument("--repeat",type=int,default= 1,help = "")

options = parser.parse_args()

