import os, sys
import argparse
import random
import copy
# import cv2
import torch
import torchattacks
import torch.nn as nn

from pathlib import Path
lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))

import setup
import models
from training import *
from utils import split_train_loader 
import warnings
warnings.filterwarnings('ignore')

def process_selftrain(args,model,train_loader, test_loader,attack ,device, local_epoch):
    print("Self-training ...")
    df,finnal_model = run_experiment(model, train_loader, test_loader, attack ,device,local_epoch)

    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_'+args.alg+ '_'+ args.attack+'.csv')
    else:
        outfile = os.path.join(outpath, f'{args.repeat}_accuracy_'+args.alg+ '_' + args.attack + '.csv')
    df.to_csv(outfile)
    print(f"Wrote to file: {outfile}")
    # outpath = './outputs'  # Make sure this is defined

    # outpath1 = os.path.join(args.outbase, 'Models_output')
    outpath1 = os.path.join(args.outbase, f'Model_{args.alg}_attack_{args.attack}.pth')
    # Path(outpath1).mkdir(parents=True, exist_ok=True)
    torch.save(finnal_model.state_dict(), outpath1)
    # print(f"Wrote to Model: {outpath1}")



def process_fedavg(args,model, federated_train_loaders, test_loader,attack ,device):
    print(f'\nDone setting up {args.alg} devices.')

    print(f'Running {args.alg} ...')

    client_df, global_df = run_federated_experiment(args ,model, federated_train_loaders, test_loader,device , attack)

    # Clients info
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_{args.alg}_{args.attack}_clients.csv')
    else:
        outfile = os.path.join(outpath, f'{args.repeat}_accuracy_{args.alg}_{args.attack}_clients.csv')
    client_df.to_csv(outfile)

    # Global info
    if args.repeat is None:
        outfile = os.path.join(outpath, f'accuracy_{args.alg}_{args.attack}_global.csv')
    else:
        outfile = os.path.join(outpath, f'{args.repeat}_accuracy_{args.alg}_{args.attack}_global.csv')
    global_df.to_csv(outfile)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda',
                        help='CPU / GPU device.')
    parser.add_argument('--NONIID', type=int, default= 0 ,
                        help='Data disturibution')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='Name of algorithms.')
    parser.add_argument('--attack', type=str, default="None",
                        help='Name of Attack.')
    parser.add_argument('--epoch', type=int, default=200,
                        help='number of local epochs or iterations that client communicate;')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate for inner solver;')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')#0.1
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for node classification.')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='num_layers of transformer')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Number of hidden_dim units.')
    parser.add_argument('--mlp_dim', type=int, default=2048,
                        help='Number of mlp_dim units.')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of num_heads units.')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--num_channels', type=int, default=3,
                        help='Number of num_channels')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=32,
                        help='Number of patch_size')

    parser.add_argument('--num_clients', type=int, default=5,
                        help='Number of clients')
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--rho', type=float, default=0.01)
    parser.add_argument('--seed', help='seed for randomness;',
                        type=int, default=1)

    parser.add_argument('--datapath', type=str, default='./Data',
                        help='The input path of data.')
    

    parser.add_argument('--network', type=str, default='resnet18',
                        help='')
    

    parser.add_argument('--alpha_non', type=float, default= 0.5 ,
                        help='')
    
    parser.add_argument('--outbase', type=str, default='./outputs',
                        help='The base path for outputting.')
    parser.add_argument('--repeat', help='index of repeating;',
                        type=int, default=None)
    parser.add_argument('--mix_img', help='mix img of test and train data and then seperate',
                        type=bool, default=True)
    parser.add_argument('--base_dir_img', help='Alzheimer_s Dataset',
                        type=str, default='Alzheimer_s Dataset')
    parser.add_argument('--split_data', type=float, default=0.8)

    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))

    # add num of patches    
    num_patches = int(args.image_size**2 / args.patch_size**2)
    args.num_patches = num_patches

    # set seeds
    seed_dataSplit = 123
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # set device
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    # set output path
    outpath = os.path.join(args.outbase, args.alg)
    Path(outpath).mkdir(parents=True, exist_ok=True)
    print(f"Output Path: {outpath}")

    # preparing data
    print("Preparing data ...")
    train_loader, test_loader = setupGC.setup_data(args)
    print("Done")
    
    model = models.get_model(args)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    # model = model.cuda()
    print("\nDone setting up devices.")

    if args.attack == "None":
        attack = None
        print("the robustness is :", "None")
    elif args.attack == "TIFGSM":
        print("the robustness is :", "TIFGSM")
        attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=1, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
    elif args.attack == "Jitter":
        print("the robustness is :", "Jitter")
        attack = torchattacks.Jitter(model, eps=6/255, alpha=2/255, steps=1)
    elif args.attack == "DIFGSM":
        print("the robustness is :", "DIFGSM")
        attack = torchattacks.DIFGSM(model, eps=6/255, alpha=2/255, steps=1, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)

    print(args.attack)


    if args.alg == 'selftrain':
        process_selftrain(args,model,train_loader, test_loader, attack ,device, local_epoch=args.epoch)
    else:
        federated_train_loaders = split_train_loader(train_loader, num_clients=args.num_clients, non_iid=args.NONIID, alpha=args.alpha_non)  # This needs to be implemented
        if args.alg == 'fedavg' or args.alg == 'fedprox' or args.alg == 'fedbn':
            process_fedavg(args,model,federated_train_loaders, test_loader, attack,device)