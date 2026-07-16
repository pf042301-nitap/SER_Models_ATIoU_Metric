#!/usr/bin/env python3
print('The file is mochang.py, the time is:')
import datetime
now = datetime.datetime.now()
bj_time = now + datetime.timedelta(hours=8)
print(bj_time.strftime('%Y-%m-%d %H:%M:%S'))

import argparse
import os
import time

import torch

from beer import BeerData, BeerAnnotation, Beer_correlated
from hotel import HotelData, HotelAnnotation
from embedding import get_embeddings, get_glove_embedding
from torch.utils.data import DataLoader

from model import Mochangmodel
from train_util import train_pred_mochang
from validate_util import validate_share_mochang, validate_dev_sentence, validate_annotation_sentence, validate_rationales

# Limit this process to ~5 GB on your GPU 0
torch.cuda.set_per_process_memory_fraction(0.3, 0)  # 30% of total GPU 0 memory
print("✅ Project 3 limited to ~30% GPU memory")

def parse():
    # Default: nonorm, dis_lr=1, data=beer, save=0
    parser = argparse.ArgumentParser(
        description="SR")
    # Additional parameters for mochang
    parser.add_argument('--mochang_loss',
                        type=str,
                        default='log')
    parser.add_argument('--mochang_lambda',
                        type=float,
                        default=1)
    parser.add_argument('--add',
                        type=float,
                        default=1)
    # pretrained embeddings
    parser.add_argument('--embedding_dir',
                        type=str,
                        default='./data/hotel/embeddings',
                        help='Dir. of pretrained embeddings [default: None]')
    parser.add_argument('--embedding_name',
                        type=str,
                        default='glove.6B.100d.txt',
                        help='File name of pretrained embeddings [default: None]')
    parser.add_argument('--max_length',
                        type=int,
                        default=256,
                        help='Max sequence length [default: 256]')

    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/beer',
                        help='Path of the dataset')
    parser.add_argument('--data_type',
                        type=str,
                        default='beer',
                        help='0:beer,1:hotel')
    parser.add_argument('--correlated',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument('--aspect',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='The aspect number of beer review [20226666,12252018]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='./data/beer/annotations.json',
                        help='Path to the annotation')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size [default: 100]')

    # model parameters
    parser.add_argument('--sp_norm',
                        type=int,
                        default=0,
                        help='0:rnp,1:norm')
    parser.add_argument('--fr',
                        type=int,
                        default=0,
                        help='0:rnp,1:fr')
    parser.add_argument('--dr',
                        type=int,
                        default=0,
                        help='0:rnp,1:dr')
    parser.add_argument('--dis_lr',
                        type=float,
                        default=0,
                        help='0:rnp,1:dis')
    parser.add_argument('--save',
                        type=int,
                        default=1,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--cell_type',
                        type=str,
                        default="GRU",
                        help='Cell type: LSTM, GRU [default: GRU]')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='RNN cell layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Network Dropout')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Embedding dims [default: 100]')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=200,
                        help='RNN hidden dims [default: 100]')
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help='Number of predicted classes [default: 2]')

    # ckpt parameters
    parser.add_argument('--output_dir',
                        type=str,
                        default='./res',
                        help='Base dir of output files')

    # learning parameters
    parser.add_argument('--epochs',
                        type=int,
                        default=37,
                        help='Number of training epoch')
    parser.add_argument('--perturb_rate',
                        type=float,
                        default=0,
                        help='perturb_rate')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=12.,
                        help='Sparsity trade-off [default: 1.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=10.,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument(
        '--sparsity_percentage',
        type=float,
        default=0.1,
        help='Regularizer to control highlight percentage [default: .2]')
    parser.add_argument(
        '--cls_lambda',
        type=float,
        default=0.9,
        help='lambda for classification loss')
    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--share',
                        type=int,
                        default=0,
                        help='')
    
    args = parser.parse_args()
    return args


#####################
# set random seed
#####################
# torch.manual_seed(args.seed)

#####################
# parse arguments
#####################
args = parse()
torch.manual_seed(args.seed)
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# device
######################
torch.cuda.set_device(int(args.gpu))
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(args.seed)

######################
# load embedding
######################
pretrained_embedding, word2idx = get_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
args.vocab_size = len(word2idx)
args.pretrained_embedding = pretrained_embedding

######################
# load dataset
######################
if args.data_type == 'beer':       # beer
    if args.correlated == 0:
        print('decorrelated')
        train_data = BeerData(args.data_dir, args.aspect, 'train', word2idx, balance=True)
        dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)
    else:
        print('correlated')
        train_data = Beer_correlated(args.data_dir, args.aspect, 'train', word2idx, balance=True)
        dev_data = Beer_correlated(args.data_dir, args.aspect, 'dev', word2idx, balance=True)

    annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)
elif args.data_type == 'hotel':       # hotel
    args.data_dir = './data/hotel'
    args.annotation_path = './data/hotel/annotations'
    train_data = HotelData(args.data_dir, args.aspect, 'train', word2idx, balance=True)
    dev_data = HotelData(args.data_dir, args.aspect, 'dev', word2idx)
    annotation_data = HotelAnnotation(args.annotation_path, args.aspect, word2idx)

# shuffle and batch the dataset
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=args.batch_size)
annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

######################
# load model
######################
model = Mochangmodel(args)
model.to(device)

######################
# Training
######################

lr2 = args.lr
lr1 = args.lr

if args.dr == 1:
    # dr, use 10x learning rate
    lr2 = args.lr / 10

if args.fr == 1:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr1)
else:
    g_para = list(map(id, model.generator.parameters()))
    p_para = filter(lambda p: id(p) not in g_para, model.parameters())

    para = [
        {'params': model.generator.parameters(), 'lr': lr1},
        {'params': p_para, 'lr': lr2}
    ]
    optimizer = torch.optim.Adam(para)
print('lr1={}, lr2={}'.format(lr1, lr2))

######################
# Training Loop
######################
strat_time = time.time()
best_all = 0
f1_best_dev = [0]
best_dev_epoch = [0]
acc_best_dev = [0]
mochang_best_dev = [0]
mochang_best_dev_epoch = [0]
f1_best_mochang = [0]
grad = []
grad_loss = []

for epoch in range(args.epochs):

    start = time.time()
    model.train()
    precision, recall, f1_score, accuracy = train_pred_mochang(model, optimizer, train_loader, device, args, epoch,
                                                          grad, grad_loss)
    end = time.time()
    print('\nTrain time for epoch #%d : %f second' % (epoch, end - start))
    print("training epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                                   precision, f1_score,
                                                                                                   accuracy))
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    total_mochang = 0
    model.eval()
    print("Validate")
    with torch.no_grad():
        for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            _, logits, clsoutputs = model.forward_predemb(inputs, masks)
            logits = torch.softmax(logits, dim=-1)
            _, pred = torch.max(logits, axis=-1)
            # compute accuracy
            # TP predict and label both 1
            TP += ((pred == 1) & (labels == 1)).cpu().sum()
            # TN predict and label both 0
            TN += ((pred == 0) & (labels == 0)).cpu().sum()
            # FN predict 0 label 1
            FN += ((pred == 0) & (labels == 1)).cpu().sum()
            # FP predict 1 label 0
            FP += ((pred == 1) & (labels == 0)).cpu().sum()

            mochang = torch.norm(clsoutputs, p=2, dim=1)
            total_mochang += torch.sum(mochang).item()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (recall + precision)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        avg_mochang = total_mochang / (TP + TN + FP + FN)
        print("dev epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f} mochang={:.4f}".format(epoch, recall,
                                                                                                   precision,
                                                                                                   f1_score, accuracy, avg_mochang))

        print("Validate Sentence")
        validate_dev_sentence(model, dev_loader, device, epoch)
        print("Annotation")
        annotation_results = validate_share_mochang(model, annotation_loader, device, epoch)
        print(
            "The annotation performance: sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
            % (100 * annotation_results[0], 100 * annotation_results[1],
               100 * annotation_results[2], 100 * annotation_results[3]))
        print('the mochang on the test set:{:.4f}'.format(annotation_results[4]))
        print("Annotation Sentence")
        validate_annotation_sentence(model, annotation_loader, device)
        print("Rationale")
        validate_rationales(model, annotation_loader, device, epoch)
        if accuracy > acc_best_dev[-1]:
            acc_best_dev.append(accuracy)
            best_dev_epoch.append(epoch)
            f1_best_dev.append(annotation_results[3])
        if best_all < annotation_results[3]:
            best_all = annotation_results[3]

        if avg_mochang > mochang_best_dev[-1]:
            mochang_best_dev.append(avg_mochang)
            mochang_best_dev_epoch.append(epoch)
            f1_best_mochang.append(annotation_results[3])

print("the best f1 is {}".format(best_all))
print("the best dev acc are {}".format(acc_best_dev))
print("the best dev epochs are {}".format(best_dev_epoch))
print("the best f1 at dev acc epochs are {}".format(f1_best_dev))
print("the best f1 at dev mochang epochs are {}".format(f1_best_mochang))

if args.save == 1:
    if args.data_type == 'beer':
        os.makedirs('./trained_model/beer', exist_ok=True)
        # Save with new pattern
        model_name = f"N2R_model_share_{args.share}_sparsity_{args.sparsity_percentage}_lr_{args.lr}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}_aspect_{args.aspect}.pkl"
        torch.save(model.state_dict(), f'./trained_model/beer/{model_name}')
        print('Model saved')
        results_filename = f"./trained_model/beer/N2R_beer_results_share_{args.share}_sparsity_{args.sparsity_percentage}_lr_{args.lr}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}_aspect_{args.aspect}.txt"
        with open(results_filename, 'w') as f:
            f.write("==== N2R Model Results Summary ====\n")
            f.write(f"the best f1 is {best_all}\n")
            f.write(f"the best dev acc are {acc_best_dev}\n")
            f.write(f"the best dev epochs are {best_dev_epoch}\n")
            f.write(f"the best f1 at dev acc epochs are {f1_best_dev}\n")
            f.write(f"the best f1 at dev mochang epochs are {f1_best_mochang}\n")
 
    elif args.data_type == 'hotel':
        os.makedirs('./trained_model/hotel', exist_ok=True)
        # Save with new pattern
        model_name = f"N2R_model_share_{args.share}_sparsity_{args.sparsity_percentage}_lr_{args.lr}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}_aspect_{args.aspect}.pkl"
        torch.save(model.state_dict(), f'./trained_model/hotel/{model_name}')
        print('Model saved')
        results_filename = f"./trained_model/hotel/N2R_hotel_results_share_{args.share}_sparsity_{args.sparsity_percentage}_lr_{args.lr}_conl_{args.continuity_lambda}_spl_{args.sparsity_lambda}_aspect_{args.aspect}.txt"
        with open(results_filename, 'w') as f:
            f.write("==== N2R Model Results Summary ====\n")
            f.write(f"the best f1 is {best_all}\n")
            f.write(f"the best dev acc are {acc_best_dev}\n")
            f.write(f"the best dev epochs are {best_dev_epoch}\n")
            f.write(f"the best f1 at dev acc epochs are {f1_best_dev}\n")
            f.write(f"the best f1 at dev mochang epochs are {f1_best_mochang}\n")
else:
    print('Model not saved')

print(f"Total training time: {time.time() - strat_time:.2f} seconds")