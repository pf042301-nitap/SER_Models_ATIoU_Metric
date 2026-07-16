import argparse
import os
import time

import torch

import model
from beer import BeerData, BeerAnnotation
from movies import MovieData, MovieAnnotation
from embedding import get_embeddings,get_glove_embedding
from torch.utils.data import DataLoader

from model import GenEncShareModel,GenEncNoShareModel, Teacher
from train_util import train_teacher,train_share
from validate_util import validate_share, validate_dev_sentence, validate_annotation_sentence, validate_rationales,validate_teacher
from tensorboardX import SummaryWriter


def parse():
    parser = argparse.ArgumentParser(
        description="tr")

    # dataset parameters
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--save',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--data_dir',
                        type=str,
                        default='/media/dinesh/4TBHDD/Misc_Downloads_25102025/Programs_Documents/noise_injection_crf/data_preprocessing/usr_movie_review/original',
                        help='Path of the dataset')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='/media/dinesh/4TBHDD/Misc_Downloads_25102025/Programs_Documents/noise_injection_crf/data_preprocessing/usr_movie_review/original/test.jsonl',
                        help='Path to the annotation')
    parser.add_argument('--max_length',
                        type=int,
                        default=256,
                        help='Max sequence length [default: 256]')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size [default: 100]')
    # pretrained embeddings
    parser.add_argument('--embedding_dir',
                        type=str,
                        default='./data/embeddings',
                        help='Dir. of pretrained embeddings [default: None]')
    parser.add_argument('--embedding_name',
                        type=str,
                        default='glove.6B.100d.txt',
                        help='File name of pretrained embeddings [default: None]')

    # model parameters
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
                        default=450,
                        help='Number of training epoch')
    parser.add_argument('--pre_epoch',
                        type=int,
                        default=300,
                        help='Number of training epoch')
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
    parser.add_argument(
        '--tea_lambda',
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
    parser.add_argument(
        '--writer',
        type=str,
        default='./noname',
        help='Regularizer to control highlight percentage [default: .2]')
    args = parser.parse_args()
    return args


#####################
# set random seed
#####################


#####################
# parse arguments
#####################
args = parse()

teacher_model_name = (
    f'DAR_tea_model_'
    f'maxlen_{args.max_length}_'
    f'share_{args.share}_'
    f'sparsity_{args.sparsity_percentage}_'
    f'lr_{args.lr}_'
    f'conl_{args.continuity_lambda}_'
    f'spl_{args.sparsity_lambda}.pth'
)

student_model_name = (
    f'DAR_model_'
    f'maxlen_{args.max_length}_'
    f'share_{args.share}_'
    f'sparsity_{args.sparsity_percentage}_'
    f'lr_{args.lr}_'
    f'conl_{args.continuity_lambda}_'
    f'spl_{args.sparsity_lambda}.pth'
)

os.makedirs('./save_model/movie', exist_ok=True)
os.makedirs('./trained_model/movie', exist_ok=True)


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# device
######################
writer=SummaryWriter(args.writer)
torch.cuda.set_device(int(args.gpu))
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")


######################
# load embedding
######################

pretrained_embedding, word2idx = get_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
args.vocab_size = len(word2idx)
args.pretrained_embedding = pretrained_embedding

######################
# load dataset
######################
# train_data = BeerData(args.data_dir, args.aspect, 'train', word2idx, balance=True)

# dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)

# annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)

train_data = MovieData(
    args.data_dir,
    'train',
    word2idx,
    balance=True,
    max_length=args.max_length
)

dev_data = MovieData(
    args.data_dir,
    'dev',
    word2idx,
    max_length=args.max_length
)

annotation_data = MovieAnnotation(
    args.annotation_path,
    word2idx,
    max_length=args.max_length
)

# shuffle and batch the dataset
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

print(len(train_loader.dataset))

dev_loader = DataLoader(dev_data, batch_size=args.batch_size)

annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

#model
tea_model=model.Teacher(args).to(device)
tea_optimizer = torch.optim.Adam(tea_model.parameters(),lr=args.lr)


#pretrain teacher
def pre_tea(epochs):
    best_dev=0
    best_e=0
    for e in range(epochs):
        precision, recall, f1_score, accuracy=train_teacher(tea_model,tea_optimizer,train_loader,device)
        _, _, dev_f1, dev_acc = validate_teacher(tea_model, dev_loader, device)
        _, _, test_f1, test_acc = validate_teacher(tea_model, annotation_loader, device,test=True)
        print('e={},train_f1={:.2f},dev_f1={:.2f},test_f1={:.2f}'.format(e,f1_score.item()*100,dev_f1.item()*100,test_f1.item()*100))
        if best_dev<dev_f1:
            best_dev=dev_f1
            best_e=e
            os.makedirs('./save_model/movie', exist_ok=True)

            torch.save(
                tea_model.state_dict(),
                os.path.join(
                    './save_model/movie',
                    teacher_model_name
                )
            )
    print('pre_acc={},epoch={}'.format(best_dev,best_e))

if args.pre_epoch>0:
    pre_tea(args.pre_epoch)
# tea_model = torch.load(
#     './save_model/movie/tea_model.pth'
# ).to(device).train()
load_path = os.path.join(
    './save_model/movie',
    teacher_model_name
)
tea_model.load_state_dict(torch.load(load_path))
tea_model.to(device).train()

# Training
######################
strat_time=time.time()
best_all = 0
f1_best_dev = [0]
best_dev_epoch = [0]
acc_best_dev = [0]
if args.share==1:
    model = GenEncShareModel(args)
elif args.share==0:
    model = GenEncNoShareModel(args)
else:
    print('please choose share of 0 or 1')
model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
for epoch in range(args.epochs):

    start = time.time()
    model.train()
    precision, recall, f1_score, accuracy = train_share(model, optimizer, train_loader, device, args,(writer,epoch),tea_model)
    end = time.time()
    print('\nTrain time for epoch #%d : %f second' % (epoch, end - start))
    print("traning epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                                   precision, f1_score,
                                                                                                   accuracy))
    writer.add_scalar('train_acc',accuracy,epoch)
    writer.add_scalar('time',time.time()-strat_time,epoch)
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    model.eval()
    print("Validate")
    with torch.no_grad():
        for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            _, logits = model(inputs, masks)
            # pdb.set_trace()
            logits = torch.softmax(logits, dim=-1)
            _, pred = torch.max(logits, axis=-1)
            # compute accuarcy
            # TP predict 和 label 同时为1
            TP += ((pred == 1) & (labels == 1)).cpu().sum()
            # TN predict 和 label 同时为0
            TN += ((pred == 0) & (labels == 0)).cpu().sum()
            # FN predict 0 label 1
            FN += ((pred == 0) & (labels == 1)).cpu().sum()
            # FP predict 1 label 0
            FP += ((pred == 1) & (labels == 0)).cpu().sum()
        precision =torch.true_divide( TP , (TP + FP))
        recall = torch.true_divide(TP , (TP + FN))
        f1_score = torch.true_divide(2 * recall * precision , (recall + precision))
        accuracy = torch.true_divide((TP + TN) , (TP + TN + FP + FN))
        print("dev epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                                   precision,
                                                                                                   f1_score, accuracy))

        writer.add_scalar('dev_acc',accuracy,epoch)
        print("Validate Sentence")
        validate_dev_sentence(model, dev_loader, device,(writer,epoch))
        print("Annotation")
        annotation_results = validate_share(model, annotation_loader, device)
        print(
            "The annotation performance (Token level): sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
            % (annotation_results[0], annotation_results[1],
               annotation_results[2], annotation_results[3]))
        writer.add_scalar('f1',100 * annotation_results[3],epoch)
        writer.add_scalar('sparsity',100 * annotation_results[0],epoch)
        writer.add_scalar('p', 100 * annotation_results[1], epoch)
        writer.add_scalar('r', 100 * annotation_results[2], epoch)
        print("Annotation Sentence")
        validate_annotation_sentence(model, annotation_loader, device)
        print("Rationale")
        validate_rationales(model, annotation_loader, device,(writer,epoch))
        if accuracy>acc_best_dev[-1]:
            acc_best_dev.append(accuracy)
            best_dev_epoch.append(epoch)
            f1_best_dev.append(annotation_results[3])
        if best_all<annotation_results[3]:
            best_all=annotation_results[3]
print(best_all)
print(acc_best_dev)
print(best_dev_epoch)
print(f1_best_dev)

os.makedirs('./trained_model/', exist_ok=True)

# ==============================================================
# ✅ Save only results summary (ignore .pkl)
# ==============================================================
results_filename = (
    f"./trained_model/DAR_results_share_{args.share}"
    f"_sparsity_{args.sparsity_percentage}"
    f"_lr_{args.lr}"
    f"_conl_{args.continuity_lambda}"
    f"_spl_{args.sparsity_lambda}.txt"
)

with open(results_filename, 'w') as f:
    f.write("==== DAR Model Results Summary ====\n")
    f.write(f"share: {args.share}\n")
    f.write(f"sparsity_percentage: {args.sparsity_percentage}\n")
    f.write(f"lr: {args.lr}\n")
    f.write(f"continuity_lambda: {args.continuity_lambda}\n")
    f.write(f"sparsity_lambda: {args.sparsity_lambda}\n")
    f.write(f"best_all: {best_all}\n") # Tracks the best token F1 score
    f.write(f"acc_best_dev: {acc_best_dev}\n") # List that tracks best accuracy scores
    f.write(f"best_dev_epoch: {best_dev_epoch}\n") # List that tracks epochs when best accuracy occurred
    f.write(f"f1_best_dev: {f1_best_dev}\n") # List that tracks best F1 scores

print(f"🧾 Results saved successfully to: {results_filename}")

if args.save==1:
    os.makedirs('./trained_model/movie', exist_ok=True)

    # save_path = (
    #     './trained_model/movie/'
    #     f'DAR_model_share_{args.share}_'
    #     f'sp_{args.sparsity_percentage}.pkl'
    # )

    torch.save(
        model.state_dict(),
        os.path.join(
            './trained_model/movie',
            student_model_name
        )
    )
    print('save the model')
else:
    print('not save')


