import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist 
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import deepspeed 

def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval',
                        type=int,
                        default=2000,
                        help="output logging information at a given interval")

    parser.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')

    parser.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')
    parser.add_argument('--num-experts',
                        type=int,
                        nargs='+',
                        default=[
                            1,
                        ],
                        help='number of experts list, MoE related.')
    parser.add_argument(
        '--mlp-type',
        type=str,
        default='standard',
        help=
        'Only applicable when num-experts > 1, accepts [standard, residual]')
    parser.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')
    parser.add_argument(
        '--min-capacity',
        default=0,
        type=int,
        help=
        '(moe) minimum capacity of an expert regardless of the capacity_factor'
    )
    parser.add_argument(
        '--noisy-gate-policy',
        default=None,
        type=str,
        help=
        '(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'
    )
    parser.add_argument(
        '--moe-param-group',
        default=False,
        action='store_true',
        help=
        '(moe) create separate moe param groups, required when using ZeRO w. MoE'
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args

def get_reviews(review_path="/scratch/hpc72a03/review_dataset/Reviews.csv"):
    df = pd.read_csv (review_path)  
    df = df[:600]
    print(df)
    print(len(df))
    df.dropna(inplace=True)
    reviews = df.Text.copy() 
    return reviews

class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:

      encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 


world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["SLURM_PROCID"])
os.environ['MASTER_PORT'] = os.environ['TRAINER_PORT']
device_id = os.environ["SLURM_LOCALID"]
os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
torch.cuda.set_device(int(device_id))
dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False, dist_init_required =False)

ngpus_per_node = torch.cuda.device_count()
device_id = rank%ngpus_per_node
torch.cuda.set_device(device_id)

args = add_argument()


reviews = get_reviews()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
train_dataset = GPT2Dataset(reviews, tokenizer)
train_loader = torch.utils.data.DataLoader(train_dataset , batch_size=2, shuffle=False, num_workers=20)
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
model.resize_token_embeddings(len(tokenizer))

model.cuda()
model.train()

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=model, optimizer=optimizer, model_parameters=parameters, training_data=train_dataset)

for batch in tqdm(train_loader):
    b_input_ids = batch[0].to(model_engine.local_rank)
    b_labels = batch[0].to(model_engine.local_rank)
    b_masks = batch[1].to(model_engine.local_rank)


    output = model_engine( b_input_ids,
                        labels=b_labels, 
                        attention_mask = b_masks,
                        token_type_ids=None
                      )
    


    loss = output[0] 
    if(rank == 0):
      print(loss)
    print(f"after forward allocated memory size is .. {torch.cuda.memory_allocated() / 1024 /1024}")   
    model_engine.backward(loss)
    model_engine.step()
