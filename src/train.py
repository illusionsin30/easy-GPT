# coding: utf-8
import argparse
import math
import torch
import torch.optim as optim
import torch.nn as nn

import data
import model
import wandb

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=16, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=256,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = args.cuda

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size, 'valid': eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)

########################################
# Build LMModel model (bulid your language model here)
model = model.CausalLMM(vocab_size=len(data_loader.vocabulary), dim=args.emb_dim,
                        max_seq_len=args.max_sql, device=device,
                        num_layers=args.num_layers, num_heads=args.num_heads)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

criterion = nn.CrossEntropyLoss()


# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.
args.optimizer = optimizer
args.criterion = criterion
wandb.init(
    project="PRML-project2",
    config=args,
    name=f"epochs{args.epochs}-lr{args.lr}-num_layers{args.num_layers}-num_heads{args.num_heads}-embed_size{args.emb_dim}-train_bsz{args.train_batch_size}-eval_bsz{args.eval_batch_size}",
    settings=wandb.Settings(console="auto")
)

def evaluate():
    data_loader.set_valid()
    data, target, end_flag = data_loader.get_batch()
    model.eval()
    idx = 0
    avg_loss = 0
    print(f"Validating")
    while not end_flag:
        with torch.no_grad():
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            logits = outputs["logits"]

            # Calculate cross-entropy loss
            loss = criterion(logits.view(-1, logits.size(-1)), target)
            avg_loss += loss
            idx += 1
    wandb.log({"validating/loss": avg_loss / idx})
    print(f"The average loss is {avg_loss / idx}")
    return math.exp(avg_loss.item() / idx)


# Train Function
def train():
    data_loader.set_train()
    data, target, end_flag = data_loader.get_batch()
    model.train()
    idx = 0
    avg_loss = 0
    while not end_flag:
        data, target, end_flag = data_loader.get_batch()
        data = data.to(device)
        target = target.to(device)
        outputs = model(data)
        logits = outputs["logits"]

        # Calculate cross-entropy loss
        optimizer.zero_grad()
        loss = criterion(logits.view(-1, logits.size(-1)), target)
        loss.backward()
        optimizer.step()
        if (idx + 1) % 10 == 0:
            print(f"Step {idx + 1}, loss: {loss}")
        wandb.log({"training/loss": loss.item()})
        idx += 1
        avg_loss += loss
    return math.exp(avg_loss.item() / idx)


# Loop over epochs.
train_perplexity = []
valid_perplexity = []
for epoch in range(1, args.epochs + 1):
    print(f"Start training epoch {epoch}")
    train_perplexity.append(train())
    valid_perplexity.append(evaluate())


print(f"Train Perpelexity {train_perplexity}")
print(f"Valid Perpelexity {valid_perplexity}")

wandb.finish()