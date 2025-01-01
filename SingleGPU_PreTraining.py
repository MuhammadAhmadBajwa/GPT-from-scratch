import matplotlib.pyplot as plt
import os
import torch
import urllib.request
import tiktoken
import os
import time 
import math

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers,pin_memory=True)

    return dataloader



class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec



class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def get_lr(iteration,max_lr,min_lr,max_steps,warmup_steps):
    
    if iteration < warmup_steps:
        return max_lr * (iteration+1)/warmup_steps
    if iteration > max_steps:
        return min_lr
    decay_ratio = (iteration-warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_ratio <= 1
    coeff = 0.5 * (1.0+math.cos(math.pi*decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(model,optimizer,global_step,prev_time,file_path='checkpoint.pth'):
    print("Saving CheckPoints ...") 
    if os.path.exists("/kaggle/working/checkpoint.pth"):
        os.remove("/kaggle/working/checkpoint.pth")
    
    model.eval()
    checkpoint = {
        'model_state_dict': model.state_dict()   ,      # Save Model state
        'optimizer_state_dict': optimizer.state_dict(), # Save Optimizer state
        'step': global_step,  # Current step
        'random_state': torch.random.get_rng_state(),  # Random state for reproducibility
        'prev_time' : prev_time
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at step {global_step} to {file_path}")
    print(50*"=")
    model.train()

def load_checkpoint(model, optimizer, file_path="checkpoint.pth"):
    print("Loading CheckPoints ...")
    # map_location = torch.device('cuda')
    checkpoint = torch.load(file_path,weights_only=True)
    
    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])
    # Restore optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore random state
    torch.random.set_rng_state(checkpoint['random_state'])
    
    step = checkpoint['step']
    prev_time = checkpoint['prev_time']
    print(f"Checkpoint loaded from {file_path}, resuming at step {step}")
    return step , prev_time


def evaluate(model,train_loader,val_loader,eval_iter,global_step,max_steps,start,epoch,device,prev_time):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
   
        print(f"Ep {epoch+1} (Step {global_step:06d}): "
            f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        curr_time = (time.time() - start) + prev_time
        ETA = ((curr_time/global_step)*(max_steps-global_step))/3600
        print(f"ETA = {ETA} hours")
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,checkpoint_step,
                       batch_size,micro_batch_size,checkpoint_path):
    
    global_step = 0
    start = time.time()
    prev_time = 0
    grad_accum_steps = batch_size//micro_batch_size
    
    max_lr = optimizer.param_groups[0]["lr"]
    min_lr = 0.1 * max_lr
    per_epoch_steps = len(train_loader)//grad_accum_steps
    max_steps = per_epoch_steps * num_epochs
    warmup_steps = int(0.1 * max_steps)

    print(f"Total Steps = {max_steps}")

    # Load Checkpoint if exists
    try:
        global_step , prev_time = load_checkpoint(model, optimizer,checkpoint_path)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")
    except :
        print("Starting from scratch, checkpoint didn't match the current architecture")

    curr_epoch = global_step // per_epoch_steps
    for _ in range((global_step%per_epoch_steps)-1):
        next(iter(train_loader))

    # Main training loop
    for epoch in range(num_epochs-curr_epoch):
        epoch += curr_epoch
        model.train()  # Set model to training mode
        
        for _ in range(max_steps//num_epochs):
            
            # optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            for param in model.parameters():
                 param.grad = None
                
            # Gradient Accumulation to overcome small batch size problem
            for _ in range(grad_accum_steps):
                input_batch, target_batch = next(iter(train_loader))
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss = loss / grad_accum_steps
                loss.backward()  # Calculate loss gradients
                
            # Learning Rate Update 
            lr = get_lr(global_step,max_lr,min_lr,max_steps,warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)   # Gradient Clipping
            
            optimizer.step()  # Update model weights using loss gradients
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                evaluate(model,train_loader,val_loader,eval_iter,global_step,max_steps,start,epoch,device,prev_time)

            # Save checkpoints
            if global_step % checkpoint_step == 0:
                curr_time = (time.time() - start) + prev_time
                save_checkpoint(model,optimizer,global_step,curr_time)
            
               
        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )



def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()

    
def main(gpt_config, settings):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = 'checkpoint.pth'
    ##############################
    # Download data if necessary
    ##############################

    file_path = "/kaggle/input/plain-text-wikipedia-simpleenglish/AllCombined.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes

    # model = torch.compile(model)   # compile model for efficiency (Doesnt work for old gpus)

    # Define decayed and non-decayed parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': settings["weight_decay"]},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_parameters,betas=(0.9,0.95),eps=1e-8, lr=settings["learning_rate"], 
        weight_decay=settings["weight_decay"], foreach = True
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["micro_batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["micro_batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=10, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer,
        checkpoint_step = 20 , batch_size = settings["batch_size"],
        micro_batch_size = settings["micro_batch_size"],
        checkpoint_path=checkpoint_path
    )

    return model


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50264,    # Vocabulary size
        "context_length": 1024,  # Shortened context length (orig: 1024)
        "emb_dim": 1024,         # Embedding dimension
        "n_heads": 16,          # Number of attention heads
        "n_layers": 16,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 3e-4,
        "num_epochs": 10,
        "batch_size": 64,
        "weight_decay": 0.1,
        "micro_batch_size": 2   # Set micro batch according to your gpu memory
    }

    ###########################
    # Initiate training
    ###########################

    model = main(GPT_CONFIG_124M, OTHER_SETTINGS)
