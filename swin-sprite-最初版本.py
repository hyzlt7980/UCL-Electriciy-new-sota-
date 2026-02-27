import os, math, random, time, glob, torch, kagglehub
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from timm.models.swin_transformer import SwinTransformerBlock

# ==========================================
# 0. 环境配置与加速
# ==========================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# Removed 'expandable_segments:True' to stop the warning spam on your Ubuntu setup

# ==========================================
# 1. 核心组件：RevIN (保持不变)
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine: self._init_params()

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(len(x.shape)-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine: x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.affine_weight)
        x = x * self.stdev + self.mean
        return x

# ==========================================
# 2. 强强联手架构：Swin-iFold
# ==========================================
class Swin_iFold(nn.Module):
    def __init__(self, seq_len=96, pred_len=96, num_vars=321, embed_dim=128, depth=2):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_vars = num_vars
        self.embed_dim = embed_dim
        self.H, self.W = 8, 12  # 8*12=96

        self.revin = RevIN(num_vars)
        
        # --- Stage 1: Swin 纹理识别 (每个变量独立) ---
        self.patch_embed = nn.Linear(1, embed_dim)
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=embed_dim, 
                input_resolution=(self.H, self.W),
                num_heads=8, 
                window_size=4,
                shift_size=0 if (i % 2 == 0) else 2
            ) for i in range(depth)
        ])
        
        # --- Stage 2: iTransformer 变量间博弈 (Global Attention) ---
        self.variable_attention = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True
        )
        
        self.head = nn.Linear(embed_dim, pred_len)

    def forward(self, x):
        # x: [B, 96, 321]
        x = self.revin(x, 'norm')
        B, L, N = x.shape
        
        # Step 1: 展开进入 Swin
        # [B, 96, 321] -> [B*N, 96, 1]
        x = x.permute(0, 2, 1).reshape(B * N, L, 1)
        x = self.patch_embed(x) # [B*N, 96, C]
        
        # Swin 独立提取
        x = x.view(B * N, self.H, self.W, self.embed_dim)
        for blk in self.swin_blocks:
            x = blk(x)
        
        # 每个变量压缩为“特征指纹”
        feat = x.mean(dim=(1, 2)) # [B*N, C]
        
        # Step 2: 变量间博弈 (Inverted Attention)
        feat = feat.view(B, N, self.embed_dim) # [B, 321, C]
        feat = self.variable_attention(feat)    # 让321个变量互相看
        
        # Step 3: 映射未来
        out = self.head(feat) # [B, 321, 96]
        
        out = out.permute(0, 2, 1) # [B, 96, 321]
        out = self.revin(out, 'denorm')
        return out

# ==========================================
# 3. 数据集定义 (基于你提供的 CSV 逻辑)
# ==========================================
class ElectricityDataset(Dataset):
    def __init__(self, csv_path, flag='train'):
        self.seq_len = 96; self.pred_len = 96
        df_raw = pd.read_csv(csv_path)
        df_data = df_raw.iloc[:, 1:] # 去掉时间列
        
        n = len(df_data)
        num_train = int(n * 0.7); num_test = int(n * 0.2); num_val = n - num_train - num_test
        
        # 定义边界
        border1s = [0, num_train - self.seq_len, n - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, n]
        
        f_idx = {'train': 0, 'val': 1, 'test': 2}[flag]
        b1, b2 = border1s[f_idx], border2s[f_idx]
        
        # 标准化
        self.scaler = StandardScaler()
        train_data = df_data.iloc[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)
        
        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]
    
    def __getitem__(self, index):
        s_begin = index; s_end = s_begin + self.seq_len
        r_begin = s_end; r_end = r_begin + self.pred_len
        return torch.tensor(self.data_x[s_begin:s_end], dtype=torch.float32), \
               torch.tensor(self.data_y[r_begin:r_end], dtype=torch.float32)

    def __len__(self): return len(self.data_x) - self.seq_len - self.pred_len + 1

# ==========================================
# 4. 主训练循环 (支持 4 卡 4090)
# ==========================================
def main():
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(rank)
        dist.init_process_group("nccl"); world_size = dist.get_world_size()
    else: rank = 0; world_size = 1; torch.cuda.set_device(0); dist.init_process_group("gloo", rank=0, world_size=1)

    log_file = "training_log.csv"

    # 数据下载 (安全的 Rank 0 逻辑)
    dataset_path_file = ".kaggle_path.tmp"
    if rank == 0:
        print("📥 Downloading dataset...")
        path = kagglehub.dataset_download("tylerfarnan/itransformer-datasets")
        with open(dataset_path_file, "w") as f: f.write(path)
        
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("Epoch,Train_Loss,Val_MSE\n")
    
    if dist.is_initialized(): dist.barrier()
    
    with open(dataset_path_file, "r") as f:
        base_path = f.read().strip()
    csv_path = glob.glob(os.path.join(base_path, "**", "electricity.csv"), recursive=True)[0]

    # DataLoader - 保持你完美的 BS 逻辑
    BS = 8 // world_size
    train_ds = ElectricityDataset(csv_path, 'train')
    val_ds = ElectricityDataset(csv_path, 'val')
    
    train_loader = DataLoader(train_ds, batch_size=BS, sampler=DistributedSampler(train_ds) if world_size > 1 else None, shuffle=(world_size==1), num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BS, sampler=DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None, shuffle=False)

    # 模型实例化 - 保持你的参数
    model = Swin_iFold(num_vars=321, embed_dim=128, depth=4).to(rank)
    if world_size > 1: model = DDP(model, device_ids=[rank])

    # 保持你原汁原味的优化器、调度器和 HuberLoss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.HuberLoss() # 对电价异常值更鲁棒

    if rank == 0: print(f"🚀 Swin-iFold 启动 | 变量博弈模式开启 | 4090 并行中")

    for epoch in range(30):
        if world_size > 1: train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, disable=(rank != 0))
        for bx, by in pbar:
            bx, by = bx.to(rank), by.to(rank)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(bx)
                loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 验证逻辑 (保持你的 MSE 统计)
        model.eval()
        v_mse = 0; count = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(rank), by.to(rank)
                pred = model(bx)
                v_mse += F.mse_loss(pred, by).item()
                count += 1
        
        avg_mse = torch.tensor(v_mse/count).to(rank)
        if world_size > 1: dist.all_reduce(avg_mse, op=dist.ReduceOp.SUM); avg_mse /= world_size

        if rank == 0:
            avg_train_loss = train_loss/len(train_loader)
            print(f"✅ Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val MSE: {avg_mse.item():.4f}")
            
            # 记录到 CSV
            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_mse.item():.4f}\n")
                
            torch.save(model.state_dict(), "swin_ifold_best.pth")
        
        scheduler.step()

if __name__ == "__main__": main()
