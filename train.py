import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import ImageDataset
from utils.fid_score import calculate_fid
import itertools
import os
from tqdm import tqdm
from pathlib import Path
import glob
from datetime import datetime
from torch.amp import autocast, GradScaler
import numpy as np

# 定义函数
def find_latest_checkpoint():
    """查找最新的检查点文件"""
    checkpoint_files = glob.glob(str(checkpoint_dir / "checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None
    
    # 按文件修改时间排序，返回最新的
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint

def load_checkpoint(checkpoint_path):
    """加载检查点"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # 加载模型状态
    G_AB.load_state_dict(checkpoint['G_AB_state_dict'])
    G_BA.load_state_dict(checkpoint['G_BA_state_dict'])
    D_A.load_state_dict(checkpoint['D_A_state_dict'])
    D_B.load_state_dict(checkpoint['D_B_state_dict'])
    
    # 加载优化器状态
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    # 加载scaler状态
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    return checkpoint['epoch']

def save_checkpoint(epoch, avg_losses, is_final=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'G_AB_state_dict': G_AB.state_dict(),
        'G_BA_state_dict': G_BA.state_dict(),
        'D_A_state_dict': D_A.state_dict(),
        'D_B_state_dict': D_B.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'scaler_state_dict': scaler.state_dict(),  # 保存scaler状态
        'config': config,
        'losses': avg_losses,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    if is_final:
        checkpoint_path = checkpoint_dir / 'final_model.pth'
    else:
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    wandb.save(str(checkpoint_path))
    return checkpoint_path

# 初始化wandb
wandb.init(project="cycle-gan-monet")

# 优化后的配置参数
config = {
    'epochs': 300,
    'batch_size': 8,
    'lr': 0.00008,  # 适度的生成器学习率
    'lr_D': 0.00002,  # 调整判别器学习率
    'b1': 0.5,
    'b2': 0.999,
    'lambda_cycle': 20.0,  # 降回之前的cycle loss权重
    'lambda_identity': 10.0,  # 降回之前的identity loss权重
    'checkpoint_freq': 5,
    'checkpoint_dir': 'checkpoints',
    'resume_training': True,
    'fid_freq': 5,
    'log_freq': 500,
    'gradient_accumulation_steps': 2,
    'warmup_epochs': 2,
    'lr_decay_start_epoch': 30,  # 提前开始学习率衰减
    'lr_decay_epochs': 200,  # 延长衰减周期
    'min_lr': 0.000002,  # 调整最小学习率
    'min_lr_D': 0.0000005,  # 调整判别器最小学习率
    'label_smoothing': 0.05,  # 减小标签平滑程度
}

wandb.config.update(config)

# 创建检查点目录
checkpoint_dir = Path(config['checkpoint_dir'])
checkpoint_dir.mkdir(exist_ok=True)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
G_AB = Generator().to(device)
G_BA = Generator().to(device)
D_A = Discriminator().to(device)
D_B = Discriminator().to(device)

# 初始化损失函数
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# 初始化优化器
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()),
    lr=config['lr'],
    betas=(config['b1'], config['b2'])
)
optimizer_D = torch.optim.Adam(
    itertools.chain(D_A.parameters(), D_B.parameters()),
    lr=config['lr_D'],  # 使用较小的判别器学习率
    betas=(config['b1'], config['b2'])
)

# 初始化混合精度训练的scaler
scaler = torch.amp.GradScaler()

# 优化数据加载
dataset = ImageDataset("data")
dataloader = DataLoader(
    dataset, 
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# 预先计算判别器输出尺寸
with torch.no_grad():
    d_out_shape = D_A(torch.randn(1, 3, 256, 256).to(device)).shape[-2:]

# 创建平滑和非平滑标签
def create_labels(batch_size, d_out_shape, device, smooth=False):
    if smooth:
        # 使用标签平滑
        valid = torch.ones((batch_size, 1, *d_out_shape)).to(device) * (1.0 - config['label_smoothing'])
        fake = torch.zeros((batch_size, 1, *d_out_shape)).to(device) + config['label_smoothing'] * 0.1
    else:
        valid = torch.ones((batch_size, 1, *d_out_shape)).to(device)
        fake = torch.zeros((batch_size, 1, *d_out_shape)).to(device)
    return valid, fake

# 从检查点恢复训练
start_epoch = 0
if config['resume_training']:
    latest_checkpoint = find_latest_checkpoint()
    if latest_checkpoint:
        start_epoch = load_checkpoint(latest_checkpoint) + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")

# 修改学习率调度器
def adjust_learning_rate(optimizer, epoch, is_discriminator=False):
    """改进的学习率调整策略"""
    base_lr = config['lr_D'] if is_discriminator else config['lr']
    min_lr = config['min_lr_D'] if is_discriminator else config['min_lr']
    
    if epoch < config['warmup_epochs']:
        # Warmup阶段
        lr = base_lr * (epoch + 1) / config['warmup_epochs']
    elif epoch >= config['lr_decay_start_epoch']:
        # 使用更平滑的余弦衰减
        decay_progress = (epoch - config['lr_decay_start_epoch']) / config['lr_decay_epochs']
        decay_factor = 0.5 * (1 + np.cos(np.pi * min(1.0, decay_progress)))
        lr = min_lr + (base_lr - min_lr) * decay_factor
    else:
        # 正常训练阶段
        lr = base_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# 训练循环
print(f"Starting training on {device}...")
for epoch in range(start_epoch, config['epochs']):
    # 分别调整生成器和判别器的学习率
    current_lr_G = adjust_learning_rate(optimizer_G, epoch, is_discriminator=False)
    current_lr_D = adjust_learning_rate(optimizer_D, epoch, is_discriminator=True)
    wandb.log({
        'learning_rate_G': current_lr_G,
        'learning_rate_D': current_lr_D
    })
    
    loop = tqdm(dataloader, desc=f"Epoch [{epoch}/{config['epochs']}]")
    
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    epoch_loss_cycle = 0.0
    epoch_loss_identity = 0.0
    epoch_loss_GAN = 0.0
    
    for i, batch in enumerate(loop):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        
        # 根据当前批次的实际大小创建标签
        curr_batch_size = real_A.size(0)
        valid, fake = create_labels(curr_batch_size, d_out_shape, device, smooth=True)

        # 训练生成器
        optimizer_G.zero_grad()

        with torch.amp.autocast('cuda'):
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)  # 使用平滑标签
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)  # 使用平滑标签
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # 总生成器损失
            loss_G = (
                loss_GAN + 
                config['lambda_cycle'] * loss_cycle + 
                config['lambda_identity'] * loss_identity
            ) / config['gradient_accumulation_steps']

        # 使用scaler进行反向传播
        scaler.scale(loss_G).backward()
        
        if (i + 1) % config['gradient_accumulation_steps'] == 0:
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()

        # 训练判别器
        optimizer_D.zero_grad()

        with torch.amp.autocast('cuda'):
            # 判别器A的损失
            loss_real_A = criterion_GAN(D_A(real_A), valid)
            loss_fake_A = criterion_GAN(D_A(fake_A.detach()), fake)
            loss_D_A = (loss_real_A + loss_fake_A) / 2

            # 判别器B的损失
            loss_real_B = criterion_GAN(D_B(real_B), valid)
            loss_fake_B = criterion_GAN(D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real_B + loss_fake_B) / 2

            loss_D = (loss_D_A + loss_D_B) / (2 * config['gradient_accumulation_steps'])

        scaler.scale(loss_D).backward()
        
        if (i + 1) % config['gradient_accumulation_steps'] == 0:
            scaler.step(optimizer_D)
            scaler.update()
            optimizer_D.zero_grad()

        # 记录损失
        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        epoch_loss_cycle += loss_cycle.item()
        epoch_loss_identity += loss_identity.item()
        epoch_loss_GAN += loss_GAN.item()
        
        # 更新进度条信息
        loop.set_postfix({
            'G_loss': f"{loss_G.item():.4f}",
            'D_loss': f"{loss_D.item():.4f}",
            'cycle_loss': f"{loss_cycle.item():.4f}"
        })

        # 记录到wandb
        if i % config['log_freq'] == 0:
            wandb.log({
                'loss_G': loss_G.item(),
                'loss_D': loss_D.item(),
                'loss_cycle': loss_cycle.item(),
                'loss_identity': loss_identity.item(),
                'loss_GAN': loss_GAN.item(),
                'generated_images': [
                    wandb.Image(fake_B[0], caption="Generated Monet"),
                    wandb.Image(fake_A[0], caption="Generated Photo")
                ]
            })
    
    # 计算并打印每个epoch的平均损失
    num_batches = len(dataloader)
    avg_losses = {
        'Generator': epoch_loss_G/num_batches,
        'Discriminator': epoch_loss_D/num_batches,
        'Cycle': epoch_loss_cycle/num_batches,
        'Identity': epoch_loss_identity/num_batches,
        'GAN': epoch_loss_GAN/num_batches
    }
    
    print(f"\nEpoch {epoch} Average Losses:")
    for loss_name, loss_value in avg_losses.items():
        print(f"{loss_name}: {loss_value:.4f}")
    print()
    
    # 保存检查点
    if (epoch + 1) % config['checkpoint_freq'] == 0:
        save_checkpoint(epoch, avg_losses)

    # 计算FID分数
    if epoch % config['fid_freq'] == 0:
        print("Calculating FID scores...")
        # 使用测试集计算FID
        test_dataset = ImageDataset("data", mode='test')
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        fid_scores = calculate_fid(G_AB, G_BA, test_dataloader, device)
        wandb.log({
            'FID_score_photo': fid_scores['photo_fid'],
            'FID_score_monet': fid_scores['monet_fid'],
            'FID_score_avg': (fid_scores['photo_fid'] + fid_scores['monet_fid']) / 2
        })
        print(f"Photo Domain FID Score: {fid_scores['photo_fid']:.4f}")
        print(f"Monet Domain FID Score: {fid_scores['monet_fid']:.4f}")
        print(f"Average FID Score: {(fid_scores['photo_fid'] + fid_scores['monet_fid']) / 2:.4f}")

# 保存最终模型
save_checkpoint(config['epochs']-1, avg_losses, is_final=True)

print("Training completed!") 