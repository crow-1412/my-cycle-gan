import torch
import wandb
from torch.utils.data import DataLoader
from models.generator import Generator
from utils.dataset import ImageDataset
from utils.fid_score import calculate_fid
from pathlib import Path
import glob

def find_checkpoints():
    """查找所有检查点文件"""
    checkpoint_dir = Path('checkpoints')
    checkpoint_files = sorted(glob.glob(str(checkpoint_dir / "checkpoint_epoch_*.pth")))
    return checkpoint_files

def load_model_from_checkpoint(checkpoint_path, device):
    """从检查点加载模型"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    
    G_AB.load_state_dict(checkpoint['G_AB_state_dict'])
    G_BA.load_state_dict(checkpoint['G_BA_state_dict'])
    
    epoch = checkpoint['epoch']
    return G_AB, G_BA, epoch

def main():
    # 初始化wandb - 使用新的run名称
    wandb.init(project="cycle-gan-monet", name="fid_recalculation")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据加载器配置 - 使用测试集
    dataset = ImageDataset("data", mode='test')  # 修改为使用测试集
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # 减小batch_size
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 获取所有检查点
    checkpoint_files = find_checkpoints()
    
    # 对每个检查点计算FID
    for checkpoint_path in checkpoint_files:
        try:
            # 加载模型
            G_AB, G_BA, epoch = load_model_from_checkpoint(checkpoint_path, device)
            G_AB.eval()
            G_BA.eval()
            
            # 计算FID分数
            print(f"\nCalculating FID scores for epoch {epoch}...")
            fid_scores = calculate_fid(G_AB, G_BA, dataloader, device)
            
            # 记录到wandb
            wandb.log({
                'epoch': epoch,
                'FID_score_photo': fid_scores['photo_fid'],
                'FID_score_monet': fid_scores['monet_fid'],
                'FID_score_avg': (fid_scores['photo_fid'] + fid_scores['monet_fid']) / 2
            })
            
            print(f"Epoch {epoch}:")
            print(f"Photo Domain FID Score: {fid_scores['photo_fid']:.4f}")
            print(f"Monet Domain FID Score: {fid_scores['monet_fid']:.4f}")
            print(f"Average FID Score: {(fid_scores['photo_fid'] + fid_scores['monet_fid']) / 2:.4f}")
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 