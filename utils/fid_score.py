import torch
import torch.nn as nn
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg
from tqdm import tqdm

def calculate_fid(G_AB, G_BA, dataloader, device):
    """
    计算两个方向的FID分数
    G_AB: 将A域(照片)转换为B域(Monet)的生成器
    G_BA: 将B域(Monet)转换为A域(照片)的生成器
    """
    # 加载预训练的Inception-v3模型
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # 移除最后的分类层
    inception_model.fc = nn.Identity()
    
    def get_features(images):
        with torch.no_grad():
            features = inception_model(images)
        return features.cpu().numpy()
    
    def calculate_statistics(features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_domain_fid(mu1, sigma1, mu2, sigma2):
        """计算两组统计量之间的FID分数"""
        diff = mu1 - mu2
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid_score = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid_score
    
    # 收集特征
    real_A_features = []
    real_B_features = []
    fake_A_features = []
    fake_B_features = []
    
    for batch in tqdm(dataloader, desc="Collecting features for FID calculation"):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        
        # 获取真实图像的特征
        real_A_features.append(get_features(real_A))
        real_B_features.append(get_features(real_B))
        
        # 生成假图像并获取特征
        with torch.no_grad():
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            fake_A_features.append(get_features(fake_A))
            fake_B_features.append(get_features(fake_B))
        
        # 清理GPU内存
        torch.cuda.empty_cache()
    
    # 连接特征
    real_A_features = np.concatenate(real_A_features, axis=0)
    real_B_features = np.concatenate(real_B_features, axis=0)
    fake_A_features = np.concatenate(fake_A_features, axis=0)
    fake_B_features = np.concatenate(fake_B_features, axis=0)
    
    # 计算统计量
    mu_real_A, sigma_real_A = calculate_statistics(real_A_features)
    mu_real_B, sigma_real_B = calculate_statistics(real_B_features)
    mu_fake_A, sigma_fake_A = calculate_statistics(fake_A_features)
    mu_fake_B, sigma_fake_B = calculate_statistics(fake_B_features)
    
    # 计算FID分数
    fid_photo = calculate_domain_fid(mu_real_A, sigma_real_A, mu_fake_A, sigma_fake_A)
    fid_monet = calculate_domain_fid(mu_real_B, sigma_real_B, mu_fake_B, sigma_fake_B)
    
    return {
        'photo_fid': float(fid_photo),  # 确保返回Python标量
        'monet_fid': float(fid_monet)
    } 