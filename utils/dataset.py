from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transforms.Compose(
            [
                transforms.Resize(286),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ) if transform is None else transform

        self.files_A = sorted(os.listdir(os.path.join(root, f'{mode}A')))
        self.files_B = sorted(os.listdir(os.path.join(root, f'{mode}B')))
        self.root = root
        self.mode = mode

    def __getitem__(self, index):
        A_path = os.path.join(self.root, f'{self.mode}A', self.files_A[index % len(self.files_A)])
        B_path = os.path.join(self.root, f'{self.mode}B', self.files_B[index % len(self.files_B)])
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)

        return {'A': A, 'B': B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B)) 