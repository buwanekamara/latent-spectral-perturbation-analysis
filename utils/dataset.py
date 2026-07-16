# import os
# import torch
# from PIL import Image
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms

# class DeepfakeDataset(Dataset):
#     def __init__(self, real_dir, fake_dir, is_train=True):
#         self.image_paths = []
#         self.labels = []
        
#         # Label 1.0 = Real, 0.0 = Fake
#         for img_name in os.listdir(real_dir):
#             if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 self.image_paths.append(os.path.join(real_dir, img_name))
#                 self.labels.append(1.0)
            
#         for img_name in os.listdir(fake_dir):
#             if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 self.image_paths.append(os.path.join(fake_dir, img_name))
#                 self.labels.append(0.0)

#         # Base ViT normalization values for CLIP
#         mean = (0.48145466, 0.4578275, 0.40821073)
#         std = (0.26862954, 0.26130258, 0.27577711)

#         if is_train:
#             self.transform = transforms.Compose([
#                 transforms.Resize((256, 256)),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.ColorJitter(brightness=0.1, contrast=0.1),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std)
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Resize((256, 256)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std)
#             ])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         image = self.transform(image)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)
#         return image, label

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile

# 1. Allow massive files to load into memory without triggering DOS protection
Image.MAX_IMAGE_PIXELS = 300_000_000 

# 2. Tell Pillow to ignore corrupted/partially downloaded image files and load whatever it can
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, is_train=True):
        self.file_paths = []
        self.labels = []
        self.is_train = is_train
        
        # Load real images (Label = 1.0)
        for fname in os.listdir(real_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                self.file_paths.append(os.path.join(real_dir, fname))
                self.labels.append(1.0)
                
        # Load fake images (Label = 0.0)
        for fname in os.listdir(fake_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                self.file_paths.append(os.path.join(fake_dir, fname))
                self.labels.append(0.0)

        # 3. The Forensically Safe Transform Pipeline
        if is_train:
            self.transform = transforms.Compose([
                # DO NOT use transforms.Resize() on massive images if you can avoid it.
                # Instead, grab a random chunk of the original high-res pixels:
                transforms.RandomCrop(224, pad_if_needed=True, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                # For validation, grab the exact center of the high-res image
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Pillow will now safely attempt to load this, even if corrupted
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)