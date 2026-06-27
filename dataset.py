import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class SuperResolutionDataset(Dataset):
    """
    Dataset PyTorch per Super-Resolution.
    Carica le immagini, isola il canale Y (Luminanza) ed estrae patch (crop) 
    sincronizzate tra LR e HR per permettere l'addestramento in batch.
    """
    def __init__(self, lr_dir, gt_dir, lr_crop_size=256, scale_factor=2):
        super().__init__()
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.lr_crop_size = lr_crop_size
        self.scale_factor = scale_factor
        
        # Filtro analitico per escludere file di sistema o directory nascoste
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
        self.lr_images = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(valid_ext)])
        self.gt_images = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(valid_ext)])
        
        # Aggiunta dei conteggi esatti nel messaggio di errore per eventuale debugging futuro
        assert len(self.lr_images) == len(self.gt_images), f"Disallineamento numerico: trovate {len(self.lr_images)} immagini LR e {len(self.gt_images)} immagini HR."

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        # 1. Caricamento file
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])

        lr_img = Image.open(lr_path).convert('YCbCr')
        gt_img = Image.open(gt_path).convert('YCbCr')

        # 2. Isolamento del canale Y
        lr_y, _, _ = lr_img.split()
        gt_y, _, _ = gt_img.split()

        # 3. Logica di Random Crop sincronizzato
        w_lr, h_lr = lr_y.size
        
        # Verifica di sicurezza strutturale
        if w_lr < self.lr_crop_size or h_lr < self.lr_crop_size:
            raise ValueError(f"Immagine {lr_path} più piccola della patch richiesta ({self.lr_crop_size})")

        # Generazione coordinate per l'immagine LR
        top_lr = random.randint(0, h_lr - self.lr_crop_size)
        left_lr = random.randint(0, w_lr - self.lr_crop_size)
        
        # Proiezione algebrica delle coordinate per l'immagine HR
        top_hr = top_lr * self.scale_factor
        left_hr = left_lr * self.scale_factor
        hr_crop_size = self.lr_crop_size * self.scale_factor

        # Estrazione effettiva delle patch
        lr_patch = TF.crop(lr_y, top_lr, left_lr, self.lr_crop_size, self.lr_crop_size)
        hr_patch = TF.crop(gt_y, top_hr, left_hr, hr_crop_size, hr_crop_size)

        # 4. Conversione in Tensori PyTorch (valori normalizzati in [0.0, 1.0])
        lr_tensor = TF.to_tensor(lr_patch)
        hr_tensor = TF.to_tensor(hr_patch)

        # Restituisce le patch grezze. Il calcolo del residuo HF e della bicubica 
        # viene demandato alla GPU nel loop di addestramento (train.py).
        return lr_tensor, hr_tensor