import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import HFMResidualNet
from dataset import SuperResolutionDataset

# =====================================================================
# 3. TRAINING PIPELINE
# =====================================================================
def train_model():
    UPSCALE_FACTOR = 2
    LR_CROP_SIZE = 256
    BATCH_SIZE = 16
    EPOCHS = 120
    LEARNING_RATE = 2e-4
    
    HR_TRAIN_DIR = "../dataset/DIV2K_train_HR"
    LR_TRAIN_DIR = "../dataset/DIV2K_train_LR_bicubic/X2"
    HR_VAL_DIR = "../dataset/DIV2K_valid_HR"
    LR_VAL_DIR = "../dataset/DIV2K_valid_LR_bicubic/X2"
    
    CHECKPOINT_LAST = "checkpoints/residual_last.pth"
    CHECKPOINT_BEST = "checkpoints/residual_best.pth"
    
    os.makedirs("checkpoints", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Assicurati che la firma della tua classe Dataset corrisponda a questi parametri
    train_dataset = SuperResolutionDataset(LR_TRAIN_DIR, HR_TRAIN_DIR, LR_CROP_SIZE, UPSCALE_FACTOR)
    val_dataset = SuperResolutionDataset(LR_VAL_DIR, HR_VAL_DIR, LR_CROP_SIZE, UPSCALE_FACTOR)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, pin_memory=True)

    # Inizializzazione Modello, Loss (L1 su HF) e Ottimizzatore
    model = HFMResidualNet(num_features=64, num_blocks=4, scale_factor=UPSCALE_FACTOR).to(device)
    criterion = nn.L1Loss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

    start_epoch = 0
    best_val_loss = float('inf')
    
    if os.path.exists(CHECKPOINT_LAST):
        print(f"Ripristino del checkpoint '{CHECKPOINT_LAST}'...")
        checkpoint = torch.load(CHECKPOINT_LAST, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Ripresa dall'epoca {start_epoch}. Miglior Loss registrata: {best_val_loss:.6f}")

    for epoch in range(start_epoch, EPOCHS):
        # ------------------- FASE DI TRAINING -------------------
        model.train()
        train_loss = 0.0
        
        for lr_patch, hr_patch in train_loader:
            lr_patch = lr_patch.to(device)
            hr_patch = hr_patch.to(device)
            
            optimizer.zero_grad()
            
            # Calcolo target HF on-the-fly su GPU
            target_size = (hr_patch.shape[2], hr_patch.shape[3])
            x_bicubic = F.interpolate(lr_patch, size=target_size, mode='bicubic', align_corners=False)
            hf_target = hr_patch - x_bicubic
            
            # Forward pass (Rete riceve solo LR)
            hf_predicted = model(lr_patch)
            
            # Calcolo Loss nel dominio dei residui
            loss = criterion(hf_predicted, hf_target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * lr_patch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # ------------------- FASE DI VALIDAZIONE -------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_patch, hr_patch in val_loader:
                lr_patch = lr_patch.to(device)
                hr_patch = hr_patch.to(device)
                
                # Calcolo target HF on-the-fly su GPU (IDENTICO al training)
                target_size = (hr_patch.shape[2], hr_patch.shape[3])
                x_bicubic = F.interpolate(lr_patch, size=target_size, mode='bicubic', align_corners=False)
                hf_target = hr_patch - x_bicubic
                
                # Forward pass corretto (Solo 1 argomento)
                hf_predicted = model(lr_patch)
                
                # Loss calcolata sui tensori HF
                loss = criterion(hf_predicted, hf_target)
                val_loss += loss.item() * lr_patch.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoca [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # ------------------- CHECKPOINTING -------------------
        state_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        
        torch.save(state_dict, CHECKPOINT_LAST)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict['best_val_loss'] = best_val_loss
            torch.save(state_dict, CHECKPOINT_BEST)
            print(f"--> Nuovo miglior modello salvato! (Val Loss: {best_val_loss:.6f})")

if __name__ == "__main__":
    train_model()