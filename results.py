import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Tentativo di importazione per le metriche rigorose
try:
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    from skimage.metrics import structural_similarity as compute_ssim
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    raise ImportError("[Errore Critico] Il modulo 'skimage' è obbligatorio per questo script. Esegui: pip install scikit-image")

from model import HFMResidualNet

def evaluate_and_plot(lr_paths_dict, gt_paths_dict, model_path, output_dir="results", scale_factor=2, max_images=20):
    """
    Esegue la valutazione comparativa (Pipeline Neurale vs Bicubica) su un set di immagini.
    Salva un plot comparativo per ciascuna iterazione.

    Args:
        lr_paths_dict (dict): Dizionario {id_immagine: path_lr}.
        gt_paths_dict (dict): Dizionario {id_immagine: path_gt}.
        model_path (str): Percorso al checkpoint del modello.
        output_dir (str): Cartella di destinazione per i grafici salvati.
        scale_factor (int): Fattore di ingrandimento spaziale.
        max_images (int): Numero massimo di immagini da processare.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Inizializzazione Hardware e Modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sistema] Inizializzazione valutazione batch su dispositivo: {device}")
    
    model = HFMResidualNet(num_features=96, num_blocks=2, scale_factor=scale_factor).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[Sistema] Pesi caricati: {model_path}")
    else:
        print(f"[Avviso] Checkpoint non trovato: {model_path}. Utilizzo pesi casuali.")
    
    if device.type == 'cuda':
        model = model.half()
    
    model.eval()

    # 2. Ciclo di Valutazione sulle immagini
    processed_count = 0
    
    # Intersezione delle chiavi per garantire coerenza tra LR e GT
    common_keys = list(set(lr_paths_dict.keys()).intersection(set(gt_paths_dict.keys())))
    
    for img_id in common_keys:
        if processed_count >= max_images:
            break
            
        lr_path = lr_paths_dict[img_id]
        gt_path = gt_paths_dict[img_id]
        
        if not os.path.exists(lr_path) or not os.path.exists(gt_path):
            print(f"[Avviso] File mancanti per l'ID {img_id}. Skip.")
            continue
            
        print(f"[Elaborazione] Immagine {processed_count+1}/{max_images} - ID: {img_id}")

        # -- Pre-processing LR --
        img_lr = Image.open(lr_path).convert('YCbCr')
        lr_y, lr_cb, lr_cr = img_lr.split()

        y_tensor = TF.to_tensor(lr_y).unsqueeze(0).to(device)
        cb_tensor = TF.to_tensor(lr_cb).unsqueeze(0).to(device)
        cr_tensor = TF.to_tensor(lr_cr).unsqueeze(0).to(device)

        if device.type == 'cuda':
            y_tensor = y_tensor.half()
            cb_tensor = cb_tensor.half()
            cr_tensor = cr_tensor.half()

        target_size = (y_tensor.shape[2] * scale_factor, y_tensor.shape[3] * scale_factor)

        # -- Inferenza --
        with torch.no_grad():
            y_base = F.interpolate(y_tensor, size=target_size, mode='bicubic', align_corners=False)
            y_hf = model(y_tensor)
            
            # Il clamp è applicato qui per l'output della CNN
            y_hr = torch.clamp(y_base + y_hf, 0.0, 1.0)
            
            cb_hr = F.interpolate(cb_tensor, size=target_size, mode='bicubic', align_corners=False)
            cr_hr = F.interpolate(cr_tensor, size=target_size, mode='bicubic', align_corners=False)

        # -- Post-processing Pipeline Neurale (HFMCNN) --
        y_hr_pil = TF.to_pil_image(y_hr.squeeze(0).cpu().float())
        # Aggiunto clamp(0,1) preventivo sui canali colore
        cb_hr_pil = TF.to_pil_image(torch.clamp(cb_hr, 0.0, 1.0).squeeze(0).cpu().float())
        cr_hr_pil = TF.to_pil_image(torch.clamp(cr_hr, 0.0, 1.0).squeeze(0).cpu().float())
        
        img_hr_ycbcr = Image.merge('YCbCr', (y_hr_pil, cb_hr_pil, cr_hr_pil))
        img_hfmcnn = img_hr_ycbcr.convert('RGB')
        arr_hfmcnn = np.array(img_hfmcnn)

        # -- Post-processing Baseline Bicubica --
        # Aggiunto clamp(0,1) preventivo sulla Y di base per bloccare l'overflow di overshoot bicubico
        y_base_pil = TF.to_pil_image(torch.clamp(y_base, 0.0, 1.0).squeeze(0).cpu().float())
        img_bicubic_ycbcr = Image.merge('YCbCr', (y_base_pil, cb_hr_pil, cr_hr_pil))
        img_bicubic = img_bicubic_ycbcr.convert('RGB')
        arr_bicubic = np.array(img_bicubic)

        # -- Preparazione GT e Calcolo Metriche --
        img_gt = Image.open(gt_path).convert('RGB')
        if img_gt.size != img_hfmcnn.size:
            img_gt = img_gt.crop((0, 0, img_hfmcnn.size[0], img_hfmcnn.size[1]))
        arr_gt = np.array(img_gt)

        # Calcolo PSNR / SSIM
        psnr_bic = compute_psnr(arr_gt, arr_bicubic)
        ssim_bic = compute_ssim(arr_gt, arr_bicubic, channel_axis=2)
        
        psnr_hfm = compute_psnr(arr_gt, arr_hfmcnn)
        ssim_hfm = compute_ssim(arr_gt, arr_hfmcnn, channel_axis=2)

        # -- Generazione Plot Comparativo con Zoom (Matplotlib) --
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Valutazione Upscaling - Immagine: {img_id}", fontsize=16)

        # Parametri e coordinate dello zoom centrale
        h, w = arr_gt.shape[:2]
        zoom_size = min(150, h, w)
        cx, cy = w // 2, h // 2
        x1, x2 = cx - zoom_size // 2, cx + zoom_size // 2
        y1, y2 = cy - zoom_size // 2, cy + zoom_size // 2

        # Organizzazione matrici e titoli per iterazione
        images = [arr_bicubic, arr_hfmcnn, arr_gt]
        titles = [
            f"Bicubic Baseline\nPSNR: {psnr_bic:.2f} dB | SSIM: {ssim_bic:.4f}",
            f"HFMCNN (Ours)\nPSNR: {psnr_hfm:.2f} dB | SSIM: {ssim_hfm:.4f}\n$\Delta$ PSNR: {psnr_hfm - psnr_bic:+.2f} dB | $\Delta$ SSIM: {ssim_hfm - ssim_bic:+.4f}",
            "Ground Truth (HR)\nTarget di riferimento"
        ]

        for i in range(3):
            # Visualizzazione immagine globale
            axes[i].imshow(images[i])
            axes[i].set_title(titles[i])
            axes[i].axis('off')

            # Definizione del sotto-asse per lo zoom (inset) posizionato in basso a destra (40% dell'area)
            axins = axes[i].inset_axes([0.55, 0.05, 0.4, 0.4])
            axins.imshow(images[i])
            axins.set_xlim(x1, x2)
            axins.set_ylim(y2, y1) # y capovolto per mantenere l'orientamento corretto
            
            # Rimozione label dagli assi di zoom per pulizia visiva
            axins.set_xticklabels([])
            axins.set_yticklabels([])
            
            # Rendering del box di riferimento sull'immagine globale
            axes[i].indicate_inset_zoom(axins, edgecolor="red", linewidth=2.0)

        plt.tight_layout()
        
        # Salvataggio
        save_path = os.path.join(output_dir, f"comparazione_{img_id}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) # Chiude la figura per evitare memory leak
        
        processed_count += 1

    print(f"\n[Completato] Generati e salvati {processed_count} plot comparativi in '{output_dir}'.")

if __name__ == "__main__":
    # Costruzione dizionari di test fittizi (sostituire con parsing della directory reale)
    # Esempio:
    lr_dir = "../dataset/DIV2K_valid_LR_bicubic/X2"
    gt_dir = "../dataset/DIV2K_valid_HR"
    
    lr_dict = {}
    lr_dict.update(
        {f"{i}": f"{lr_dir}/080{i}x2.png" for i in range(1, 10)}
    )
    lr_dict.update({f"{i}": f"{lr_dir}/08{i}x2.png" for i in range(10, 15)})

    gt_dict = {}
    gt_dict.update(
        {f"{i}": f"{gt_dir}/080{i}.png" for i in range(1, 10)}
    )
    gt_dict.update({f"{i}": f"{gt_dir}/08{i}.png" for i in range(10, 15)})
    
    evaluate_and_plot(
        lr_paths_dict=lr_dict,
        gt_paths_dict=gt_dict,
        model_path="checkpoints/2res_96f_best.pth",
        output_dir="results",
        scale_factor=2,
        max_images=16
    )