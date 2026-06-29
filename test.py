import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import time
import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    from skimage.metrics import structural_similarity as compute_ssim
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("[Avviso] Modulo 'skimage' non trovato. Impossibile calcolare PSNR e SSIM. Esegui: pip install scikit-image")

from model import HFMResidualNet

def run_inference(image_path, model_path, output_path, gt_path=None, scale_factor=2):
    """
    Esegue la pipeline di inferenza end-to-end e calcola le metriche di validazione:
    1. Scomposizione YCbCr
    2. Upscaling bicubico hardware per Y (Basse Frequenze), Cb e Cr
    3. Predizione neurale per Y (Alte Frequenze / Residuo)
    4. Somma algebrica del canale Y e ricostruzione RGB
    5. Calcolo comparativo PSNR e SSIM (Pipeline vs Baseline Bicubica)
    """
    # 1. Inizializzazione Hardware e Modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sistema] Esecuzione su dispositivo: {device}")
    
    # Inizializzazione della rete con i parametri architetturali standard
    model = HFMResidualNet(num_features=64, num_blocks=4, scale_factor=scale_factor).to(device)
    
    # Caricamento dei pesi addestrati
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"[Sistema] Modello caricato con successo da: {model_path}")
    else:
        print(f"[Avviso] Checkpoint '{model_path}' non trovato. Esecuzione con pesi casuali (solo per test di latenza).")
    
    # Abilitazione FP16 per i pesi del modello se in esecuzione su CUDA (Tensor Cores)
    if device.type == 'cuda':
        model = model.half()
        print("[Sistema] Modello convertito in FP16 (Half-Precision).")

    model.eval()

    # 2. Pre-processing Immagine (CPU)
    print(f"[Dati] Lettura immagine: {image_path}")
    img_lr = Image.open(image_path).convert('YCbCr')
    lr_y, lr_cb, lr_cr = img_lr.split()

    # Conversione in tensori e trasferimento su GPU [Shape: 1, 1, H, W]
    y_tensor = TF.to_tensor(lr_y).unsqueeze(0).to(device)
    cb_tensor = TF.to_tensor(lr_cb).unsqueeze(0).to(device)
    cr_tensor = TF.to_tensor(lr_cr).unsqueeze(0).to(device)

    # Conversione in FP16 dei tensori di input per computazione accelerata
    if device.type == 'cuda':
        y_tensor = y_tensor.half()
        cb_tensor = cb_tensor.half()
        cr_tensor = cr_tensor.half()

    target_size = (y_tensor.shape[2] * scale_factor, y_tensor.shape[3] * scale_factor)

    # 3. Warm-up GPU
    if device.type == 'cuda':
        print("[GPU] Fase di warm-up in corso...")
        with torch.no_grad():
            for _ in range(10):
                _ = model(y_tensor)
        torch.cuda.synchronize()

    # 4. Profilazione ed Esecuzione della Pipeline
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print("[Calcolo] Inizio inferenza pipeline (FP16)...")
    with torch.no_grad():
        start_event.record()

        # Path 1: Upscaling Hardware Basse Frequenze (Y_base)
        y_base = F.interpolate(y_tensor, size=target_size, mode='bicubic', align_corners=False)
        
        # Path 2: Estrazione Feature e Predizione Alte Frequenze (Y_hf)
        y_hf = model(y_tensor)
        
        # Merge Algebrico
        y_hr = torch.clamp(y_base + y_hf, 0.0, 1.0)
        
        # Path 3: Upscaling Hardware Cromaticità (Cr, Cb)
        cb_hr = F.interpolate(cb_tensor, size=target_size, mode='bicubic', align_corners=False)
        cr_hr = F.interpolate(cr_tensor, size=target_size, mode='bicubic', align_corners=False)

        end_event.record()
    
    # Sincronizzazione per misurazione esatta
    if device.type == 'cuda':
        torch.cuda.synchronize()
        latency_ms = start_event.elapsed_time(end_event)
    else:
        latency_ms = 0.0

    print(f"[Risultato] Latenza Inferenza GPU (Pipeline Completa): {latency_ms:.3f} ms")
    if latency_ms > 0:
        print(f"[Risultato] Framerate Equivalente (FPS): {1000 / latency_ms:.1f} FPS")

    # 5. Post-processing e Salvataggio (CPU)
    # Upcasting obbligatorio a FP32 (.float()) per la compatibilità con le librerie PIL/Numpy
    y_hr_pil = TF.to_pil_image(y_hr.squeeze(0).cpu().float())
    y_base_pil = TF.to_pil_image(y_base.squeeze(0).cpu().float()) # Conservato per confronto baseline
    cb_hr_pil = TF.to_pil_image(cb_hr.squeeze(0).cpu().float())
    cr_hr_pil = TF.to_pil_image(cr_hr.squeeze(0).cpu().float())

    # Ricostruzione Pipeline (CNN + Bicubica)
    img_hr_ycbcr = Image.merge('YCbCr', (y_hr_pil, cb_hr_pil, cr_hr_pil))
    img_hr_rgb = img_hr_ycbcr.convert('RGB')
    
    # Ricostruzione Baseline (Solo Bicubica)
    img_bicubic_ycbcr = Image.merge('YCbCr', (y_base_pil, cb_hr_pil, cr_hr_pil))
    img_bicubic_rgb = img_bicubic_ycbcr.convert('RGB')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_hr_rgb.save(output_path)
    print(f"[Dati] Immagine ricostruita salvata in: {output_path}")

    # 6. Calcolo Metriche Quantitative (PSNR / SSIM)
    if gt_path and os.path.exists(gt_path) and METRICS_AVAILABLE:
        print(f"[Metriche] Elaborazione misurazioni contro la GT: {gt_path}")
        img_gt = Image.open(gt_path).convert('RGB')
        
        # Allineamento dimensionale di sicurezza
        if img_gt.size != img_hr_rgb.size:
            print(f"[Avviso] Dimensioni discordanti: GT {img_gt.size} vs Output {img_hr_rgb.size}. Esecuzione crop algebrico per validazione.")
            img_gt = img_gt.crop((0, 0, img_hr_rgb.size[0], img_hr_rgb.size[1]))

        # Conversione in matrici Numpy
        arr_pred = np.array(img_hr_rgb)
        arr_bicubic = np.array(img_bicubic_rgb)
        arr_gt = np.array(img_gt)

        # Calcolo rigoroso su asse RGB (channel_axis=2)
        psnr_pipe = compute_psnr(arr_gt, arr_pred)
        ssim_pipe = compute_ssim(arr_gt, arr_pred, channel_axis=2)
        
        psnr_bic = compute_psnr(arr_gt, arr_bicubic)
        ssim_bic = compute_ssim(arr_gt, arr_bicubic, channel_axis=2)

        print("-" * 55)
        print(" CONFRONTO METRICHE (vs Ground Truth)")
        print("-" * 55)
        print(f" [Baseline Bicubica] PSNR : {psnr_bic:.3f} dB  |  SSIM : {ssim_bic:.4f}")
        print(f" [Pipeline Residual] PSNR : {psnr_pipe:.3f} dB  |  SSIM : {ssim_pipe:.4f}")
        print(f" [Delta (Guadagno)]  PSNR : {psnr_pipe - psnr_bic:+.3f} dB  |  SSIM : {ssim_pipe - ssim_bic:+.4f}")
        print("-" * 55)
    elif gt_path and not os.path.exists(gt_path):
        print(f"[Avviso] Ground Truth '{gt_path}' non trovata. Calcolo metriche ignorato.")

if __name__ == "__main__":
    # Parametri di esecuzione del test
    imgs = ["0801", "0802", "0803", "0804", "0805", "0806", "0807", "0808", "0809", "0810"]
    #INPUT_IMAGE = f"../dataset/DIV2K_valid_LR_bicubic/X2/{img}x2.png"         
    #GT_IMAGE = f"../dataset/DIV2K_valid_HR/{img}.png"          # Aggiunto parametro per la Ground Truth
    MODEL_CHECKPOINT = "checkpoints/residual_best.pth"
    OUTPUT_IMAGE = "results/test_image_hr.png"
    SCALE = 2
    
    #if not os.path.exists(INPUT_IMAGE):
    #    raise FileNotFoundError(f"[Errore Critico] L'immagine di input '{INPUT_IMAGE}' non è stata trovata.")

    # Esecuzione del test passando il path della GT
    for img in imgs:
        INPUT_IMAGE = f"../dataset/DIV2K_valid_LR_bicubic/X2/{img}x2.png"         
        GT_IMAGE = f"../dataset/DIV2K_valid_HR/{img}.png" 
        run_inference(INPUT_IMAGE, MODEL_CHECKPOINT, OUTPUT_IMAGE, gt_path=GT_IMAGE, scale_factor=SCALE)