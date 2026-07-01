import torch
import torch.nn.functional as F
import numpy as np

# Assicurarsi che il modulo model.py sia presente nella stessa directory
try:
    from model import HFMResidualNet
except ImportError:
    raise ImportError("Impossibile importare HFMResidualNet. Verificare che model.py sia accessibile.")

def run_dummy_benchmark(model, device, height, width, scale_factor=2, num_iterations=50, precision='FP32'):
    """
    Esegue il benchmark prestazionale su tensori dummy allocati in VRAM,
    sfruttando stream CUDA asincroni per parallelizzare le computazioni.
    """
    target_size = (height * scale_factor, width * scale_factor)
    dtype = torch.float16 if precision == 'FP16' else torch.float32
    
    # 1. Allocazione tensori dummy direttamente in VRAM
    y_tensor = torch.rand(1, 1, height, width, device=device, dtype=dtype)
    cb_tensor = torch.rand(1, 1, height, width, device=device, dtype=dtype)
    cr_tensor = torch.rand(1, 1, height, width, device=device, dtype=dtype)

    # 2. Inizializzazione Stream CUDA
    lf_path = torch.cuda.Stream()
    hf_path = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream()

    # 3. Fase di Warm-up (Stabilizzazione frequenze di clock)
    with torch.no_grad():
        for _ in range(10):
            _ = model(y_tensor)
            _ = F.interpolate(cb_tensor, size=target_size, mode='bicubic', align_corners=False)
    torch.cuda.synchronize()

    # 4. Profilazione Temporale Iterativa
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        
        for _ in range(num_iterations):
            # PATH 1: Interpolazioni Analitiche
            with torch.cuda.stream(lf_path):
                y_base = F.interpolate(y_tensor, size=target_size, mode='bicubic', align_corners=False)
                cb_hr = F.interpolate(cb_tensor, size=target_size, mode='bicubic', align_corners=False)
                cr_hr = F.interpolate(cr_tensor, size=target_size, mode='bicubic', align_corners=False)
            
            # PATH 2: Inferenza Neurale
            with torch.cuda.stream(hf_path):
                y_hf = model(y_tensor)
            
            # SINCRONIZZAZIONE
            default_stream.wait_stream(lf_path)
            default_stream.wait_stream(hf_path)
            
            # MERGE
            y_hr = torch.clamp(y_base + y_hf, 0.0, 1.0)
            
        end_event.record()
        torch.cuda.synchronize()

    # Calcolo medie
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / num_iterations
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return avg_latency_ms, fps

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        print("[Avviso] Benchmark interrotto. Dispositivo CUDA non rilevato. I test di latenza richiedono l'accelerazione hardware GPU.")
        exit(1)

    print(f"[Sistema] Esecuzione su dispositivo: {torch.cuda.get_device_name(0)}")
    
    scale = 4
    precisions = ['FP32', 'FP16']

    # Definizione Risoluzioni Native 16:9
    resolutions = [
        {"name": "480p", "width": 854, "height": 480},
        {"name": "720p", "width": 1280, "height": 720},
        {"name": "1080p", "width": 1920, "height": 1080}
    ]

    for prec in precisions:
        print("\n" + "="*60)
        print(f" BENCHMARK LATENZA INFERENZA (GPU {prec}) - SCALA X2")
        print("="*60)

        # Inizializzazione Rete HFM-ResNet
        model = HFMResidualNet(num_features=64, num_blocks=8, scale_factor=scale).to(device)
        
        if prec == 'FP16':
            model = model.half() # Forzatura Tensor Cores
            print("[Sistema] Modello inizializzato in FP16 (Half-Precision).")
        else:
            model = model.float() # Precisione singola standard
            print("[Sistema] Modello inizializzato in FP32 (Single-Precision).")
            
        model.eval()

        for res in resolutions:
            name = res["name"]
            w = res["width"]
            h = res["height"]
            
            print(f"\n[Test] Esecuzione risoluzione nativa: {name} ({w}x{h}) -> Upscaling a {w*scale}x{h*scale}")
            
            latency, fps = run_dummy_benchmark(model, device, height=h, width=w, scale_factor=scale, precision=prec)
            
            print("-" * 50)
            print(f" Latenza Media (su 50 iterazioni) : {latency:.3f} ms")
            print(f" Throughput Equivalente           : {fps:.1f} FPS")
            print("-" * 50)