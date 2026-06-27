import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Blocco residuo standard per l'estrazione profonda delle feature.
    L'utilizzo di bias=False nelle convoluzioni intermedie riduce i parametri 
    e ottimizza il calcolo sui Tensor Cores.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + residual

class HFMResidualNet(nn.Module):
    """
    High-Frequency Mapping Residual Network (HFM-ResNet).
    Mappa un input spaziale LR a 1 canale verso un residuo HF spaziale ad alta risoluzione.
    """
    def __init__(self, num_features=64, num_blocks=4, scale_factor=2):
        """
        Args:
            num_features (int): Numero di canali interni per l'estrazione delle feature.
            num_blocks (int): Numero di blocchi residui nel corpo della rete.
            scale_factor (int): Fattore di ingrandimento (default: 2 per 720p -> 1440p).
        """
        super().__init__()
        self.scale_factor = scale_factor
        
        # 1. Estrazione (Head): Proiezione dell'input LR (1 canale Y) nello spazio delle feature
        self.head = nn.Conv2d(1, num_features, kernel_size=3, padding=1)
        
        # 2. Mappatura Non-Lineare (Body): Sequenza parametrizzabile di blocchi residui
        self.body = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        
        # Convoluzione di transizione post-corpo
        self.body_tail = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # 3. Preparazione Upsampling (Tail): Compressione delle feature per il PixelShuffle
        # Per un output a 1 canale con scale_factor 2, servono 1 * 2^2 = 4 canali
        out_channels_before_shuffle = 1 * (scale_factor ** 2)
        self.tail = nn.Conv2d(num_features, out_channels_before_shuffle, kernel_size=3, padding=1)
        
        # 4. Upsampling Spaziale: Riorganizza i canali in espansione spaziale
        self.upsample = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        # Estrazione iniziale
        x_head = self.head(x)
        
        # Passaggio attraverso i blocchi residui
        res = self.body(x_head)
        res = self.body_tail(res)
        
        # Connessione residua globale: stabilizza il gradiente per reti più profonde
        x_features = x_head + res
        
        # Proiezione e rimescolamento dei pixel (PixelShuffle)
        x_tail = self.tail(x_features)
        hf_out = self.upsample(x_tail)
        
        # L'output NON ha funzioni di attivazione (es. Sigmoid o Tanh) 
        # perché le alte frequenze (Target HF) possiedono valori sia negativi (undershoot) che positivi.
        return hf_out
