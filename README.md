# Documentazione Architetturale: HFM-ResNet

Il modello HFM-ResNet opera una separazione logica e spaziale del segnale. Di seguito la descrizione tecnica dei blocchi computazionali.

## 1. Pre-Processing e Separazione Spaziale (Dominio YCbCr)

- **Funzione**: Il tensore RGB a bassa risoluzione (LR) viene convertito nello spazio colore YCbCr.
- **Dinamica**: I canali di crominanza (Cb, Cr) vengono sovracampionati tramite interpolazione bicubica hardware, escludendoli dall'elaborazione neurale.
- **Input Rete**: La rete riceve esclusivamente il tensore della luminanza (Y) con dimensioni spaziali `[1, H, W]`.

## 2. Head (Estrazione Iniziale)

- **Composizione**: Convoluzione spaziale $3\times3$ con stride 1 e padding 1.
- **Funzione**: Mappa l'input in uno spazio latente.
- **Output**: Tensore di shallow features con dimensioni `[64, H, W]`.

## 3. Body (Mappatura Non-Lineare Profonda)

- **Composizione**: Cascata di 4 Blocchi Residui. Struttura del singolo blocco:
  `Conv2d (3x3) -> LeakyReLU -> Conv2d (3x3) -> Somma Algebrica`.
- **Vincoli Matematici**: Le convoluzioni operano senza bias (`bias=False`).
- **Funzione**: Estrae le caratteristiche ad alta frequenza mantenendo la dimensionalità fissa a 64 canali.
- **Output**: Tensore profondo con dimensioni `[64, H, W]`.

## 4. Body Tail e Global Skip Connection

- **Composizione**: Convoluzione Body Tail $3\times3$ seguita da un nodo di somma algebrica.
- **Dinamica**: Il tensore di output dell'Head viene sommato all'output del Body Tail (Global Skip Connection).
- **Funzione**: Forza la rete ad apprendere unicamente il residuo ad alta frequenza.

## 5. Tail (Espansione Latente) e Pixel Shuffle (Upsampling)

- **Tail**: Convoluzione $3\times3$ che proietta le 64 feature map in un output di dimensione `[r², H, W]` (dove $r$ è il fattore di scala).
- **Pixel Shuffle**: Operatore deterministico che converte la profondità in risoluzione spaziale.
  - **Meccanica**: Riorganizza i pixel impilati lungo l'asse dei canali in blocchi spaziali bidimensionali.
  - **Funzione**: Esegue l'upsampling spaziale. Il tensore passa dalla dimensione `[r², H, W]` alla dimensione finale `[1, r*H, r*W]`.

## 6. Merge Algebrico e Post-Processing

- **Dinamica**: Il residuo HF (output neurale) viene sommato al canale Y upscalato analiticamente in bicubica (LF).
- **Clamp**: Il tensore risultante viene vincolato nel dominio continuo `[0.0, 1.0]`.
- **Output Finale**: Il canale di luminanza HR viene concatenato con i canali cromatici HR (Cb, Cr). Il tensore finale viene riconvertito nello spazio RGB nativo.
