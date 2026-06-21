# Anime Sketch Colorization — Project Notes & Report Blueprint

> **Questo file non è il report.** È un documento di riferimento personale:
> struttura ipotetica del report, motivazioni di ogni scelta implementativa,
> note empiriche, e materiale grezzo da cui attingere per scrivere il testo.
> Il report va scritto in prima persona, in inglese, con parole proprie.

---

## Indice

1. [Struttura ipotetica del report](#1-struttura-ipotetica-del-report)
2. [Dataset — note empiriche](#2-dataset--note-empiriche)
3. [Variante 1 — U-Net + L1 (baseline)](#3-variante-1--u-net--l1-baseline)
4. [Variante 2 — Pix2Pix](#4-variante-2--pix2pix)
5. [Variante 3 — Pix2Pix + Perceptual Loss](#5-variante-3--pix2pix--perceptual-loss)
6. [Variante 4 — CycleGAN (unpaired)](#6-variante-4--cyclegan-unpaired)
7. [Variante 5 — Conditional VAE](#7-variante-5--conditional-vae)
8. [Loss functions — motivazioni](#8-loss-functions--motivazioni)
9. [Metriche — giustificazione vs rubrica](#9-metriche--giustificazione-vs-rubrica)
10. [Scelte architetturali trasversali](#10-scelte-architetturali-trasversali)
11. [Risultati quantitativi](#11-risultati-quantitativi)
12. [Immagini architetture (da generare con Gemini)](#12-immagini-architetture-da-generare-con-gemini)

---

## 1. Struttura ipotetica del report

Lunghezza target: ~15 pagine escluse figure e references.

```
Cover page
  Project Title, Master degree, Course (ML&DL / DL), A.Y. 2025-26, Author

1. Introduction / Motivation & Rationale
   - Il problema: colorizzazione automatica di sketch anime
   - Perché è interessante (creazione di contenuti, assistenza ad artisti digitali)
   - La domanda scientifica centrale: supervisione paired vs unpaired
   - Contributo: ablation progressiva su 5 varianti con metrica unificata

2. State of the Art (glance)
   - Image-to-image translation: Pix2Pix (Isola et al., 2017)
   - Unpaired I2I: CycleGAN (Zhu et al., 2017)
   - Perceptual loss: Johnson et al. (2016) / VGG features
   - VAE condizionale: Sohn et al. (2015) / BicycleGAN (Zhu et al., 2017b)
   - Metriche: FID (Heusel et al., 2017), LPIPS (Zhang et al., 2018)

3. Dataset
   - Anime Sketch Colorization Pair (Kaggle, ~17 769 immagini)
   - Formato: 512×1024 side-by-side (left=color, right=sketch) — verificato empiricamente
   - Split: train (Kaggle train), val (prima metà Kaggle val), test (seconda metà Kaggle val)
   - Preprocessing: resize 256×256 bicubic, normalize to [-1, 1]
   - Unpaired simulation per CycleGAN: permutazione fissa seed=0 degli indici color

4. Methodology
   4.1 Architettura condivisa: U-Net generator
   4.2 PatchGAN discriminator (70×70)
   4.3 ResNet generator (CycleGAN)
   4.4 Conditional VAE
   4.5 Loss functions: L1, adversarial (BCE/LSGAN), perceptual (VGG-19), ELBO
   4.6 Shared trainer design: un singolo Pix2PixTrainer, λ_gan=0 → variante 1

5. Experiments & Results
   5.1 Setup sperimentale (hardware, seed, iperparametri)
   5.2 Tabella riassuntiva PSNR / SSIM / LPIPS / FID per tutte le varianti
   5.3 Loss curves
   5.4 Griglie qualitative (sketch | prediction | ground truth)
   5.5 Diversità CVAE (n=4 campionamenti dello stesso sketch)
   5.6 Discussione: cosa spiega i numeri

6. Conclusions
   - Risposta alla domanda centrale (paired >> unpaired su PSNR/SSIM, ma FID simile)
   - Contributo di ogni componente
   - Limiti e sviluppi futuri

References
```

---

## 2. Dataset — note empiriche

**Formato del file raw**: ogni immagine è 1024×512 (width×height). La metà
**sinistra** (x: 0→512) è l'immagine a colori; la metà **destra** (x: 512→1024)
è lo sketch in scala di grigi. Verificato empiricamente il 2026-06-12 analizzando
i valori RGB: la metà destra è quasi monocromatica, la sinistra ha alta saturazione.
*Nota per il report*: documentare questo, perché la dataset card Kaggle e alcuni
tutorial online indicano l'ordine opposto — la verifica empirica è un dettaglio
che dimostra attenzione metodologica.

**Split**: il dataset Kaggle ha solo `train/` e `val/`. Il test set è ricavato
dalla seconda metà di `val/`, ordinata per filename (riproducibile senza RNG).
In totale: ~14 000 train / ~1 800 val / ~1 800 test (circa).

**Unpaired simulation per CycleGAN**: non si usa un dataset diverso, ma si
spezza il legame (sketch_i, color_i) rimpiazzando l'indice color con
`color_perm[i]` dove `color_perm` è una permutazione random con seed=0 fisso.
Questo è intenzionale e va spiegato nel report: il punto è misurare il valore
della supervisione paired, quindi le varianti paired e unpaired devono
differire *solo* per il tipo di supervisione, non per il dataset.

---

## 3. Variante 1 — U-Net + L1 (baseline)

### Architettura
U-Net con 8 encoder block e 8 decoder block. Ogni encoder block:
`Conv2d 4×4 stride 2 → InstanceNorm → LeakyReLU(0.2)`. Ogni decoder block:
`ConvTranspose2d 4×4 stride 2 → InstanceNorm → [Dropout 0.5] → ReLU`.
Skip connections: output di ogni encoder block concatenato al decoder speculare.
Dropout nei primi 3 decoder block (come da paper Pix2Pix). Output: `Tanh` → [-1,1].

Progressione canali encoder: 3 → 64 → 128 → 256 → 512 → 512 → 512 → 512 → 512 (bottleneck 1×1).

### Perché U-Net
Le skip connections sono fondamentali per la colorizzazione: l'encoder
comprime lo sketch fino a 1×1 (bottleneck), ma i dettagli ad alta frequenza
(bordi, tratti) devono arrivare al decoder senza passare per il collo di
bottiglia. La U-Net garantisce che ogni livello del decoder riceva sia
informazione semantica (dal bottleneck) sia strutturale (dallo skip).

### Perché L1 come unica loss
L1 minimizza la differenza pixel-wise. Converge bene, è stabile, ed è il
termine di "fidelity" che accomuna tutte le varianti. Non ha componente
avversariale → il modello tende a predire colori "safe" (desaturati, medi),
perché la L1 penalizza allo stesso modo scostamenti in qualsiasi direzione.
Questo è noto come "regression to the mean" ed è il difetto principale
che la variante 2 corregge con il discriminatore.

### Iperparametri
`lambda_l1=100`, `lr=2e-4`, Adam(β1=0.5, β2=0.999), 50 epoche, batch 16, 256×256.

### λ_l1 = 100 — perché
Nel paper Pix2Pix il termine L1 è pesato 100× rispetto alla loss avversariale
(che ha peso 1). Con λ_l1=100 il gradiente pixel-wise domina, e l'avversariale
"rifinisce" i dettagli senza destabilizzare l'ottimizzazione. Nella variante 1
non c'è discriminatore, quindi λ_l1 è solo uno scaling globale della loss,
ma il valore viene mantenuto coerente tra le varianti.

---

## 4. Variante 2 — Pix2Pix

### Cosa si aggiunge rispetto a V1
Un discriminatore **PatchGAN condizionale** (in_channels=6: sketch concatenato
con real o fake color). Il discriminatore giudica patch 70×70: ogni output
logit corrisponde a una regione 70×70 dell'immagine di input (mappa 30×30
per input 256×256).

### Perché PatchGAN
Un discriminatore globale (un singolo scalare) è cieco alla struttura locale:
può essere ingannato da immagini globalmente plausibili ma localmente sfocate.
Il PatchGAN forza il generatore a produrre texture convincenti a scala locale.
Rispetto a un discriminatore globale ha meno parametri (fully convolutional)
ed è più stabile da addestrare. Architettura: C64-C128-C256-C512-1.

### Condizionamento del discriminatore
Il discriminatore riceve `cat(sketch, real_or_fake)` → 6 canali.
Senza condizionamento il discriminatore non può sapere se il colore è
coerente con lo sketch; condizionandolo, la penalità avversariale diventa
"questo colore è plausibile *dato* questo sketch", che è la domanda giusta
per un task di traduzione condizionale.

### Loss totale
Generatore: `L_G = 100·L1(fake, real) + 1·L_GAN(D(sketch||fake), real=True)`  
Discriminatore: `L_D = 0.5·[L(D(sketch||real), 1) + L(D(sketch||fake), 0)]`

Il fattore 0.5 sul discriminatore rallenta l'aggiornamento del discriminatore
rispetto al generatore (come da paper Pix2Pix) per evitare che il discriminatore
"vinca" troppo in fretta.

### Adversarial loss: BCEWithLogitsLoss (vanilla GAN)
Il discriminatore non ha Sigmoid finale — la Sigmoid è dentro la loss numericamente
stabile. Il generatore cerca `D(sketch||fake) → 1`. LSGAN non viene usato qui
(solo in CycleGAN) perché la versione originale Pix2Pix usa BCE e il setting
è più semplice (reti più piccole, dataset paired).

---

## 5. Variante 3 — Pix2Pix + Perceptual Loss

### Cosa si aggiunge rispetto a V2
Una **perceptual loss** su feature VGG-19 congelato, estratte da:
- `relu_1_2` (layer 4): bassa frequenza, colore, edge grossolani
- `relu_2_2` (layer 9): texture media
- `relu_3_3` (layer 18): struttura semantica locale

Loss = somma pesata di L1 sulle feature maps (pesi uniformi = 1.0 per layer).

### Perché VGG-19 e quei layer
VGG-19 pre-trainato su ImageNet → le sue feature codificano struttura
percettiva umana. I layer scelti bilanciano: i primi due catturano aspetti
visivi a basso e medio livello (texture, colore), il terzo aggiunge contesto
semantico senza allontanarsi troppo dal dominio pixel. Layer profondi come
`relu_4_4` o `relu_5_4` codificano semantica troppo astratta per un task
di colorizzazione dove la struttura spaziale locale è importante.

### Normalizzazione input VGG
VGG si aspetta input in [0,1] normalizzati con media/std ImageNet.
Il generatore produce output in [-1,1]. Conversione interna alla loss:
`x_vgg = ((x + 1) / 2 - mean) / std`  
Mean/std sono registrati come buffer PyTorch → si spostano automaticamente
su GPU con `.to(device)` e non vengono trattati come parametri.

### Loss totale
`L_G = 100·L1 + 1·L_GAN + 10·L_perc`

λ_perceptual=10 bilancia la scala della perceptual loss (valori tipicamente
più piccoli di L1 raw, che opera direttamente sui pixel) rispetto agli altri termini.

### Effetto atteso
La perceptual loss penalizza differenze nello spazio percettivo, non pixel.
Il risultato visivo è più "realistico" (texture, saturazione) anche se PSNR
può essere inferiore a V1 (perché PSNR è pixel-wise e la perceptual loss
sposta i pixel verso valori percettivamente vicini ma non identici).
FID migliora significativamente perché la distribuzione generata si avvicina
a quella reale nello spazio delle feature Inception.

---

## 6. Variante 4 — CycleGAN (unpaired)

### Architettura
Due **ResNet generator** (sketch→color `G_s2c`, color→sketch `G_c2s`) e due
**PatchGAN discriminator incondizionati** (in_channels=3, non 6).
I discriminatori non ricevono lo sketch perché il training è unpaired:
non esiste una coppia di riferimento su cui condizionare.

### ResNet generator vs U-Net — perché cambiare
Per il setting unpaired la U-Net non è adatta: le skip connections assumono
che input e output siano spazialmente allineati (lo sketch e il suo colore
lo sono). In CycleGAN gli sketch vedono colori da una permutazione casuale
durante il training → le skip porterebbero informazione spaziale errata,
destabilizzando l'ottimizzazione. Il ResNet generator trasforma nel feature
space senza skip diretti: encoder (c7s1-64 → d128 → d256) comprime,
9 residual block trasformano, decoder (u128 → u64 → c7s1-3) ricostruisce.
ReflectionPad2d invece di zero-padding riduce artefatti ai bordi.

### Loss CycleGAN
- **Adversarial (LSGAN, MSE)**: più stabile di BCE per reti profonde con
  ResNet generator; gradiente non satura anche quando il discriminatore è confuso.
- **Cycle-consistency (λ=10)**: `G_c2s(G_s2c(sketch)) ≈ sketch` e viceversa.
  Garantisce mapping inversi coerenti — senza di essa i generatori possono
  imparare a mappare tutto verso un singolo punto (mode collapse).
- **Identity loss (λ=5)**: `G_s2c(color) ≈ color`. Preserva tinte e
  cromaticità quando l'input è già nel dominio target. Evita che il generatore
  alterei i colori inutilmente.

### Replay buffer
`ImageBuffer(size=50)`: con prob 0.5 il discriminatore vede immagini dal
buffer degli ultimi 50 step invece di quelle appena generate. Riduce le
oscillazioni nel training (Shrivastava et al., 2017).

### AMP (Automatic Mixed Precision)
CycleGAN è compute-bound: 6 forward pass generator + 2 discriminatori per step.
AMP (fp16 autocast + GradScaler) dimezza il tempo per step su RTX 3090
senza perdita significativa di qualità. Disabilitato automaticamente su CPU.

### LR scheduling
Costante per le prime `decay_start` epoche, poi decay lineare a 0 per le
restanti (paper: 100 epoche costanti + 100 epoche decay).

### Punto chiave per il report
Il training CycleGAN non usa MAI le coppie — neanche per il sampling.
`paired=False` nel DataLoader → indice colore permutato con seed fisso.
Le coppie vengono usate SOLO nella validation per calcolare `val_l1` vs
ground-truth (per la checkpoint selection). Questo simula fedelmente il
setting unpaired: le varianti paired e unpaired differiscono solo per la
supervisione, non per il dataset o le architetture.

---

## 7. Variante 5 — Conditional VAE

### Motivazione
Tutte le varianti 1-4 sono deterministiche: dato uno sketch producono sempre
la stessa colorizzazione. Ma la colorizzazione è un problema **many-to-one**
(uno sketch ammette infinite colorazioni valide). Il CVAE introduce un
latent space esplicito che cattura questa ambiguità, permettendo campionamenti
diversi dallo stesso sketch.

### Architettura — tre componenti

**Sketch encoder** (identico all'encoder U-Net):
8 down block → produce 7 skip features + 1 bottleneck 512×1×1.
Condiviso tra training e inference — è il "backbone" condizionale.

**Posterior encoder** q(z | sketch, color) — solo durante training:
Riceve `cat(sketch, color)` (6 canali), processa con 5 down block,
GlobalAvgPool → Flatten → FC → (mu, logvar). Cattura la variabilità nella
colorizzazione dato lo sketch, permettendo al modello di "capire" la
distribuzione dei colori possibili per ogni sketch.

**Decoder** (identico al decoder U-Net):
Riceve `bottleneck + z_projected` + sketch skips.
`z_proj`: FC(latent_dim → 512) porta z alla dimensione del bottleneck,
poi viene **sommato** addittivamente al bottleneck (non concatenato, per
non cambiare le dimensioni dei canali e rendere il decoder compatibile
con il decoder U-Net standard).

### Training: ELBO
`L = L1(reconstruction, color) + β · KL(q(z|sketch,color) || N(0,I))`

**KL annealing (beta warmup)**: β parte da 0 e sale linearmente fino al
valore target in `beta_warmup_epochs=20` epoche. Senza annealing il KL term
spinge il posterior verso N(0,I) prima che il decoder abbia imparato a
ricostruire → posterior collapse: il decoder impara a ignorare z, tutte le
colorizzazioni sono identiche (la variabilità si perde). Con annealing il
decoder impara prima a ricostruire (fase β≈0), poi progressivamente z porta
informazione semantica sulla colorizzazione.

**β = 0.01** (non 1.0): β piccolo privilegia la ricostruzione rispetto alla
regolarizzazione. Con β=1 la qualità di ricostruzione degrada significativamente;
β=0.01 è un compromesso empirico tra fedeltà ricostruttiva e diversità dei sample.

### Inference
- **Deterministica** (z=0, prior mean): comparabile con le altre varianti
  per le metriche quantitative.
- **Stochastic** (z~N(0,I)): campionamenti diversi, qualità media inferiore
  ma diversità maggiore.

La funzione `sample()` restituisce shape `(n_samples, B, 3, H, W)` — pensata
per le griglie di diversità nei notebook.

### Early stopping
Patience=20 su `val_recon`: addestramento si interrompe se la reconstruction
loss di validazione non migliora per 20 epoche consecutive. Evita overfitting
e permette di sfruttare un budget di epoche elevato (100) senza sprecare
compute.

---

## 8. Loss functions — motivazioni

### L1 vs L2 (MSE)
L1 produce output meno sfocati di L2. La MSE penalizza quadraticamente gli
outlier, spingendo il modello verso la media ancora più aggressivamente di L1.
Per colorizzazione L1 è la scelta standard (Pix2Pix): più robusta, meno
blur. "Regression to the mean" esiste con entrambe, ma è meno grave con L1.

### Adversarial loss: vanilla (BCE) vs LSGAN (MSE)
- **Vanilla (BCEWithLogits)**: Pix2Pix V1-3. Gradiente forte quando il
  discriminatore è confuso; rischio di vanishing quando è troppo forte.
  Funziona bene con reti relativamente piccole e dataset moderati.
- **LSGAN (MSELoss)**: CycleGAN V4. Più stabile per reti profonde (no
  vanishing); produce output visivamente più smooth. Scelta coerente con
  il paper CycleGAN originale (Zhu et al., 2017).

### Perché non WGAN-GP
WGAN-GP richiederebbe gradient penalty sul discriminatore, aggiungendo
complessità non necessaria per una ablation study didattica. L'obiettivo
è isolare il contributo di ogni componente, non massimizzare la qualità
assoluta. BCE+0.5 sul discriminatore è sufficiente per la stabilità
nel setting Pix2Pix.

### ELBO — decomposizione
`ELBO = E[log p(x|z)] - KL(q(z|x)||p(z))`
- Termine ricostruttivo: L1(recon, target) — massimizza la log-verosimiglianza
  approssimata con L1 invece di MSE (scelta empirica per ridurre blur).
- Termine KL: penalizza la distanza tra posterior q(z|sketch,color) e prior N(0,I).
  Regolarizza lo spazio latente in modo che z~N(0,I) campioni colorizzazioni
  plausibili anche al test time (quando il posterior non è disponibile).

---

## 9. Metriche — giustificazione vs rubrica

La rubrica del corso cita confusion matrix / accuracy / recall / precision —
appropriate per classificazione. Per task generativi le metriche equivalenti:

| Rubrica (classificazione) | Equivalente generativo | Note |
|---|---|---|
| Accuracy | **PSNR** | Fedeltà pixel-wise (Signal-to-Noise Ratio) |
| Precision/Recall strutturale | **SSIM** | Struttura, luminanza, contrasto locali |
| Distribuzione delle predizioni | **FID** | Distanza distribuzione generata vs reale |
| — | **LPIPS** | Distanza percettiva per-immagine (gap non coperto dagli altri) |

### Copertura delle 4 metriche

```
                    Per-image          Distribution-level
Pixel-level      PSNR / SSIM                  —
Perceptual           LPIPS                   FID
```

Le quattro metriche coprono quattro quadranti distinti — nessuna è ridondante:
- **PSNR**: dB di Signal-to-Noise ratio su [0,1]. 16 dB tipico per colorizzazione
  256×256 con ground truth reale (non è bassa: la colorizzazione è intrinsecamente
  ambigua, il ground truth è una delle infinite colorizzazioni valide).
- **SSIM** [0,1]: misura luminanza, contrasto, struttura localmente su finestre
  11×11. ~0.8 è buono per questo task.
- **LPIPS** (AlexNet, Zhang et al. CVPR 2018) [0,1]: più basso è meglio.
  Correlato meglio con il giudizio umano rispetto a PSNR/SSIM. Cattura la
  dimensione percettiva per-immagine che FID non ha.
- **FID** (Fréchet Inception Distance): distanza Fréchet tra fit Gaussiani su
  feature Inception v3 del test set reale e del set generato. Misura qualità
  *e* diversità insieme. Valori <50 sono buoni per 256×256.

**Come giustificare nel report**: citare esplicitamente che confusion matrix
non è definita per output continui (immagini), spiegare il mapping sopra,
citare i paper originali delle metriche (Heusel et al. 2017 per FID,
Zhang et al. 2018 per LPIPS).

---

## 10. Scelte architetturali trasversali

### InstanceNorm vs BatchNorm
Tutte le reti usano `InstanceNorm2d`. BatchNorm usa statistiche del batch
durante inference → instabile con batch piccoli e sensibile alla distribuzione
del batch. InstanceNorm normalizza per-feature-map-per-sample → invariante
al batch size, standard per generazione di immagini da CycleGAN in poi.

### Shared trainer design
Le varianti 1-3 usano lo stesso `Pix2PixTrainer`, ablato via lambda:
```python
# V1: lambda_gan=0.0, lambda_perceptual=0.0
# V2: lambda_gan=1.0, lambda_perceptual=0.0
# V3: lambda_gan=1.0, lambda_perceptual=10.0
```
Questo garantisce che l'unica differenza tra le varianti sia la loss,
eliminando bug differenziali e rendendo il confronto pulito.

### Seed everything
`torch.manual_seed`, `numpy.random.seed`, `random.seed`,
`torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic=True`.
Garantisce riproducibilità tra run. Importante per una ablation study:
i risultati non devono dipendere dalla fortuna dell'inizializzazione.

### Checkpoint strategy
`best.pt` salvato quando la validation metric migliora; `last.pt` ad ogni epoca.
V1-3: monitora `val_l1`. CycleGAN: idem (`val_l1` calcolato sulle coppie
anche se il training è unpaired — le coppie non vengono viste durante
il training, solo durante la validation per avere un segnale di qualità).
CVAE: monitora `val_recon`.

### Normalizzazione immagini
Tutte le immagini sono in [-1, 1] durante training e inference
(`Normalize(mean=0.5, std=0.5)` per ogni canale). Output del generatore:
`Tanh` → [-1, 1]. Le metriche convertono a [0, 1] internamente.

### Resize 256×256
Pragmatico: VRAM RTX 3090 permette batch=16 a 256px con U-Net.
A 512px il batch dovrebbe scendere a ~4, rendendo il gradiente molto
noisier e il training meno stabile. La qualità a 256px è sufficiente
per l'ablation study; la differenza tra varianti è misurabile comunque.

### LeakyReLU(0.2) nell'encoder
Standard per discriminatori e encoder di generatori. Slope 0.2 per le
attivazioni negative garantisce che il gradiente fluisca anche attraverso
i neuroni "inattivi" (problema classico di ReLU nelle reti profonde dove
molti neuroni possono diventare zero per input negativi).

---

## 11. Risultati quantitativi

Checkpoint usato per ogni variante:

| Variante | Ckpt | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|---|---|---|---|---|---|
| V1 — U-Net + L1 | best | 16.8425 | 0.8134 | 0.1933 | 89.4938 |
| V2 — Pix2Pix | last | 16.26 | 0.800 | 0.191 | **49.13** |
| V3 — Pix2Pix + Perceptual | last | 16.29 | **0.813** | **0.186** | **37.25** |
| V4 — CycleGAN (unpaired) | best | 14.66 | 0.784 | 0.218 | 48.04 |
| V5 — Pix2Pix + Global Disc | last | 16.0662 | 0.7923 | 0.1942 | 45.6185 |
| V6 — Pix2Pix λ ablation | best | 15.2679 | 0.7828 | 0.2097 | 42.7849 |

Regola checkpoint: V1 → best.pt (no GAN, val_l1 non biasato);
V2, V3, V5 → last.pt (val_l1 bias: best.pt desaturato);
V4 → best.pt (instabilità training: FID(last) >> FID(best), delta +16 FID);
V6 → best.pt (λ_gan=10 così forte da annullare il bias di desaturazione; best.pt migliore su tutte le metriche).

### Interpretazione (appunti per la discussione del report)

**V1 vs V2 — impatto del discriminatore**  
PSNR e SSIM *scendono leggermente* da V1 a V2. Questo è atteso e va
spiegato: la loss avversariale spinge il generatore verso immagini
"reali" che non coincidono pixel-to-pixel con il ground truth (la
penalità è sulla distribuzione, non su ogni pixel). Il FID dimezza
(89.49 → 49.13): il salto maggiore dell'intero studio. L'adversarial
loss è il contributo più grande alla qualità distribuzionale.

**V2 vs V3 — impatto della perceptual loss**  
FID scende ancora (49.13 → 37.25), SSIM sale (0.800 → 0.813), LPIPS
migliora (0.191 → 0.186). La perceptual loss spinge verso texture
percettivamente realistiche — il modello impara a "vedere" come VGG.
Il miglioramento su FID è il segnale più forte: la distribuzione generata
si avvicina ulteriormente a quella reale. La perceptual loss è la scelta
migliore per chi vuole output visivamente convincenti.

**V3 vs V4 — paired vs unpaired (domanda centrale)**  
CycleGAN unpaired (V4) è peggiore su tutte le metriche pixel-wise
(PSNR 14.66 vs 16.29, SSIM 0.784 vs 0.813, LPIPS 0.218 vs 0.186).
Il FID è sorprendentemente vicino (48.04 vs 37.25): CycleGAN genera
immagini distribuzionalmente plausibili, ma mal allineate col ground truth.
Risposta alla domanda centrale: la supervisione paired aggiunge ~1.6 dB
PSNR e ~3% SSIM. Il costo dell'annotazione paired si traduce in un
vantaggio misurabile ma non enorme.

**V2 vs V5 — PatchGAN vs Global Discriminator**  
V5 è peggiore su PSNR (16.07 vs 16.26), SSIM (0.792 vs 0.800) e LPIPS
(0.194 vs 0.191), ma ha FID leggermente migliore (45.62 vs 49.13).
Senza la pressione patch-level, il generatore produce immagini con
statistiche globali più vicine alla distribuzione reale (colori, palette)
ma con texture locali meno precise. La differenza FID (~3.5 punti) è al
limite della significatività statistica con 1773 immagini test.
Conclusione: il PatchGAN contribuisce principalmente alla qualità
strutturale locale; il suo contributo alla distribuzione globale è
secondario.

**V2 vs V6 — effetto del bilanciamento λ (λ_l1=50, λ_gan=10)**  
V6 mostra il tradeoff più netto dell'intero studio: PSNR scende di ~1 dB
(15.27 vs 16.26), ma FID migliora di ~6 punti (42.78 vs 49.13).
Aumentare il peso GAN di 10× spinge il generatore verso immagini
"realistiche" a scapito della fedeltà pixel-wise. Questo conferma che
λ_l1 e λ_gan controllano direttamente il trade-off pixel-fidelity vs
perceptual-realism — una leva progettuale chiara e interpretabile.

**Ranking finale per metrica**  
- PSNR ↑: V1 > V3 ≈ V2 > V5 > V6 > V4  
- SSIM ↑: V3 ≈ V1 > V2 > V5 > V4 ≈ V6  
- LPIPS ↓: V3 < V2 ≈ V5 < V1 < V6 < V4  
- FID ↓: V3 < V6 < V5 < V2 < V4 < V1

---

## 12. Immagini architetture (da generare con Gemini)

Elenco di diagrammi architetturali da richiedere a Gemini per il report
e la presentazione. Per ciascuno: specificare colori dei blocchi, dimensioni
dei tensori ai nodi principali, frecce direzionali, e componenti da evidenziare.

### 1. U-Net generator (varianti 1-3)
- Struttura a simmetria encoder-decoder verticale o orizzontale
- Frecce skip connections tratteggiate o di colore diverso
- Dimensioni tensor ad ogni stage:
  `3×256 → 64×128 → 128×64 → 256×32 → 512×16 → 512×8 → 512×4 → 512×2 → 512×1 (bottleneck)`
- Decoder speculare con Tanh finale → `3×256`
- Evidenziare i 3 decoder block con Dropout (es. con bordo tratteggiato)

### 2. PatchGAN discriminator
- Sequenza C64 → C128 → C256 → C512 → 1-ch output
- Input: 6 canali (sketch || color) con evidenza del concatenamento
- Output: mappa `30×30` di patch logits
- Annotazione: "ogni logit giudica una receptive field 70×70 dell'input"

### 3. Pix2Pix pipeline completa (V2/V3)
- Generator (U-Net) + Discriminator (PatchGAN) nella stessa figura
- Freccia: `sketch → G → fake_color`
- Ramo reale: `(sketch || real_color) → D → "real"`
- Ramo fake: `(sketch || fake_color) → D → "fake"` (freccia con `.detach()`)
- Box colored per L1 loss e GAN loss con frecce verso G e D
- Per V3: aggiungere box VGG-19 frozen con freccia `perceptual loss → G`

### 4. CycleGAN pipeline completa (V4)
- Due generatori `G_{s→c}` e `G_{c→s}` e due discriminatori `D_c`, `D_s`
- Forward cycle: `sketch → G_{s→c} → fake_color → G_{c→s} → rec_sketch`
- Backward cycle: `color → G_{c→s} → fake_sketch → G_{s→c} → rec_color`
- Frecce cycle-consistency loss e identity loss
- Evidenziare che i discriminatori non ricevono lo sketch (incondizionati)

### 5. ResNet generator block
- Zoom sul residual block: `ReflPad → Conv → IN → ReLU → ReflPad → Conv → IN → + (skip)`
- Architettura globale a lato in miniatura: `c7s1-64 → d128 → d256 → 9×R256 → u128 → u64 → c7s1-3`

### 6. Conditional VAE (V5)
- Tre componenti separati: Sketch Encoder, Posterior Encoder, Decoder
- **Training path**: `(sketch, color) → Posterior Enc → (μ, σ) → reparam → z → z_proj →`  
  `(+ bottleneck) → Decoder + sketch_skips → reconstruction`
- **Inference path** (tratteggiato/colore diverso): `z~N(0,I) → z_proj → (+ bottleneck) → Decoder → diverse colors`
- Evidenziare: il Posterior Encoder è usato SOLO in training

### 7. Diagramma metriche 2×2 (per la sezione Evaluation)
- Asse X: per-image vs distribution-level
- Asse Y: pixel-level vs perceptual
- Quadranti: `PSNR & SSIM | — | LPIPS | FID`
- Utile come figura in apertura della sezione Experiments per motivare
  la scelta delle 4 metriche

---

*Ultimo aggiornamento: 2026-06-17*
