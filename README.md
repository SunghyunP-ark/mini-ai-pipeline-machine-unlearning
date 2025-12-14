# mini-ai-pipeline-machine-unlearning

A small **machine unlearning pipeline** on CIFAR‑10 using a pretrained ViT model.  
This project compares several unlearning strategies. The methods are evaluated on **accuracy, uncertainty (entropy), distributional shift (Jensen–Shannon divergence)** and **t‑SNE visualisations**.

## 1. Project overview

We study the following setting:

- **Dataset:** CIFAR‑10 (10 classes)
- **Base model:** ViT Tiny (`vit_tiny_patch16_224`) from `timm`, pretrained on ImageNet
- **Forget class:** class index 0 (Airplane in CIFAR‑10)

The data is split into:

- **Train**
  - `forget`: all training samples whose label equals the forget class
  - `retain`: all training samples with other labels
  - `all`: the full training set (retain + forget)
- **Test**
  - `test_forget`: test samples belonging to the forget class
  - `test_retain`: test samples for the remaining classes

Two reference models are trained:

1. **Original model** – trained on the full `all` data. This represents an *infected* model that knows everything, including the forget class.
2. **Gold standard model** – trained only on the `retain` set. This is an *ideal* model that never saw any forget samples.

Then three unlearning methods are applied to the original model and compared:

- **Fine‑tuning (Retain only)** – unlearn by retraining on only the retain set.
- **Gradient Ascent (Forget only)** – actively destroy knowledge by maximising loss on the forget set.
- **2‑Stage unlearning (ours)** – first make the model unsure about the forget class, then separate forget and retain outputs while preserving accuracy on the retain data.

The performance of these unlearned models is compared against the original and gold models.

## 2. Dependencies

Core libraries used:

- **PyTorch** & **torchvision** – model and dataset utilities
- **timm** – pretrained ViT
- **numpy** and **pandas**
- **matplotlib** 
- **tqdm** 
- **scikit‑learn** – t‑SNE
- **scipy** – Jensen–Shannon divergence
- **seaborn** – nicer scatter plots

Install dependencies via `pip install -r requirements.txt`.

## 3. Dataset & dataloaders

The code uses `torchvision.datasets.CIFAR10` and the following preprocessing pipeline:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
```

The helper `get_dataloaders(forget_class_idx)`:

- downloads CIFAR‑10,
- creates index splits for `forget`, `retain` and `all`,
- creates the corresponding **train** and **test** `DataLoader` objects:

```python
loaders = {
    'all':         ...  # train on all data
    'forget':      ...  # train forget subset
    'retain':      ...  # train retain subset
    'test_forget': ...  # evaluate on forget class in test set
    'test_retain': ...  # evaluate on retain classes in test set
}
```

## 4. Model

We use a pretrained ViT from `timm` and replace the classification head.

```python
def get_model():
    model = timm.create_model(Config.MODEL_NAME, pretrained=True)
    model.head = nn.Linear(model.head.in_features, Config.NUM_CLASSES)
    return model.to(Config.DEVICE)
```

Here `Config.MODEL_NAME` is `'vit_tiny_patch16_224'`. The linear head is adjusted to the ten CIFAR‑10 classes.

## 5. Training procedures

### 5.1 Seed & config

```python
class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SEED = 42
    FORGET_CLASS = 0
    MODEL_NAME = 'vit_tiny_patch16_224'
    NUM_CLASSES = 10
    BATCH_SIZE = 64
    LR_TRAIN = 1e-4
    LR_UNLEARN = 1e-4
    NUM_WORKERS = 0
    EPOCHS_TRAIN = 3
    EPOCHS_UNLEARN = 5
```

A `set_seed` helper sets seeds for reproducibility across `torch`, `numpy` and Python’s `random`.

### 5.2 Standard training

A helper trains a model with standard cross‑entropy loss on a given loader:

```python
def train_standard(model, loader, epochs, name="Model"):
    optimizer = optim.SGD(model.parameters(), lr=Config.LR_TRAIN, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        ...
        optimizer.step()
    return model
```

We train two reference models:

1. **Original model** on `loaders['all']`.
2. **Gold model** on `loaders['retain']`.

## 6. Unlearning methods

### 6.1 Fine‑tuning (Retain only)

**Idea:** re‑train only on the retain set, hoping the forget class information fades away.

- Objective:
  \$[\min_{\theta} \mathbb{E}_{(x_r,y_r)\in D_r} \mathrm{CE}(p_\theta(y_r\mid x_r), y_r).\]$
- Implementation: reuse standard training but feed only `retain` data.

### 6.2 Gradient ascent (Forget only)

**Idea:** actively destroy the knowledge of the forget class by maximising its classification loss.

- Objective (per step): maximise loss on forget data:
  \$[\max_{\theta} \mathbb{E}_{(x_f,y_f)\in D_f} \mathrm{CE}(p_\theta(y_f\mid x_f), y_f).\]$
  In code this is implemented by minimising the negative loss on `forget` data.
- Very aggressive and can cause catastrophic forgetting on retain classes.

### 6.3 2‑Stage unlearning (ours)

**Idea:** first confuse the model on the forget class, then separate forget and retain outputs while healing the retain performance.

1. **Stage 1 ** – enforce a uniform distribution on the forget data:
   \$[\min_{\theta} \mathbb{E}_{x_f\in D_f} D_\mathrm{KL}(u\,\|\,p_\theta(\cdot\mid x_f))\]$  
   where $u$ is the uniform distribution over the ten classes. This is applied once to break the model’s confidence on forget samples.

2. **Stage 2 ** – optimize the sum of two terms:
   - **Contrastive loss**: reduce the similarity between forget outputs and retain outputs:
     \$[\ell_{\text{con}}(Z_f,Z_r) = \frac{1}{m}\sum_{i=1}^m -\log\mathrm{softmax}(Z_fZ_r^\top)_{i,:}\]$
     thereby pushing forget representations away from retain representations.
   - **Retain loss**: standard cross entropy on retain data:
     \$[\ell_{\text{ret}}(\theta;B_r) = \frac{1}{m}\sum_j -\log p_\theta(y_r^{(j)}\mid x_r^{(j)}).\]$
   - The final loss for Stage 2 is:
     \$[L_{\text{stage2}} = \ell_{\text{con}} + \ell_{\text{ret}}.\]$

## 7. Evaluation metrics

### 7.1 Accuracy

`evaluate_accuracy` computes accuracy on the forget train set, retain train set, and the test splits. Results are reported in percentage.

### 7.2 Entropy on forget test

`evaluate_entropy` computes mean entropy of the predictive distribution on `test_forget`. Higher entropy indicates more uncertainty, which suggests better unlearning (to a point).

### 7.3 Jensen–Shannon divergence vs gold model

`compute_jsd` compares each forget test sample’s predictive distribution under the unlearned model to that of the gold model via the Jensen–Shannon distance. Lower values indicate the unlearned model behaves more like the ideal gold model on forget data.

### 7.4 t‑SNE visualisation

`visualize_tsne` samples a few retain and forget images, extracts features from each model using `model.forward_features`, embeds them into 2D via t‑SNE, and plots them. This helps visualise how well forget and retain representations separate after unlearning.


## 9. Possible extensions

- Test different forget classes by changing `Config.FORGET_CLASS`.
- Try other pretrained backbones by adjusting `Config.MODEL_NAME`.
- Explore more sophisticated unlearning objectives (e.g. weight‑saliency, multi‑task bilevel methods).  
- Log results to CSV or dashboards for better tracking.

---
