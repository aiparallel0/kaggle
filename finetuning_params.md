# DONUT · TrOCR · YOLO — Finetuning Parameter Reference

> **Format:** AI-agent optimized. Machine-readable JSON at `finetuning_params.json`. Structured by model → parameter with type, default, range, category, and impact rating.

> **Impact legend:** 🔴 critical · 🟠 high · 🟡 medium · 🟢 low

> **Sources:** clovaai/donut GitHub (config/train_cord.yaml, issues) | microsoft/unilm TrOCR README (fairseq training script) | docs.ultralytics.com/usage/cfg and /modes/train | arxiv:2111.15664 (DONUT paper) | arxiv:2109.10282 (TrOCR paper) | philschmid.de/fine-tuning-donut | towardsdatascience.com — UBIAI DONUT invoice finetuning | learnopencv.com — TrOCR getting started | medium.com — YOLOv8 best practices 2024

---

## DONUT — Document Understanding Transformer

| Field | Value |
|-------|-------|
| Architecture | Swin Transformer encoder + BART decoder |
| Task | OCR-free visual document understanding (parsing, classification, VQA) |
| Base model | `naver-clova-ix/donut-base` |
| Framework | PyTorch Lightning + HuggingFace Transformers |

### Image Resolution

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `input_size` | `list[int]` | `[1280,960]` | — | image_resolution | 🔴 critical | Image resolution fed to the Swin encoder. Width × Height. Pre-training used [2560,1920]; finetuning commonly downscaled to [1280,960] or [960,720] to reduce VRA |

<details><summary><code>input_size</code> — extended notes</summary>


**⚠️ Note:** Must be multiples of patch_size (32). Very large reductions from pretrain resolution may hurt convergence.

**Common values:** [[640,480],[960,720],[1280,960],[1920,1440],[2560,1920]]

</details>

### Training Core

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `train_batch_sizes` | `list[int]` | `[1]` | — | training_core | 🟠 high | Per-device batch sizes for training. Usually 1–2 on a single consumer GPU; can be [8] with larger GPUs or gradient accumulation. |
| `val_batch_sizes` | `list[int]` | `[1]` | — | training_core | 🟢 low | Per-device batch sizes for validation. |
| `max_epochs` | `int` | `30` | 5–200 | training_core | 🟠 high | Number of full passes over the training data. Set to -1 when using max_steps instead. WARNING: setting both max_epochs > 0 and max_steps > 0 causes division-by- |
| `max_steps` | `int` | `-1` | — | training_core | 🟡 medium | If > 0, overrides max_epochs. Useful for CPU training where epoch-based training causes division-by-zero. Set max_epochs=-1 when using max_steps. |

<details><summary><code>train_batch_sizes</code> — extended notes</summary>


**⚠️ Note:** List format because DONUT supports multi-dataset training with different batch sizes.

</details>

<details><summary><code>max_steps</code> — extended notes</summary>


**⚠️ Note:** Use max_steps for CPU runs or when you want step-based control.

</details>

### Sequence Length

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `max_length` | `int` | `768` | — | sequence_length | 🟠 high | Maximum decoder output token length. Caps the generated JSON sequence length. Longer documents or more complex JSON outputs require larger values. Larger values |

### Optimization

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `lr` | `float` | `0.00003` | 0.00001–0.0001 | optimization | 🔴 critical | Peak learning rate with AdamW optimizer. 3e-5 is the official CORD finetune default. Use 2e-5 for smaller datasets to avoid catastrophic forgetting. |
| `warmup_steps` | `int` | `300` | — | optimization | 🟡 medium | Linear LR warmup steps before reaching peak LR. Recommended ≈10% of total training steps. Formula: total_steps = (num_training_samples_per_epoch / batch_size) × |
| `gradient_clip_val` | `float` | `1` | — | optimization | 🟡 medium | Global gradient norm clipping threshold. Prevents exploding gradients in the transformer decoder. 1.0 is the universal default. |
| `weight_decay` | `float` | `0.01` | 0–0.1 | optimization | 🟢 low | L2 regularisation on AdamW optimizer parameters. |
| `gradient_accumulation_steps` | `int` | `1` | 1–16 | optimization | 🟠 high | Accumulate gradients over N steps before optimizer update. Effective batch = batch_size × gradient_accumulation_steps. Use to simulate larger batches on VRAM-co |

### Image Preprocessing

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `align_long_axis` | `bool` | `false` | — | image_preprocessing | 🟡 medium | Whether to rotate portrait images to landscape orientation to better utilise horizontal text reading. Usually False for structured documents like receipts. Set  |

<details><summary><code>align_long_axis</code> — extended notes</summary>

**Alias:** `do_align_long_axis (on DonutProcessor)`

</details>

### Data

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `num_training_samples_per_epoch` | `int` | `800` | — | data | 🟡 medium | Number of training samples drawn per epoch. Set to -1 to use all available training samples. Useful to control iteration count when dataset is very large or ver |
| `sort_json_key` | `bool` | `false` | — | data | 🟡 medium | Whether to sort JSON keys alphabetically when preparing ground truth tokens. MUST be False for preprocessed datasets like CORD-v2. Setting True on CORD corrupts |

<details><summary><code>sort_json_key</code> — extended notes</summary>


**⚠️ Note:** CRITICAL: set False for CORD-v2. Can be True for raw JSON datasets that need normalization.

</details>

### Validation

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `check_val_every_n_epoch` | `int` | `3` | 1–20 | validation | 🟢 low | Run validation every N epochs. Reduce to 1 for small datasets to catch early convergence. |
| `val_check_interval` | `float | int` | `1` | — | validation | 🟢 low | Validation frequency within an epoch. Float = fraction of epoch (0.2 = validate 5× per epoch). Int = number of steps. Use 0.2 for fast-converging small datasets |

### Distributed

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `num_nodes` | `int` | `1` | — | distributed | 🟠 high | Number of compute nodes for multi-node distributed training. |

### Performance

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `num_workers` | `int` | `8` | 0–32 | performance | 🟡 medium | DataLoader worker processes for parallel image loading. Set to 0 for debugging. |

### Reproducibility

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `seed` | `int` | `2022` | — | reproducibility | 🟢 low | Random seed for reproducible training runs. |

### Mixed Precision

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `precision` | `int | str` | `16` | 16 \| 32 \| bf16 | mixed_precision | 🟠 high | PyTorch Lightning precision flag. 16 = FP16 AMP mixed precision, halving VRAM usage and accelerating training on Tensor Core GPUs. 32 = full precision. bf16 = b |

<details><summary><code>precision</code> — extended notes</summary>

**Alias:** `fp16=True in HuggingFace Seq2SeqTrainingArguments`

</details>

### Architecture

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `swin_window_size` | `int` | `—` | 7 \| 8 \| 10 \| 12 | architecture | 🔴 critical | Swin Transformer attention window size. donut-base uses 10; donut-proto used 8. CHANGING THIS from the pretrain value forces re-initialization of all attention  |
| `encoder_layer_depths` | `list[int]` | `—` | — | architecture | 🔴 critical | Swin encoder stage depths. donut-base = {2,2,14,2}; donut-proto = {2,2,18,2}. Only relevant if training a custom base model from scratch. |
| `decoder_layers` | `int` | `—` | — | architecture | 🟡 medium | Number of BART decoder transformer layers. |

<details><summary><code>swin_window_size</code> — extended notes</summary>


**⚠️ Note:** RARELY DOCUMENTED in finetuning guides. One of the most impactful hidden parameters.

</details>

### Training Management

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `resume_from_checkpoint_path` | `str | null` | `null` | — | training_management | 🟢 low | Path to a PyTorch Lightning checkpoint to resume training from. Set to checkpoint path string to continue interrupted training. |

---

## TrOCR — Transformer-based Optical Character Recognition

| Field | Value |
|-------|-------|
| Architecture | BEiT/DeiT encoder + RoBERTa/UniLM decoder |
| Task | Line-level text recognition (printed, handwritten, scene text) |
| Base model | `microsoft/trocr-base-printed | microsoft/trocr-large-handwritten` |
| Framework | fairseq (official) | HuggingFace Transformers (community) |

### Image Resolution

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `input_size` | `int` | `384` | — | image_resolution | 🔴 critical | All input images are ALWAYS resized to 384×384 before patch extraction. This is hardcoded by the DeiT/BEiT encoder architecture — it is NOT configurable without |

<details><summary><code>input_size</code> — extended notes</summary>


**⚠️ Note:** Known limitation — discussed in microsoft/unilm issue #674. Workaround: pad image to square before resizing, or use sliding window.

</details>

### Architecture

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `patch_size` | `int` | `16` | — | architecture | 🔴 critical | Each image is divided into 16×16 pixel patches. (384/16)² = 576 patch tokens are fed to the encoder. Not configurable without changing architecture. |
| `decoder_pretrained` | `str` | `"roberta2"` | roberta2 \| unilm | architecture | 🟠 high | Which pretrained checkpoint to initialize the text decoder from. roberta2 for printed/scene text. unilm for handwritten text. |
| `arch` | `str` | `"trocr_base"` | trocr_small \| trocr_base \| trocr_large | architecture | 🔴 critical | Model size variant. large: 558M params, 384px encoder; base: 334M; small: 62M. Larger = better accuracy, more VRAM, slower inference. |

<details><summary><code>patch_size</code> — extended notes</summary>


**Derived formula:** `num_patches = (input_size / patch_size)^2 = 576`

</details>

<details><summary><code>arch</code> — extended notes</summary>


**Model size details:**
- `trocr_small`: {"params":"62M","encoder":"DeiT-Small"}
- `trocr_base`: {"params":"334M","encoder":"BEiT-Base"}
- `trocr_large`: {"params":"558M","encoder":"BEiT-Large"}

</details>

### Training Core

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `batch_size` | `int` | `8` | — | training_core | 🟠 high | Per-GPU batch size. Official Microsoft training used BSZ=8 across 8 GPUs = 64 effective batch. Community finetuning uses 4–16 per GPU. |
| `max_epoch` | `int` | `300` | — | training_core | 🟠 high | Maximum training epochs. Official used 300 with patience=20 early stopping. Community finetuning typically 5–30 epochs with small datasets. |
| `patience` | `int` | `20` | — | training_core | 🟠 high | Early stopping: halt if no CER improvement for N epochs. Use 3–5 for community finetuning; 20 for full official training. |

<details><summary><code>batch_size</code> — extended notes</summary>

**Alias:** `per_device_train_batch_size (HF Trainer) | BSZ (fairseq)`

**Common values:** [4,8,16,32]

</details>

<details><summary><code>max_epoch</code> — extended notes</summary>

**Alias:** `num_train_epochs (HF Trainer)`

</details>

<details><summary><code>patience</code> — extended notes</summary>

**Alias:** `early_stopping_patience (HF EarlyStoppingCallback)`

</details>

### Optimization

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `learning_rate` | `float` | `0.00002` | 0.000005–0.0001 | optimization | 🔴 critical | Peak LR with Adam optimizer. Official training: 2e-5. Community HF finetuning typically uses 4e-5–5e-5 for smaller datasets. Lower LR reduces catastrophic forge |
| `lr_scheduler` | `str` | `"inverse_sqrt"` | inverse_sqrt \| linear \| cosine \| constant | optimization | 🟡 medium | LR schedule type. Official uses inverse square-root warmup decay. HuggingFace Trainer default is linear. Cosine is recommended for longer finetune runs. |
| `warmup_updates` | `int` | `500` | — | optimization | 🟡 medium | Steps of linear LR warmup before decay schedule begins. |
| `warmup_init_lr` | `float` | `1e-8` | — | optimization | 🟢 low | Starting LR at the very first warmup step. Prevents unstable initial gradient updates. Only in fairseq config — absent from most blog posts. |
| `weight_decay` | `float` | `0.0001` | 0–0.01 | optimization | 🟢 low | L2 regularisation coefficient on Adam optimizer. |
| `adam_beta1` | `float` | `0.9` | — | optimization | 🟢 low | Adam first moment decay rate. |
| `adam_beta2` | `float` | `0.999` | — | optimization | 🟢 low | Adam second moment decay rate. |
| `adam_epsilon` | `float` | `1e-8` | — | optimization | 🟢 low | Adam numerical stability epsilon term. |
| `update_freq` | `int` | `1` | 1–16 | optimization | 🟠 high | Gradient accumulation factor in fairseq. update_freq=4 with BSZ=2 = effective BSZ of 8. Maps to gradient_accumulation_steps in HuggingFace Trainer. |

<details><summary><code>lr_scheduler</code> — extended notes</summary>

**Alias:** `lr-scheduler (fairseq) | lr_scheduler_type (HF Trainer)`

</details>

<details><summary><code>warmup_updates</code> — extended notes</summary>

**Alias:** `warmup_steps (HF Trainer) | num_warmup_steps`

**Common values:** [300,500,1000]

</details>

<details><summary><code>warmup_init_lr</code> — extended notes</summary>


**⚠️ Note:** OFTEN MISSING from tutorials. Too-high value causes early divergence.

</details>

<details><summary><code>update_freq</code> — extended notes</summary>

**Alias:** `gradient_accumulation_steps (HF Trainer)`

**⚠️ Note:** NAMING DIFFERENCE causes confusion when porting fairseq configs to HF.

</details>

### Mixed Precision

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `fp16` | `bool` | `true` | — | mixed_precision | 🟠 high | FP16 AMP mixed precision training. Enabled via --fp16 in official fairseq script. Reduces VRAM ~50%, speeds training ~1.5–2× on Tensor Core GPUs. |

<details><summary><code>fp16</code> — extended notes</summary>

**Alias:** `fp16 (HF Trainer) | --fp16 (fairseq)`

</details>

### Augmentation

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `preprocess` | `str` | `"DA2"` | None \| DA1 \| DA2 | augmentation | 🟠 high | Data augmentation preset for training images. DA2 is stronger augmentation used in official IAM finetuning. Using no augmentation gives significantly worse CER. |

<details><summary><code>preprocess</code> — extended notes</summary>


**⚠️ Note:** Fairseq-specific parameter. HF training uses separate augmentation transforms.

</details>

### Performance

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `num_workers` | `int` | `8` | — | performance | 🟡 medium | DataLoader worker processes. |

### Reproducibility

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `seed` | `int` | `1111` | — | reproducibility | 🟢 low | Random seed. |

### Tokenizer

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `bpe` | `str` | `"gpt2"` | gpt2 \| sentencepiece | tokenizer | 🟡 medium | Byte-pair encoding method for the decoder vocabulary. gpt2 BPE is the default. Use sentencepiece with --sentencepiece-model for multilingual models. |

<details><summary><code>bpe</code> — extended notes</summary>

**Alias:** `tokenizer (HF Transformers)`

</details>

### Inference

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `max_length` | `int` | `512` | — | inference | 🟡 medium | Maximum token sequence length during autoregressive decoding. TrOCR processes line-cropped images; 64 tokens covers most text lines. Use 512 for full-document i |

<details><summary><code>max_length</code> — extended notes</summary>

**Alias:** `generation_max_length (HF Trainer) | max_new_tokens`

</details>

### Evaluation

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `predict_with_generate` | `bool` | `true` | — | evaluation | 🟡 medium | Must be True for CER/WER evaluation during training in HuggingFace Seq2SeqTrainer. When False, uses teacher-forcing logits for loss only. |

<details><summary><code>predict_with_generate</code> — extended notes</summary>

**Alias:** `predict_with_generate (HF Seq2SeqTrainingArguments)`

</details>

### Peft

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `lora_rank` | `int` | `—` | 4–64 | peft | 🟠 high | LoRA rank for parameter-efficient finetuning. Used in DLoRA-TrOCR: LoRA applied to decoder, DoRA applied to encoder. Reduces trainable params from 334M to <10M  |
| `use_dora_encoder` | `bool` | `—` | — | peft | 🟠 high | Apply DoRA (weight decomposition adapter) to ViT encoder instead of standard LoRA. Achieves comparable or better accuracy than full finetuning with drastically  |

<details><summary><code>lora_rank</code> — extended notes</summary>


**⚠️ Note:** PEFT approach — not in original TrOCR paper. From Chang et al. 2024.

</details>

<details><summary><code>use_dora_encoder</code> — extended notes</summary>


**⚠️ Note:** DLoRA-TrOCR paper: encoder uses DoRA, decoder uses LoRA.

</details>

---

## YOLO — You Only Look Once (Ultralytics YOLOv5/v8/v11/v26)

| Field | Value |
|-------|-------|
| Architecture | CSPDarknet backbone + FPN/PAN neck + detection head |
| Task | Object detection, instance segmentation, pose estimation, classification |
| Base model | `yolov8n.pt | yolov8s.pt | yolov8l.pt | yolo11n.pt` |
| Framework | Ultralytics |
| Config file | `ultralytics/cfg/default.yaml` |

### Image Resolution

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `imgsz` | `int` | `640` | 32–8192 | image_resolution | 🔴 critical | Square input resolution. Images are auto-resized and letterboxed to this size. Larger = better small object detection, more VRAM, slower training. Must match be |

<details><summary><code>imgsz</code> — extended notes</summary>


**⚠️ Note:** For high-res images (>2MP), use sliding window instead of single large imgsz.

**Common values:** [320,416,512,640,832,1024,1280]

</details>

### Training Core

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `batch` | `int | float` | `16` | — | training_core | 🟠 high | Images per training step. -1 = auto (fills ~60% GPU VRAM). Float 0.0–1.0 = VRAM fraction. Larger batches stabilise gradients but require more memory. OOM trigge |
| `epochs` | `int` | `100` | 1–1000 | training_core | 🟠 high | Number of full training passes. Default 100 for finetuning; 300 for training from scratch. Use patience for early stopping to prevent overfitting. |
| `patience` | `int` | `50` | 5–200 | training_core | 🟠 high | Early stopping: halt training if no mAP50-95 improvement for N epochs. Set lower (10–20) for fast finetuning; higher for uncertain convergence. |

### Optimization

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `lr0` | `float` | `—` | 0.00001–0.1 | optimization | 🔴 critical | Initial (peak) learning rate. SGD default 0.01; AdamW default ~0.001. Lower values recommended for finetuning to avoid overwriting pretrained features. |
| `lrf` | `float` | `0.01` | 0.001–0.1 | optimization | 🟡 medium | Final LR as a fraction of lr0. Controls cosine/linear decay endpoint: final_lr = lr0 × lrf. |
| `momentum` | `float` | `0.937` | 0.6–0.98 | optimization | 🟡 medium | SGD momentum / Adam beta1. Controls gradient history weighting. |
| `weight_decay` | `float` | `0.0005` | 0–0.01 | optimization | 🟡 medium | L2 regularisation. Helps prevent overfitting on small datasets. |
| `warmup_epochs` | `float` | `3` | 0–10 | optimization | 🟡 medium | Linear LR warmup from near-zero to lr0 over N epochs at training start. |
| `warmup_momentum` | `float` | `0.8` | 0–0.95 | optimization | 🟢 low | Starting SGD momentum value during warmup phase. |
| `warmup_bias_lr` | `float` | `0.1` | 0–0.2 | optimization | 🟢 low | Starting LR specifically for bias parameters during warmup. |
| `optimizer` | `str` | `"auto"` | auto \| SGD \| Adam \| AdamW \| NAdam \| RAdam \| RMSProp | optimization | 🟠 high | 'auto' selects AdamW for ≤10 warmup epochs, SGD otherwise. Explicit AdamW recommended for finetuning. SGD recommended for full training from scratch with 300+ e |
| `cos_lr` | `bool` | `false` | — | optimization | 🟡 medium | Use cosine annealing LR schedule (smoother decay). Recommended for longer runs. When False, uses linear decay. |

### Mixed Precision

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `amp` | `bool` | `true` | — | mixed_precision | 🟠 high | Automatic Mixed Precision (FP16/FP32). On by default. Reduces VRAM ~50%, speeds training. Disable (amp=False) if you see NaN losses in early training. OOM trigg |

<details><summary><code>amp</code> — extended notes</summary>

**Alias:** `half / fp16 in export/inference modes`

</details>

### Finetuning

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `freeze` | `int | list[int]` | `null` | — | finetuning | 🔴 critical | Freeze N backbone layers (int) or specific layer indices (list). freeze=10 freezes entire backbone, trains only detection head. Dramatically reduces training ti |

<details><summary><code>freeze</code> — extended notes</summary>


**⚠️ Note:** THE most impactful underdocumented parameter for domain finetuning.

**Common values:** [null,0,3,10]

</details>

### Augmentation

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `close_mosaic` | `int` | `10` | 0–50 | augmentation | 🟠 high | Disable mosaic augmentation for the last N epochs of training. Stabilises final learning phase and improves mAP convergence. Rarely mentioned in basic tutorials |
| `multi_scale` | `bool` | `false` | — | augmentation | 🟡 medium | Randomly vary imgsz by ±50% each batch during training. Improves multi-scale object detection robustness. Increases training time. |
| `mosaic` | `float` | `1` | 0–1 | augmentation | 🔴 critical | Probability of mosaic augmentation: combines 4 training images into one 2×2 grid. Massively improves detection of objects in varied contexts. Disabled for last  |
| `mixup` | `float` | `0` | 0–0.5 | augmentation | 🟡 medium | Probability of mixup: blends two images and their labels. Improves generalisation. Use 0.1–0.2 max; too high hurts precision. |
| `copy_paste` | `float` | `0` | 0–0.5 | augmentation | 🟠 high | Probability of copy-paste: copies object instances from one image to another. Very effective for rare-class augmentation in imbalanced datasets. |
| `hsv_h` | `float` | `0.015` | 0–0.1 | augmentation | 🟡 medium | Hue jitter magnitude. Helps with varying lighting and colour conditions. |
| `hsv_s` | `float` | `0.7` | 0–1 | augmentation | 🟡 medium | Saturation jitter magnitude. |
| `hsv_v` | `float` | `0.4` | 0–1 | augmentation | 🟡 medium | Brightness/value jitter magnitude. |
| `fliplr` | `float` | `0.5` | 0–1 | augmentation | 🟡 medium | Horizontal flip probability. Set 0 if left-right orientation matters (e.g. text, asymmetric objects). |
| `flipud` | `float` | `0` | 0–1 | augmentation | 🟢 low | Vertical flip probability. Set 0 if vertical orientation matters (most use cases). |
| `translate` | `float` | `0.1` | 0–0.5 | augmentation | 🟡 medium | Random translation as fraction of image size. |
| `scale` | `float` | `0.5` | 0–0.9 | augmentation | 🟠 high | Random scale (zoom) augmentation range. Critical for detecting objects at varying distances. |
| `shear` | `float` | `0` | 0–10 | augmentation | 🟢 low | Shear transformation in degrees. |
| `perspective` | `float` | `0` | 0–0.001 | augmentation | 🟢 low | Random perspective distortion coefficient. |
| `degrees` | `float` | `0` | 0–45 | augmentation | 🟡 medium | Random rotation range in degrees. Set 0 if object orientation matters. |
| `bgr` | `float` | `0` | 0–1 | augmentation | 🟢 low | Probability of randomly swapping BGR channel order. Guards against incorrect channel ordering in deployment. |

<details><summary><code>close_mosaic</code> — extended notes</summary>


**⚠️ Note:** CRITICALLY IMPORTANT for final accuracy. Do not set to 0.

</details>

### Data

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `rect` | `bool` | `false` | — | data | 🟡 medium | Rectangular training: batches images by similar aspect ratio to reduce letterbox padding waste. Incompatible with DataLoader shuffle — silently disables shuffle |
| `fraction` | `float` | `1` | 0.1–1 | data | 🟠 high | Fraction of dataset to use for training. Reduce (e.g. 0.1) for fast hyperparameter sweep experiments. Proportionally reduces training time. |

<details><summary><code>rect</code> — extended notes</summary>


**⚠️ Note:** WARNING: silently disables shuffle. Can bias training if unaware.

</details>

<details><summary><code>fraction</code> — extended notes</summary>


**⚠️ Note:** Underused but very practical for HPO.

</details>

### Performance

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `cache` | `bool | str` | `false` | false \| ram \| disk | performance | 🟠 high | 'ram' caches entire dataset in memory for maximum I/O speed. 'disk' caches preprocessed images. Dramatically speeds up training when dataset fits in RAM. |
| `workers` | `int` | `8` | 0–32 | performance | 🟡 medium | DataLoader worker threads per GPU rank. |

### Reproducibility

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `seed` | `int` | `0` | — | reproducibility | 🟢 low | Random seed. |
| `deterministic` | `bool` | `true` | — | reproducibility | 🟢 low | Forces CUDA deterministic algorithms. Slight training speed penalty. |

### Regularisation

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `dropout` | `float` | `0` | 0–0.5 | regularisation | 🟡 medium | Dropout probability in classification head. Use 0.1–0.2 for small datasets to prevent overfitting. |

### Task

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `single_cls` | `bool` | `false` | — | task | 🟡 medium | Treat all classes as a single class. Useful for binary detection (object/background) tasks. |

### Checkpointing

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `save_period` | `int` | `-1` | — | checkpointing | 🟢 low | Save checkpoint every N epochs. -1 = only save best.pt and last.pt. |

### Loss

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `box` | `float` | `7.5` | 1–20 | loss | 🟠 high | Bounding box regression loss weight (GIoU/CIoU loss coefficient). Increase for tight localisation tasks. |
| `cls` | `float` | `0.5` | 0.1–5 | loss | 🟠 high | Classification loss weight. Increase for many-class tasks where classification accuracy is critical. |
| `dfl` | `float` | `1.5` | 0.5–5 | loss | 🟡 medium | Distribution Focal Loss weight for fine-grained bounding box regression. |

### Inference

| Parameter | Type | Default | Range/Options | Category | Impact | Description |
|-----------|------|---------|---------------|----------|--------|-------------|
| `iou` | `float` | `0.7` | 0.1–0.95 | inference | 🟡 medium | IoU threshold for Non-Maximum Suppression (NMS) during inference. Higher = fewer boxes, allows more overlap. |
| `conf` | `float` | `0.25` | 0.01–0.99 | inference | 🟡 medium | Minimum confidence threshold for predictions. Lower = more detections, more false positives. |
| `max_det` | `int` | `300` | 1–10000 | inference | 🟢 low | Maximum number of detections per image after NMS. |

---

## Cross-Model Comparison

| Topic | DONUT | TrOCR | YOLO |
|-------|-------|-------|------|
| **mixed precision** | precision=16 (PyTorch Lightning flag) | --fp16 (fairseq) | fp16=True (HF Trainer) | amp=True (Ultralytics default) |
| **gradient accumulation** | gradient_accumulation_steps (HF Trainer) | update_freq (fairseq) | gradient_accumulation_steps (HF Trainer) | Not directly exposed; use smaller batch + amp instead |
| **input resolution** | Configurable [width, height]. Default finetune: [1280, 960] | FIXED at 384×384. Not configurable without re-training encoder. | Configurable imgsz. Default: 640. Must match train/inference. |
| **layer freezing for finetuning** | No native freeze parameter; freeze manually via requires_grad=False | Freeze encoder layers manually via requires_grad=False | freeze=10 (freeze backbone). Native, well-documented. |
| **upsampling in architecture** | N/A — autoregressive token generation, no spatial upsampling in output | N/A — autoregressive token generation, no spatial upsampling in output | nn.Upsample(scale_factor=2, mode='nearest') in FPN/PAN neck. Not a training parameter — architectural constant. |

> **mixed precision:** All three support FP16 AMP. All are safe to enable on Tensor Core GPUs (NVIDIA V100, A100, RTX series). Disable if NaN losses appear.

> **gradient accumulation:** TrOCR fairseq uses 'update_freq' not 'gradient_accumulation_steps' — common porting confusion.

> **input resolution:** TrOCR's fixed resolution is a known limitation for wide-aspect text strips.

> **layer freezing for finetuning:** YOLO freeze parameter is the most ergonomic. Others require manual intervention.

> **upsampling in architecture:** Upsampling in YOLO is architectural (feature pyramid), not a tunable hyperparameter.


---

## ⚠️ Commonly Underdocumented Parameters

These parameters have outsized impact but are missing from most tutorials.

| Parameter | Model | Severity | Why It Matters |
|-----------|-------|----------|----------------|
| `swin_window_size` | DONUT | 🔴 critical | Changing from pretrain value forces full weight re-init — effectively trains from scratch. Never mentioned in finetuning tutorials. |
| `align_long_axis` | DONUT | 🟡 medium | Rotating portrait documents can significantly improve text extraction quality but is off by default with no explanation. |
| `sort_json_key` | DONUT | 🟠 high | Must be False for CORD-v2. Setting True silently corrupts token ordering in preprocessed datasets. |
| `max_steps vs max_epochs CPU trap` | DONUT | 🟠 high | Setting both > 0 causes division-by-zero on CPU. Undocumented trap in PyTorch Lightning training script. |
| `warmup_init_lr` | TROCR | 🟡 medium | Starting LR at warmup step 0. Too high causes early divergence. Present in fairseq script but absent from all blog posts. |
| `update_freq vs gradient_accumulation_steps` | TROCR | 🟠 high | Different names in fairseq vs HuggingFace. Causes silent config errors when porting. |
| `preprocess DA1/DA2` | TROCR | 🟠 high | Official augmentation pipeline is DA2 for IAM. Omitting gives significantly worse CER. Not mentioned in HF tutorials. |
| `freeze` | YOLO | 🔴 critical | Most impactful parameter for domain finetuning on small datasets. Freezing backbone prevents overfitting and reduces training time dramatically. Underemphasised in tutorials. |
| `close_mosaic` | YOLO | 🟠 high | Disabling mosaic for final 10 epochs is crucial for mAP convergence. Almost never mentioned in basic tutorials. |
| `rect shuffle conflict` | YOLO | 🟡 medium | rect=True silently disables DataLoader shuffle. Can cause systematic batch bias without user awareness. |
| `amp OOM auto-retry` | YOLO | 🟡 medium | YOLO silently halves batch size on CUDA OOM. Can cause inconsistent effective batch sizes across experiment runs. |
| `val_check_interval (float)` | DONUT | 🟢 low | Can be a float (0.2 = validate 5× per epoch) — crucial for fast-converging small datasets. Documented but rarely used. |
| `fraction` | YOLO | 🟢 low | Subset training for fast HPO sweeps. Essentially invisible in documentation. |
| `lora_rank / use_dora_encoder` | TROCR | 🟡 medium | PEFT approach reducing trainable params from 334M to <10M. Not in original TrOCR docs; from 2024 DLoRA-TrOCR paper. |

---

## Recommended Starter Configs

### DONUT — Single GPU Finetune
```yaml
input_size: [1280, 960]
train_batch_sizes: [1]
max_length: 768
lr: 3.0e-5
warmup_steps: 300
max_epochs: 30
gradient_clip_val: 1.0
align_long_axis: false
precision: 16
num_workers: 8
sort_json_key: false
```
### TrOCR — HuggingFace Seq2SeqTrainer
```yaml
per_device_train_batch_size: 8
num_train_epochs: 15
learning_rate: 4.0e-5
weight_decay: 0.01
fp16: true
predict_with_generate: true
generation_max_length: 64
warmup_steps: 500
gradient_accumulation_steps: 2
```
### YOLO — Domain Finetune
```yaml
imgsz: 640
batch: 16
epochs: 100
lr0: 0.001
optimizer: AdamW
freeze: 10
close_mosaic: 10
mosaic: 1.0
amp: true
patience: 20
cos_lr: true
weight_decay: 0.0005
```

---

## JSON Schema Reference

The companion `finetuning_params.json` file has the following structure for programmatic consumption:

```
finetuning_params.json
├── meta                          # title, version, sources
├── models
│   ├── donut
│   │   ├── name, architecture, task, base_model, framework
│   │   └── parameters[]
│   │       ├── name (string)
│   │       ├── type (string — Python type annotation)
│   │       ├── default_finetune | default_value | default_official
│   │       ├── range [min, max]  OR  options [...]
│   │       ├── unit (optional)
│   │       ├── category (string)
│   │       ├── required (bool)
│   │       ├── impact: "critical"|"high"|"medium"|"low"
│   │       ├── description (string)
│   │       ├── notes (optional — warnings)
│   │       ├── alias (optional — alternate names in other frameworks)
│   │       └── common_values (optional)
│   ├── trocr  (same structure)
│   └── yolo   (same structure)
├── cross_model_notes[]
│   ├── topic
│   ├── donut | trocr | yolo (string — how each model handles the concept)
│   └── note
└── commonly_underdocumented[]
    ├── param, model, severity, reason
```

*Generated 2026-03-08 | v1.0*