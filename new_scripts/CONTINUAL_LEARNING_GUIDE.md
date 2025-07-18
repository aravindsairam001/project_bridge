# Continual Learning for Bridge Defect Detection

This guide explains how to use the continual learning system to expand your model's capabilities with new data while preserving previous knowledge.

## 📋 Overview

The continual learning system uses several techniques to prevent catastrophic forgetting:

1. **Knowledge Distillation**: The original model (teacher) guides the updated model (student)
2. **Elastic Weight Consolidation (EWC)**: Protects important weights from previous tasks
3. **Experience Replay**: Includes samples from the original dataset during training
4. **Conservative Learning**: Lower learning rates and careful regularization

## 🚀 Quick Start

### Step 1: Prepare Your New Dataset

You have two options for preparing new data:

#### Option A: From JSON Annotations
```bash
cd new_scripts
python3 prepare_continual_dataset.py --mode json --source /path/to/new/data --target /path/to/prepared/data
```

#### Option B: From Existing Masks
```bash
cd new_scripts
python3 prepare_continual_dataset.py --mode masks --source /path/to/new/data --target /path/to/prepared/data
```

#### Interactive Mode
```bash
cd new_scripts
python3 prepare_continual_dataset.py
```

### Step 2: Validate Your Dataset
```bash
python3 prepare_continual_dataset.py --mode validate --source /path/to/prepared/data
```

### Step 3: Run Continual Learning
```bash
python3 continual_learning.py
```

You'll be prompted for:
- Path to your pretrained model (.pth file)
- Path to new dataset directory (prepared in Step 1)
- Path to original dataset directory (for replay buffer)

## 📁 Expected Directory Structure

### New Dataset Structure (after preparation)
```
new_dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/
    ├── image1.jpg.png
    ├── image2.jpg.png
    └── ...
```

### Original Dataset Structure (for replay buffer)
```
original_dataset/
└── train/
    ├── images/
    │   ├── dacl10k_v2_train_XXXX.jpg
    │   └── ...
    └── masks/
        ├── dacl10k_v2_train_XXXX.jpg.png
        └── ...
```

## ⚙️ Configuration

Edit the following parameters in `continual_learning.py`:

```python
# Model Architecture (MUST match your pretrained model)
ARCHITECTURE = 'unetplusplus'  # Options: 'unetplusplus', 'fpn', 'linknet', 'pspnet', 'deeplabv3plus'
ENCODER_NAME = 'efficientnet-b5'  # Must match your model's encoder

# Training Parameters
BATCH_SIZE = 2              # Small batch for memory efficiency
NUM_EPOCHS = 5              # Conservative epoch count
INITIAL_LR = 1e-5          # Low learning rate for fine-tuning

# Continual Learning Parameters
KNOWLEDGE_DISTILL_ALPHA = 0.7  # Balance between old and new knowledge
TEMPERATURE = 4                 # Softmax temperature for distillation
EWC_LAMBDA = 400               # EWC regularization strength
REPLAY_BUFFER_SIZE = 200       # Number of old samples to replay
```

## 🎯 Class Mapping

The system uses the same 11-class mapping as your original training:

```python
CLASSES = {
    0: "Background",
    1: "Rust", 
    2: "ACrack",           # Alligator Crack
    3: "WConccor",         # Washouts/Concrete Corrosion
    4: "Cavity",
    5: "Hollowareas", 
    6: "Spalling",
    7: "Rockpocket",
    8: "ExposedRebars",
    9: "Crack",
    10: "Weathering",
    11: "Efflorescence"
}
```

## 📊 Monitoring Training

The system provides detailed logging:

- **Total Loss**: Combined continual learning loss
- **KD Loss**: Knowledge distillation loss
- **EWC Loss**: Elastic weight consolidation penalty
- **Progress Bars**: Real-time training progress

Training history is saved to JSON files for analysis.

## 🔧 Troubleshooting

### Common Issues

1. **Model Architecture Mismatch**
   ```
   Error: Size mismatch for encoder.conv1.weight
   ```
   **Solution**: Ensure `ARCHITECTURE` and `ENCODER_NAME` match your pretrained model exactly.

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce `BATCH_SIZE` to 1 or use smaller image size.

3. **No New Samples Found**
   ```
   Found 0 samples (0 new + 0 replay)
   ```
   **Solution**: Check dataset structure and run validation first.

### Memory Optimization

If you encounter memory issues:

1. Reduce batch size:
   ```python
   BATCH_SIZE = 1
   ```

2. Reduce image size:
   ```python
   IMAGE_SIZE = 256  # Instead of 384
   ```

3. Reduce replay buffer:
   ```python
   REPLAY_BUFFER_SIZE = 50  # Instead of 200
   ```

4. Use gradient accumulation:
   ```python
   # Manually implement gradient accumulation
   if batch_idx % 4 == 0:  # Accumulate 4 batches
       optimizer.step()
       optimizer.zero_grad()
   ```

## 📈 Best Practices

### Dataset Quality
- Ensure new data is high quality and properly annotated
- Validate dataset before training
- Have at least 50-100 new samples for meaningful improvement

### Training Strategy
- Start with conservative parameters (low LR, few epochs)
- Monitor training carefully - stop if validation loss increases significantly
- Keep original model as backup

### Model Evaluation
- Test on both old and new data to ensure no catastrophic forgetting
- Compare performance with original model on validation sets
- Use the Streamlit app to visually inspect results

## 🎯 Expected Results

With proper setup, you should see:

1. **Preserved Performance**: No significant degradation on original test data
2. **Improved Capability**: Better performance on new data types
3. **Stable Training**: Smooth loss curves without dramatic spikes
4. **Memory Efficiency**: Training within available GPU memory

## 📚 Advanced Usage

### Custom Loss Functions
You can modify the loss combination in `continual_train_epoch()`:

```python
# Add focal loss for hard examples
focal_loss = FocalLoss(alpha=1, gamma=2)
base_loss = ce + dice + focal_loss(student_outputs, masks)
```

### Crack-Focused Training
For crack-only datasets, consider these optimizations:

```python
# In continual_learning.py, adjust these parameters:
KNOWLEDGE_DISTILL_ALPHA = 0.8  # Higher emphasis on preserving old knowledge
TEMPERATURE = 3                 # Lower temperature for sharper predictions
EWC_LAMBDA = 500               # Stronger protection for other classes
NUM_EPOCHS = 8                 # More epochs for crack specialization

# Add class weights to focus on crack learning:
class_weights = torch.ones(12)  # 12 classes (0-11)
class_weights[9] = 2.0         # Double weight for crack class
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Different Replay Strategies
Implement smart replay buffer selection:

```python
# Select diverse samples instead of random
replay_samples = select_diverse_samples(original_dataset, num_samples)
```

### Progressive Learning Rates
Use different learning rates for different parts:

```python
# Lower LR for encoder (pretrained features)
optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': 1e-6},
    {'params': model.decoder.parameters(), 'lr': 1e-5}
])
```

## 🆘 Support

For issues or questions:

1. Check the logs in `continual_learning.log`
2. Validate your dataset structure
3. Ensure model architecture matches
4. Start with minimal parameters and gradually increase

Remember: Continual learning is an iterative process. Start small, validate results, and gradually expand your model's capabilities!
