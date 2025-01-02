# Monet Style Transfer CycleGAN | 莫奈风格转换 CycleGAN

[English](#english) | [中文](#chinese)

<a name="english"></a>
# Monet Style Transfer CycleGAN

A CycleGAN-based project for bidirectional style transfer between photographs and Monet paintings:
- Convert real photos to Monet painting style
- Convert Monet painting style to realistic photo style

## Features

- Bidirectional style transfer
- Improved FID evaluation method
- Optimized training strategy
- Experiment tracking with Wandb
- Mixed precision training
- Automatic learning rate adjustment

## Requirements

```bash
torch>=1.7.0
torchvision>=0.8.0
wandb
numpy
pillow
tqdm
scipy
pytorch-fid
```

## Project Structure

```
.
├── data/
│   ├── trainA/    # Training set photos
│   ├── trainB/    # Training set Monet paintings
│   ├── testA/     # Test set photos
│   └── testB/     # Test set Monet paintings
├── models/
│   ├── generator.py       # Generator model
│   └── discriminator.py   # Discriminator model
├── utils/
│   ├── dataset.py        # Dataset loading
│   └── fid_score.py      # FID score calculation
├── train.py              # Training script
├── recalculate_fid.py    # FID recalculation script
└── README.md
```

## Technical Details

### Model Architecture

#### Generator
- Based on ResNet architecture
- 9 residual blocks for feature extraction and transformation
- Instance normalization to preserve style information
- Reflection padding for edge consistency
- Tanh activation for output range [-1,1]

#### Discriminator
- PatchGAN structure focusing on local features
- Instance normalization
- LeakyReLU activation (slope 0.2)
- 70x70 receptive field

### Loss Functions

1. GAN Loss
   - Using MSE loss
   - Label smoothing for real and fake labels
   - Enhances generation authenticity

2. Cycle Loss
   - L1 loss
   - Weight: 20.0
   - Ensures image transformation reversibility
   - Maintains content consistency

3. Identity Loss
   - L1 loss
   - Weight: 10.0
   - Helps preserve color and composition
   - Prevents unnecessary style transfer

### Evaluation Metrics

#### FID Score (Fréchet Inception Distance)
- Uses pretrained Inception-v3 for feature extraction
- Measures feature distribution distance
- Evaluates both domains separately:
  - Photo domain FID: Test set photos (774) vs generated photos
  - Monet domain FID: Test set Monet paintings (131) vs generated paintings
- Calculated every 10 epochs
- Lower values indicate better quality
- Uses independent test set for objective evaluation

## Training Strategy

### Base Configuration
```python
config = {
    'epochs': 200,
    'batch_size': 8,
    'lr': 0.0001,
    'b1': 0.5,
    'b2': 0.999,
    'lambda_cycle': 15.0,
    'lambda_identity': 7.5,
    'gradient_accumulation_steps': 4
}
```

### Optimization Techniques
1. Learning Rate Schedule:
   - Warmup phase (5 epochs)
   - Constant learning rate phase
   - Linear decay (starting from epoch 100)

2. Training Stability:
   - Gradient accumulation (updates every 4 steps)
   - Mixed precision training (reduces memory usage)
   - Larger batch size (8) for stability

3. Checkpoint Saving:
   - Every 5 epochs
   - Saves complete training state
   - Supports training resumption

## Experimental Results

This section integrates results from both **our custom training** and **CycleGAN source code training (125 epochs)** for comparison.

### 1. Our Training Process and Results

#### 1.1 Training Curves
<div align="center">
  <div class="training-visualization">
    <img src="assets/200epoch_1.jpg" alt="Training Curves 1" width="80%" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <p><i>📈 First 200 epochs: Learning rate, FID scores, and generator loss curves</i></p>
    <br>
    <img src="assets/200epoch_2.jpg" alt="Training Curves 2" width="80%" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <p><i>📊 First 200 epochs: Discriminator loss, cycle consistency loss, and identity loss curves</i></p>
  </div>
</div>

#### 1.2 Test Set FID Evaluation
<div align="center">
  <img src="assets/test_fid_scores.png" alt="Test Set FID Scores" width="80%">
  <p><i>📊 Test Set FID Score Trends</i></p>
  <br>
  <table>
    <tr>
      <th>Metric</th>
      <th>Value</th>
    </tr>
    <tr>
      <td>Photo Domain FID</td>
      <td>121.49</td>
    </tr>
    <tr>
      <td>Monet Domain FID</td>
      <td>131.68</td>
    </tr>
    <tr>
      <td>Average FID</td>
      <td>126.59</td>
    </tr>
  </table>
</div>

#### 1.3 Training Metrics Analysis

- **Learning Rate Changes**
  - Initial learning rate: 0.0001
  - Constant for first 100 epochs
  - Linear decay from 100-200 epochs
  - New strategy for 200-280 epochs
  - Separate adjustments for discriminator and generator

- **FID Score Changes**
  - Training set: ~46.41 at 200 epochs; increased to 51.08 at 280 epochs
  - Test set: Photo domain ~121.49; Monet domain ~131.68; Average 126.59

- **Loss Function Changes**
  - Generator loss decreased to ~0.45
  - Discriminator loss stabilized at 0.02-0.04
  - Cycle consistency loss reduced to 0.05-0.06
  - Identity loss around 0.05

#### 1.4 Optimization Strategy Adjustments (200-280 epochs)

- **Learning Rate Optimization**:
  - Generator: 0.00008 → cosine decay → minimum 0.000002
  - Discriminator: 0.00002 → cosine decay → minimum 0.0000005

- **Loss Weights**:
  - Cycle Loss: 20.0
  - Identity Loss: 10.0

- **Training Stability**:
  - Label smoothing: 0.05
  - Gradient accumulation steps: 2
  - Warmup phase: 2 epochs

#### 1.5 Optimization Effect Comparison
<div align="center">
  <div class="comparison-results">
    <img src="assets/280与200epoch生成莫奈图片.jpg" 
         alt="280 vs 200 Epochs Monet Generation" 
         width="90%"
         style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <p><i>🎨 200 vs 280 Epochs: Monet Style Generation Comparison</i></p>
    <br>
    <img src="assets/280与200epoch生成照片对比-1.jpg" 
         alt="280 vs 200 Epochs Photo Generation" 
         width="90%"
         style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <p><i>📸 200 vs 280 Epochs: Real Photo Style Generation Comparison</i></p>
  </div>
</div>

#### 1.6 Conclusions

1. **Best Model Configuration (200 epochs)**
   - Optimal average FID (~46.41)
   - Balanced loss values and image quality

2. **Over-optimization Effects**
   - FID scores increased in later stages
   - Possible overfitting or model oscillation

3. **Recommendations**
   - Use 200-epoch model as final model
   - Avoid over-optimization
   - Maintain balanced training parameters

### 2. CycleGAN Source Code Results (125 epochs)

Results from the **official CycleGAN source code** at 125 epochs, with domain definitions adjusted to match our project (A: photos, B: Monet paintings).

#### 2.1 Training Process Visualization
<div align="center">
  <div class="source-code-visualization">
    <img src="assets/源码125epoch_1.jpg" alt="Source Code Training Curves 1" width="80%" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <p><i>📈 Source Code 125 epochs: Generator and discriminator loss curves</i></p>
    <br>
    <img src="assets/源码125epoch_2.jpg" alt="Source Code Training Curves 2" width="80%" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <p><i>📊 Source Code 125 epochs: Cycle consistency and identity loss curves</i></p>
  </div>
</div>

#### 2.2 FID Analysis
<div align="center">
  <img src="assets/源码fid.jpg" alt="Source Code FID Scores" width="80%" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
  <p><i>📊 Source Code 125 epochs: FID score trends</i></p>
</div>

**Current FID Values (125 epochs)**
- **A->B** (Photo -> Monet): **132.02**
- **B->A** (Monet -> Photo): **127.59**

**FID Trends**
- **A->B direction**: Decreased from ~142 to 132.02, smaller improvement
- **B->A direction**: Decreased from ~170 to 127.59, more significant improvement
- **Overall trend**: FID values decreasing but at a slower rate

#### 2.3 Loss Value Analysis (Step 693300)

- **Generator Loss**
  - G_A (Photo->Monet): 0.8700
  - G_B (Monet->Photo): 0.9152
  - Both generator losses similar, indicating balanced training

- **Cycle Consistency Loss**
  - cycle_A (Photo->Monet->Photo): **0.8396**
  - cycle_B (Monet->Photo->Monet): **0.5270**
  - Both cycle losses at low levels, A direction slightly higher

- **Discriminator Loss**
  - D_A: 0.0514 (Photo->Monet discriminator)
  - D_B: 0.2747 (Monet->Photo discriminator)
  - Discriminator losses within reasonable range, no mode collapse

- **Identity Mapping Loss**
  - idt_A (Photo domain): 0.2105
  - idt_B (Monet domain): 0.2637
  - Low identity losses indicate good style consistency

#### 2.4 Training Progress Assessment

1. All loss values show stable decrease
2. Cycle loss reduced to 0.5~0.8 range, good reversibility
3. FID value decrease rate slowing, possible bottleneck

**Areas of Concern**
1. Slowing FID improvement
2. Slower improvement in A->B direction
3. Higher cycle_A loss indicates different difficulty levels

#### 2.5 Recommendations

1. **Continue to 150 epochs** to potentially break through bottleneck
2. **Increase A->B supervision**
3. Focus on visual quality with quantitative monitoring
4. Save FID evaluations every 5 epochs

### 3. Performance Comparison Analysis

#### 3.1 FID Score Comparison

| Implementation | Photo FID | Monet FID | Average FID | Epochs |
|---------------|-----------|------------|-------------|---------|
| Our Project   | 121.49    | 131.68    | 126.59      | 200    |
| Source Code   | 127.59    | 132.02    | 129.81      | 125    |

#### 3.2 Loss Function Comparison

| Loss Type | Our Project (200 epoch) | Source Code (125 epoch) | Analysis |
|-----------|------------------------|-------------------------|-----------|
| Generator Loss | 0.45 | 0.89 | Our project shows lower generator loss, potentially better generation |
| Discriminator Loss | 0.02-0.04 | 0.05-0.27 | Our project has more stable discriminator, smaller fluctuation |
| Cycle Consistency Loss | 0.05-0.06 | 0.53-0.84 | Our project shows significantly better cycle consistency |
| Identity Loss | 0.05 | 0.21-0.26 | Our project maintains better identity preservation |

#### 3.3 Key Improvements Analysis

1. **Training Stability**
   - Our project: Better stability through gradient accumulation and label smoothing
   - Source code: Larger fluctuations, generally higher loss values

2. **FID Performance**
   - Our project: Better results after more epochs (200)
   - Source code: Stabilizing at 125 epochs but room for improvement

3. **Loss Control**
   - Our project: Significantly lower loss values across all metrics
   - Source code: Higher but reasonable loss values

4. **Optimization Strategy**
   - Our project:
     * Dynamic learning rate adjustment
     * Gradient accumulation mechanism
     * Label smoothing technique
   - Source code:
     * Fixed learning rate
     * Basic optimizer settings
     * Simple training process

#### 3.4 Overall Assessment

1. **Advantages**
   - Lower loss values
   - More stable training process
   - Better cycle consistency
   - Stronger identity preservation

2. **Areas for Improvement**
   - FID scores can be further improved
   - Longer training time
   - Higher computational resource requirements

3. **Suggestions**
   - Combine advantages of both implementations
   - Further optimize training efficiency
   - Explore more training stability techniques

## Usage Instructions

1. Prepare data:
```bash
# Place data in the data directory
data/
  ├── trainA/  # Training set photos
  ├── trainB/  # Training set Monet paintings
  ├── testA/   # Test set photos
  └── testB/   # Test set Monet paintings
```

2. Train model:
```bash
python train.py
```

3. Recalculate FID (if needed):
```bash
python recalculate_fid.py
```

## Notes

1. GPU Memory Usage:
   - Small batch size (2) for FID calculation
   - Regular GPU cache clearing
   - Mixed precision training for memory efficiency

2. Training Stability:
   - Monitor loss value spikes
   - Watch generator-discriminator balance
   - Adjust learning rate as needed

3. Experiment Monitoring:
   - Track training process with Wandb
   - Regular image quality checks
   - Monitor FID score trends

## Future Improvements

1. Model Architecture:
   - Explore attention mechanisms
   - Try different normalization methods
   - Optimize network depth and width

2. Training Strategy:
   - Implement dynamic learning rate adjustment
   - Research new loss function combinations
   - Add data augmentation methods

3. Evaluation System:
   - Introduce human evaluation mechanism
   - Add more quantitative metrics (SSIM, LPIPS, etc.)
   - Develop automated testing process

## Acknowledgments

This project is based on the CycleGAN paper:
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)

## License

MIT License

---

<a name="chinese"></a>
[原中文版本保持不变，从这里开始]
// ... existing Chinese content ... 