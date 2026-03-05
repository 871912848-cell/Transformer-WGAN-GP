## Transformer-WGAN-GP Overview (Paired Augmentation for Spectra–Property Data)

Transformer-WGAN-GP is a generative data augmentation framework designed for spectroscopy under limited-sample settings. It uses the Wasserstein adversarial objective with gradient penalty (WGAN-GP) and implements both the generator and the discriminator (critic) as stacked Transformer encoders to model long-range dependencies in spectral sequences. This design stabilizes training, mitigates mode collapse, and produces samples that better match the real data distribution.

### Key idea: joint generation of “spectrum + label” to ensure paired consistency
Unlike approaches that generate spectra first and then assign labels afterward, this framework adopts a joint-generation strategy:
- The generator outputs a spectral sequence together with its corresponding label.
- This maximizes feature–label coupling, allowing augmented data to be directly used by downstream supervised regression models.

### Architecture (tokenized joint representation)
- The output is organized as a token sequence: `L_total = L + 1`
  - `L` tokens correspond to spectral bands
  - `1` token corresponds to target property
- Generator
  - Input: latent noise vector `z` (latent dim = 64)
  - Linearly expanded into a `L_total × d_model` token representation (d_model = 96)
  - Multi-head self-attention (4 heads) with 3 Transformer blocks
  - Token-wise linear projection followed by `tanh`, producing a normalized paired sample in `[-1, 1]`
- Discriminator/Critic
  - 2 Transformer blocks
  - Global average pooling + linear output head to produce a scalar critic score

### Training setup (stable adversarial learning)
- Adam optimizer with learning rate `1e-4`, β = (0.5, 0.9)
- Critic update schedule: for each generator update, update the critic 5 times (n_critic = 5)
- Gradient penalty in WGAN-GP is used to improve convergence stability and reduce training oscillations

### Augmentation and evaluation (to prevent information leakage)
- Synthetic samples are added only to the training partition and are never used in the prediction set or external validation set
- Multiple augmentation ratios are supported (relative to the original training set)
- Generation quality and distribution alignment can be tracked using
  - Point-wise agreement: MSE
  - Spectral shape similarity: SAM
  - Distribution alignment: MMD
  - Global structural consistency: SVD
  - Visualization: t-SNE

### Interpretability (optional)
Visualizing multi-head attention weights in the Transformer generator can help reveal how dependencies across different wavelength regions are modeled, supporting interpretation of the generation mechanism and spectral characteristics.

---

> In our single-kernel wheat protein prediction study, this framework was used to expand calibration samples and was combined with attention-enhanced deep regression models. If your task also suffers from limited labeled samples and high-dimensional spectra, Transformer-WGAN-GP can serve as a reusable paired augmentation module.
