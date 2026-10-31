"""Differentiable augmentation of discriminator inputs (DiffAugment) and instance noise.

DiffAugment: Zhao, Liu, Lin, Zhu & Han, "Differentiable Augmentation for
Data-Efficient GAN Training", NeurIPS 2020. The SAME kind of random transform is
applied to real and generated images, every time they enter D (in the D step and
in the G step). The transforms are differentiable, so G still receives gradients.

Instance noise: Sonderby et al. 2017; Arjovsky & Bottou 2017. Gaussian noise added
to both real and generated D inputs, annealed to zero, so D cannot separate them
by exact pixel values (e.g. real MNIST backgrounds are exactly -1).

Inputs/outputs are flattened 784-vectors in [-1, 1] (the fixed MLP's format);
-1 is the MNIST background, used for padding and cutout instead of 0.
"""
import torch

SIDE = 28
BACKGROUND = -1.0


def translation(img, ratio=0.125):
    """Shift each image by up to ratio*28 (= 4) pixels in x and y, padding with background."""
    n = img.shape[0]
    shift = int(SIDE * ratio + 0.5)
    padded = torch.nn.functional.pad(img, (shift, shift, shift, shift), value=BACKGROUND)
    dx = torch.randint(-shift, shift + 1, (n,), device=img.device)
    dy = torch.randint(-shift, shift + 1, (n,), device=img.device)
    rows = (torch.arange(SIDE, device=img.device)[None, :] + shift + dy[:, None])   # n x 28
    cols = (torch.arange(SIDE, device=img.device)[None, :] + shift + dx[:, None])   # n x 28
    batch = torch.arange(n, device=img.device)[:, None, None]
    return padded[:, 0][batch, rows[:, :, None], cols[:, None, :]].unsqueeze(1)          # n x 1 x 28 x 28


def cutout(img, ratio=0.5):
    """Set a random ratio*28 (= 14) pixel square of each image to background."""
    n = img.shape[0]
    size = int(SIDE * ratio + 0.5)
    cy = torch.randint(0, SIDE + (1 - size % 2), (n, 1, 1), device=img.device)
    cx = torch.randint(0, SIDE + (1 - size % 2), (n, 1, 1), device=img.device)
    grid = torch.arange(SIDE, device=img.device)
    inside_y = (grid[None, :, None] >= cy - size // 2) & (grid[None, :, None] < cy - size // 2 + size)
    inside_x = (grid[None, None, :] >= cx - size // 2) & (grid[None, None, :] < cx - size // 2 + size)
    mask = (inside_y & inside_x)[:, None]                                            # n x 1 x 28 x 28
    return torch.where(mask, torch.full_like(img, BACKGROUND), img)


def brightness_contrast(img):
    """Official DiffAugment 'color' minus saturation (meaningless for grayscale)."""
    n = img.shape[0]
    img = img + (torch.rand(n, 1, 1, 1, device=img.device) - 0.5)
    mean = img.mean(dim=(1, 2, 3), keepdim=True)
    return (img - mean) * (torch.rand(n, 1, 1, 1, device=img.device) + 0.5) + mean


TRANSFORMS = {"translation": translation, "cutout": cutout, "color": brightness_contrast}


def diff_augment(x, policy=(), noise_std=0.0, p=1.0):
    """Apply the listed transforms (in order), then optional Gaussian instance noise.

    p < 1 (adaptive augmentation, Karras et al. 2020): each transform is applied to
    each image independently with probability p, otherwise the image passes unchanged.
    """
    if not policy and noise_std == 0:
        return x
    img = x.view(-1, 1, SIDE, SIDE)
    for name in policy:
        augmented = TRANSFORMS[name](img)
        if p >= 1:
            img = augmented
        else:
            keep = torch.rand(img.shape[0], 1, 1, 1, device=img.device) < p
            img = torch.where(keep, augmented, img)
    out = img.reshape(x.shape[0], -1)
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    return out


def D_train_aug(x, G, D, D_optimizer, criterion, device, policy=(), noise_std=0.0, p=1.0):
    """utils.D_train with every D input passed through diff_augment.

    Returns (D loss, r) where r = mean(sign(D(real) - 0.5)) is ADA's overfitting
    signal: near 1 when D is confidently right on (augmented) real images.
    """
    D.zero_grad()
    x_real = x.to(device)
    y_real = torch.ones(x.shape[0], 1, device=device)
    D_real = D(diff_augment(x_real, policy, noise_std, p))
    D_real_loss = criterion(D_real, y_real)
    z = torch.randn(x.shape[0], 100, device=device)
    x_fake = G(z)
    y_fake = torch.zeros(x.shape[0], 1, device=device)
    D_fake_loss = criterion(D(diff_augment(x_fake, policy, noise_std, p)), y_fake)
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
    return D_loss.data.item(), torch.sign(D_real.detach() - 0.5).mean().item()


def G_train_aug(x, G, D, G_optimizer, criterion, device, policy=(), noise_std=0.0, p=1.0):
    """utils.G_train with the generated images augmented before D (gradients flow through)."""
    G.zero_grad()
    z = torch.randn(x.shape[0], 100, device=device)
    y = torch.ones(x.shape[0], 1, device=device)
    G_loss = criterion(D(diff_augment(G(z), policy, noise_std, p)), y)
    G_loss.backward()
    G_optimizer.step()
    return G_loss.data.item()
