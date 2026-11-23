import torch
from genesis.models.image_unet import ImageUNet, encode_texts
from genesis.diffusion.image_scheduler import ImageDDPM


def test_image_unet_output_shape():
    model = ImageUNet(base_channels=16, text_dim=32)
    image = torch.randn(2, 3, 32, 32)
    text = encode_texts(["a red car", "a blue house"], "cpu")
    output = model(image, torch.tensor([2, 7]), text)
    assert output.shape == image.shape


def test_image_diffusion_forward_noise_shape():
    scheduler = ImageDDPM(timesteps=20)
    clean = torch.randn(2, 3, 16, 16)
    noisy = scheduler.add_noise(clean, torch.randn_like(clean), torch.tensor([0, 10]))
    assert noisy.shape == clean.shape
