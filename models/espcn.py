import torch
import torch.nn as nn

class ESPCN(nn.Module):
    def __init__(self, in_channels=3, channels=64, out_channels=1, upscale_factor=1):
        super(ESPCN, self).__init__()
        hidden_channels = channels // 2
        self.bn = nn.BatchNorm2d(in_channels)
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.sub_pixel_0 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Sigmoid(),
        )
        self.sub_pixel_1 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.bn(x)
        features = self.feature_maps(x)
        return self.sub_pixel_0(features), self.sub_pixel_1(features)


class FrequencyMaskingModule(nn.Module):
    def __init__(self):
        super(FrequencyMaskingModule, self).__init__()
        self.mask_autoencoder = ESPCN()

    def forward(self, image: torch.Tensor):
        # 1. FFT
        freq_image = torch.fft.fftn(image * 255.0, dim=(-2, -1))
        freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))
        magnitude_spectrum = torch.abs(freq_image)

        # 2. Prepare ESPCN Input
        log_scaled_input = (20 * torch.log(magnitude_spectrum + 1e-7)) / 255.0

        # 3. Predict Masks
        mask_mid_frq, mask_mid_filterd = self.mask_autoencoder(log_scaled_input)
        mask_mid_frq = mask_mid_frq.to(freq_image.dtype)
        mask_mid_filterd = mask_mid_filterd.to(freq_image.dtype)

        # 4. Create Pseudo-Fake
        middle_filtered = freq_image * mask_mid_filterd
        middle_filtered = torch.fft.ifftshift(middle_filtered, dim=(-2, -1))
        spatial_filtered_array = torch.abs(torch.fft.ifftn(middle_filtered, dim=(-2, -1)))
        
        # Keep dimensions for safe broadcasting
        min_filt = spatial_filtered_array.amin(dim=(-2, -1), keepdim=True)
        max_filt = spatial_filtered_array.amax(dim=(-2, -1), keepdim=True)
        middle_filtered_image = spatial_filtered_array / (max_filt - min_filt + 1e-7)

        return middle_filtered_image

def load_pretrained_fmre(fmre_module, checkpoint_path, device):
    print(f"Loading specialized weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    full_state_dict = checkpoint.get('state_dict', checkpoint)
    
    espc_weights = {}
    for key, value in full_state_dict.items():
        if "fft_filter_module.mask_autoencoder" in key:
            new_key = key.replace("fft_filter_module.mask_autoencoder.", "")
            espc_weights[new_key] = value

    fmre_module.mask_autoencoder.load_state_dict(espc_weights, strict=True)
    return fmre_module