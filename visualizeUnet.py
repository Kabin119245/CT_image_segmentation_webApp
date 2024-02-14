import torch
import os
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from transform import test_transforms
from monai.inferers import sliding_window_inference
from torchvision.transforms import ToPILImage
from PIL import Image
from io import BytesIO
import numpy as np
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

device = torch.device("cuda:0")
model_dir='./model'
model_Unet = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


model_Unet.load_state_dict(torch.load(
    os.path.join(model_dir, "Unet.pth")))
model_Unet.eval()

def UNet_output(volume, model, roi_size=(128, 128, 64), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    softmax_activation = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        # Preprocess input
        test_data = {"vol": volume}
        test_data = test_transforms(test_data)
        t_volume = test_data['vol'].unsqueeze(0).to(device)

        # Perform sliding window inference
        test_outputs = sliding_window_inference(t_volume.to(device), roi_size, 1, model)

        # Apply softmax activation to convert logits to probabilities
        test_outputs = softmax_activation(test_outputs)

        # Create a list to store base64-encoded images
        Unet_base64_list = []

        # Loop through all slices
        for slice_idx in range(test_outputs.shape[-1]):
            # Create a new figure and axis for each visualization
            fig, ax = plt.subplots(figsize=(6, 6))

            # Plot predicted segmentation with original colors
            overlay = torch.argmax(test_outputs, dim=1)[0, :, :, slice_idx].cpu().numpy()
            overlay = np.ma.masked_where(overlay == 0, overlay)  # Mask background (class 0)
            ax.imshow(t_volume[0, 0, :, :, slice_idx].cpu(), cmap="gray")
            ax.imshow(overlay, cmap="jet", alpha=0.4, vmin=0, vmax=4)

            ax.set_title(f"Unet Segmentation Output - Slice {slice_idx}")
            ax.axis('off')

            # Save the figure to a BytesIO buffer
            img_buf = BytesIO()
            FigureCanvasAgg(fig).print_png(img_buf)
            img_buf.seek(0)
            img_Unet = base64.b64encode(img_buf.read()).decode('utf-8')

            Unet_base64_list.append(  img_Unet)

        return Unet_base64_list