import torch
import os
import matplotlib.patches as mpatches
from monai.networks.nets import SegResNet
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
model_SegResNet = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,
    act=('RELU',{'inplace':True}),
    norm=Norm.BATCH,
).to(device)


model_SegResNet.load_state_dict(torch.load(
    os.path.join(model_dir, "SegResNet.pth")))
model_SegResNet.eval()

def SegResNet_output(volume, model, roi_size=(128, 128, 64), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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
        SegResNet_base64_list = []

        # Loop through all slices
        for slice_idx in range(test_outputs.shape[-1]):
            # Create a new figure and axis for each visualization
            fig, ax = plt.subplots(figsize=(6, 6))
            brown_patch = mpatches.Patch(color='brown', label='Pancrease')
            green_patch = mpatches.Patch(color='green', label='Kidney')
            blue_patch = mpatches.Patch(color='blue', label='Liver')
            orange_patch = mpatches.Patch(color='orange', label='Spleen')
            ax.legend(handles=[brown_patch,green_patch,blue_patch,orange_patch]) 
            # Plot predicted segmentation with original colors
            overlay = torch.argmax(test_outputs, dim=1)[0, :, :, slice_idx].cpu().numpy()
            overlay = np.ma.masked_where(overlay == 0, overlay)  # Mask background (class 0)
            ax.imshow(t_volume[0, 0, :, :, slice_idx].cpu(), cmap="gray")
            ax.imshow(overlay, cmap="jet", alpha=0.4, vmin=0, vmax=4)

            ax.set_title(f"SegResNet Segmentation Output - Slice {slice_idx}")
            ax.axis('off')

            # Save the figure to a BytesIO buffer
            img_buf = BytesIO()
            FigureCanvasAgg(fig).print_png(img_buf)
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

            SegResNet_base64_list.append(img_base64)

        return  SegResNet_base64_list