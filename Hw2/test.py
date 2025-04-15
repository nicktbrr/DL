import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import time
import numpy as np
from tqdm import tqdm

# Define the ConvNeXtLayer as provided


class ConvNeXtLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super(ConvNeXtLayer, self).__init__()
        # Depthwise Convolution
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, groups=in_channels)
        # Layer Normalization
        self.norm = nn.LayerNorm(in_channels)  # Normalize over channels only
        # Activation
        self.activation = nn.GELU()
        # Pointwise Convolution
        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)

        # Permute to [B, H, W, C] for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)  # Apply normalization over last dimension (channels)
        x = x.permute(0, 3, 1, 2)  # Back to [B, C, H, W]

        x = self.activation(x)
        x = self.pwconv(x)
        return x

# Utility function to set device


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Data Exploration and Preprocessing
# Define transformations for images and masks


def get_transforms(img_size=224):
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # For masks, we just resize and convert to tensor (no normalization)
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    return image_transform, mask_transform

# Custom dataset class to apply different transforms to image and mask


class PetSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, img_transform=None, mask_transform=None):
        self.dataset = dataset
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        if self.img_transform:
            img = self.img_transform(img)

        if self.mask_transform:
            mask = self.mask_transform(mask)
            # Convert mask to long tensor and squeeze unnecessary dimensions
            mask = mask.squeeze().long()
        else:
            # Convert PIL mask to tensor if no transform
            mask = torch.tensor(np.array(mask), dtype=torch.long)

        return img, mask

# Download and prepare the dataset


def prepare_data(batch_size=16, img_size=224, val_split=0.2):
    # Get transforms
    img_transform, mask_transform = get_transforms(img_size)

    # Download the Oxford-IIIT Pet dataset
    # First, just get the dataset for visualization purposes
    base_dataset = torchvision.datasets.OxfordIIITPet(
        root='./data',
        split='trainval',
        target_types='segmentation',
        download=True,
        transform=None,
        target_transform=None
    )

    # Create custom dataset with separate transforms for training/validation
    custom_dataset = PetSegmentationDataset(
        base_dataset,
        img_transform=img_transform,
        mask_transform=mask_transform
    )

    # Split into training and validation sets
    num_val = int(len(custom_dataset) * val_split)
    num_train = len(custom_dataset) - num_val
    train_dataset, val_dataset = random_split(
        custom_dataset, [num_train, num_val])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, base_dataset

# Function to visualize sample images and masks


def visualize_samples(dataset, num_samples=3):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))

    for i in range(num_samples):
        # Get a sample
        img, mask = dataset[i]

        # Display image (handle both PIL Image and Tensor)
        if hasattr(img, 'permute'):  # It's a tensor
            axes[i, 0].imshow(img.permute(1, 2, 0))
        else:  # It's a PIL Image
            axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Sample {i+1} - Image")
        axes[i, 0].axis('off')

        # Display mask (handle both PIL Image and Tensor)
        if hasattr(mask, 'cpu'):  # It's a tensor
            axes[i, 1].imshow(mask, cmap='tab20')
        else:  # It's a PIL Image
            axes[i, 1].imshow(mask, cmap='tab20')
        axes[i, 1].set_title(f"Sample {i+1} - Mask")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# 2. Model Design and Implementation
# Encoder block with ConvNeXtLayer and optional downsampling


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(EncoderBlock, self).__init__()
        self.downsample = downsample

        self.conv = ConvNeXtLayer(in_channels, out_channels)

        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        if self.downsample:
            # Return both downsampled and original for skip connections
            return self.pool(x), x
        return x, x

# Decoder block for upsampling


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super(DecoderBlock, self).__init__()

        # For upsampling path, the input to the upsample is just the bottleneck features
        # not the concatenated ones
        if upsample:
            self.upsample = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 4, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels // 4 + in_channels // 2,
                          out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.upsample = nn.Identity()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if skip is not None:
            # Ensure the spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)

# Complete Segmentation Model with ConvNeXt Encoder and UNet-style Decoder


class SegmentationModel(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(SegmentationModel, self).__init__()

        # Encoder pathway (downsampling)
        self.enc1 = EncoderBlock(
            in_channels, 64, downsample=True)     # 64x112x112
        self.enc2 = EncoderBlock(
            64, 128, downsample=True)             # 128x56x56
        self.enc3 = EncoderBlock(
            128, 256, downsample=True)            # 256x28x28
        self.enc4 = EncoderBlock(
            256, 512, downsample=True)            # 512x14x14

        # Bridge
        self.bridge = ConvNeXtLayer(
            512, 512)                          # 512x14x14

        # Decoder pathway (upsampling with skip connections)
        # Each decoder takes the upsampled features and concatenates with skip connection
        self.up4 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2)  # 256x28x28
        self.dec4 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2)  # 128x56x56
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2)   # 64x112x112
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2)    # 32x224x224
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final layer
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder pathway with skip connections
        # 64x112x112
        x, skip1 = self.enc1(x)
        # 128x56x56
        x, skip2 = self.enc2(x)
        # 256x28x28
        x, skip3 = self.enc3(x)
        # 512x14x14
        x, skip4 = self.enc4(x)

        # Bridge
        # 512x14x14
        x = self.bridge(x)

        # Decoder pathway using skip connections
        # 256x28x28
        x = self.up4(x)
        # (256+512)x28x28
        x = torch.cat([x, skip4], dim=1)
        # 256x28x28
        x = self.dec4(x)

        # 128x56x56
        x = self.up3(x)
        # (128+256)x56x56
        x = torch.cat([x, skip3], dim=1)
        # 128x56x56
        x = self.dec3(x)

        # 64x112x112
        x = self.up2(x)
        # (64+128)x112x112
        x = torch.cat([x, skip2], dim=1)
        # 64x112x112
        x = self.dec2(x)

        # 32x224x224
        x = self.up1(x)
        # (32+64)x224x224
        x = torch.cat([x, skip1], dim=1)
        # 32x224x224
        x = self.dec1(x)

        # Final layer to get segmentation map
        # num_classesx224x224
        x = self.final(x)

        return x

# 3. Training and Evaluation
# Metric functions


def pixel_accuracy(output, target):
    """
    Computes pixel accuracy.

    Parameters:
      output (torch.Tensor): Model predictions of shape [B, num_classes, H, W].
      target (torch.Tensor): Ground truth labels of shape [B, H, W].

    Returns:
      float: Pixel accuracy.
    """
    with torch.no_grad():
        # Get predicted classes from output
        pred = output.argmax(dim=1)
        correct = (pred == target).float()
        acc = correct.sum() / correct.numel()
    return acc


def intersection_over_union(output, target, num_classes):
    """
    Computes Mean Intersection-over-Union (IoU) over all classes.

    Parameters:
      output (torch.Tensor): Model predictions of shape [B, num_classes, H, W].
      target (torch.Tensor): Ground truth labels of shape [B, H, W].
      num_classes (int): Number of segmentation classes.

    Returns:
      float: Mean IoU over all classes.
    """
    with torch.no_grad():
        pred = output.argmax(dim=1)
        ious = []
        for cls in range(num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()
            if union == 0:
                # If no ground truth for class, count IoU as 1.
                ious.append(torch.tensor(1.0, device=pred.device))
            else:
                ious.append(intersection / union)
        mean_iou = torch.mean(torch.stack(ious))
    return mean_iou

# Training function


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, num_classes, device):
    # For tracking metrics
    train_losses = []
    train_accuracies = []
    train_ious = []
    val_losses = []
    val_accuracies = []
    val_ious = []

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training phase
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_iou = 0
        num_batches = 0

        # Add tqdm progress bar for training loop
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            acc = pixel_accuracy(outputs, masks)
            iou = intersection_over_union(outputs, masks, num_classes)

            epoch_loss += loss.item()
            epoch_accuracy += acc.item()
            epoch_iou += iou.item()
            num_batches += 1

            # Update progress bar with current metrics
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc.item():.4f}",
                'iou': f"{iou.item():.4f}"
            })

        # Average metrics for the epoch
        epoch_loss /= num_batches
        epoch_accuracy /= num_batches
        epoch_iou /= num_batches

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        train_ious.append(epoch_iou)

        # Validation phase
        model.eval()
        val_loss = 0
        val_accuracy = 0
        val_iou = 0
        num_val_batches = 0

        # Add tqdm progress bar for validation loop
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                acc = pixel_accuracy(outputs, masks)
                iou = intersection_over_union(outputs, masks, num_classes)

                val_loss += loss.item()
                val_accuracy += acc.item()
                val_iou += iou.item()
                num_val_batches += 1

                # Update progress bar with current metrics
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{acc.item():.4f}",
                    'iou': f"{iou.item():.4f}"
                })

            val_loss /= num_val_batches
            val_accuracy /= num_val_batches
            val_iou /= num_val_batches

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_ious.append(val_iou)

        # Print epoch results
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s - Loss: {epoch_loss:.4f} - Acc: {epoch_accuracy:.4f} - IoU: {epoch_iou:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f} - Val IoU: {val_iou:.4f}")

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'train_ious': train_ious,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_ious': val_ious
    }

# Function to visualize model predictions


def visualize_predictions(model, dataset, device, num_samples=3):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))

    # Get transforms for inference if needed
    img_transform, _ = get_transforms()

    for i in range(num_samples):
        # Get a random sample
        idx = np.random.randint(0, len(dataset))
        img, mask = dataset[idx]

        # Convert PIL image to tensor if needed
        if not hasattr(img, 'unsqueeze'):  # If it's not a tensor
            img_tensor = img_transform(img).unsqueeze(0).to(device)
        else:
            img_tensor = img.unsqueeze(0).to(device)

        # Get model prediction
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()

        # Display image
        if hasattr(img, 'permute'):  # It's a tensor
            img_display = img.permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed
            img_display = np.clip((img_display * np.array([0.229, 0.224, 0.225]) +
                                  np.array([0.485, 0.456, 0.406])), 0, 1)
        else:  # It's a PIL Image
            img_display = np.array(img)
            if img_display.max() > 1.0:  # Normalize to [0,1] if needed
                img_display = img_display / 255.0

        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')

        # Display ground truth mask
        if hasattr(mask, 'cpu'):  # It's a tensor
            mask_display = mask.cpu().numpy()
        else:  # It's a PIL Image
            mask_display = np.array(mask)

        axes[i, 1].imshow(mask_display, cmap='tab20')
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis('off')

        # Display predicted mask
        axes[i, 2].imshow(pred, cmap='tab20')
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Function to visualize training metrics


def plot_metrics(metrics):
    epochs = range(1, len(metrics['train_losses']) + 1)

    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics['train_losses'], 'b-', label='Training Loss')
    plt.plot(epochs, metrics['val_losses'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics['train_accuracies'],
             'b-', label='Training Accuracy')
    plt.plot(epochs, metrics['val_accuracies'],
             'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot IoUs
    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics['train_ious'], 'b-', label='Training IoU')
    plt.plot(epochs, metrics['val_ious'], 'r-', label='Validation IoU')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main function to run everything


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parameters
    batch_size = 16
    img_size = 224
    num_classes = 3  # Background, foreground (pet), and boundary/trimap
    num_epochs = 10
    learning_rate = 0.001

    # Get device
    device = get_device()
    print(f"Using device: {device}")

    # Prepare data
    train_loader, val_loader, base_dataset = prepare_data(batch_size, img_size)

    # Visualize some samples from the raw dataset
    # print("Visualizing sample images and masks:")
    # visualize_samples(base_dataset, num_samples=3)

    img, mask = next(iter(train_loader))
    print(mask[0])

    # Get the number of classes from masks
    # Sample a few masks to determine the unique classes
    unique_classes = set()
    for i in range(min(10, len(base_dataset))):
        _, mask = base_dataset[i]
        mask_array = np.array(mask)
        unique_classes.update(np.unique(mask_array))

    # Determine number of classes from the masks
    num_classes = len(unique_classes)
    print(f"Detected {num_classes} unique classes in the masks")

    # Create model, loss function, and optimizer
    model = SegmentationModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # Train the model
    # print("Starting training...")
    # metrics = train(model, train_loader, val_loader, criterion,
    #                 optimizer, num_epochs, num_classes, device)

    # # Plot training metrics
    # print("Plotting training metrics:")
    # plot_metrics(metrics)

    # # Create a custom validation dataset with a few samples for visualization
    # val_samples = []
    # for i in range(3):
    #     idx = np.random.randint(0, len(val_loader.dataset))
    #     val_samples.append(val_loader.dataset[idx])

    # # Visualize predictions on validation samples
    # print("Visualizing model predictions:")
    # model.eval()
    # fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # for i, (img, mask) in enumerate(val_samples):
    #     # Process image for model input
    #     img_input = img.unsqueeze(0).to(device)

    #     # Get model prediction
    #     with torch.no_grad():
    #         output = model(img_input)
    #         pred = output.argmax(dim=1).squeeze().cpu().numpy()

    #     # Display original image
    #     if img.shape[0] == 3:  # If it's already a tensor with channels first
    #         # Denormalize
    #         img_np = img.cpu().numpy().transpose(1, 2, 0)
    #         img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) +
    #                          np.array([0.485, 0.456, 0.406]), 0, 1)
    #         axes[i, 0].imshow(img_np)
    #     else:
    #         axes[i, 0].imshow(img)
    #     axes[i, 0].set_title("Original Image")
    #     axes[i, 0].axis('off')

    #     # Display ground truth mask
    #     axes[i, 1].imshow(mask.cpu().numpy(), cmap='tab20')
    #     axes[i, 1].set_title("Ground Truth Mask")
    #     axes[i, 1].axis('off')

    #     # Display predicted mask
    #     axes[i, 2].imshow(pred, cmap='tab20')
    #     axes[i, 2].set_title("Predicted Mask")
    #     axes[i, 2].axis('off')

    # plt.tight_layout()
    # plt.show()

    # # Print final results
    # print(f"Final results:")
    # print(f"Training Loss: {metrics['train_losses'][-1]:.4f}")
    # print(f"Training Accuracy: {metrics['train_accuracies'][-1]:.4f}")
    # print(f"Training IoU: {metrics['train_ious'][-1]:.4f}")
    # print(f"Validation Loss: {metrics['val_losses'][-1]:.4f}")
    # print(f"Validation Accuracy: {metrics['val_accuracies'][-1]:.4f}")
    # print(f"Validation IoU: {metrics['val_ious'][-1]:.4f}")

    # return model, metrics


# Run the main function if this script is executed
if __name__ == "__main__":
    main()
