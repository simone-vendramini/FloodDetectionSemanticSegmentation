import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

classes = {
    0: "background",
    1: "building-flooded",
    2: "building-not-flooded",
    3: "road-flooded",
    4: "road-not-flooded",
    5: "water",
    6: "tree",
    7: "vehicle",
    8: "pool",
    9: "grass",
}

color_to_label_map = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (120, 120, 180): 2,
    (20, 150, 160): 3,
    (140, 140, 140): 4,
    (250, 230, 61): 5,
    (255, 82, 0): 6,
    (245, 0, 255): 7,
    (0, 235, 255): 8,
    (7, 250, 4): 9,
}

label_to_color_map = {v: k for k, v in color_to_label_map.items()}


def convert_label_to_image(label):
    image = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for k, v in label_to_color_map.items():
        image[label == k] = v
    return image


def _compute_patches():
    patches = []
    for color, label_index in color_to_label_map.items():
        label_name = classes[label_index]
        patch = mpatches.Patch(color=[c / 255 for c in color], label=label_name)
        patches.append(patch)
    return patches


def plot_image(image, label):
    im = image.permute(1, 2, 0)
    im_grayscale = np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes = axes.ravel()

    axes[0].imshow(image.permute(1, 2, 0).cpu(), cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(convert_label_to_image(label[0].cpu()), cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    axes[2].imshow(im_grayscale, cmap="gray")
    axes[2].imshow(convert_label_to_image(label[0].cpu()), alpha=0.5)
    axes[2].set_title("Ground Truth Mask + Image")
    axes[2].axis("off")

    patches = _compute_patches()
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_tensor(tensor, title=None):
    im = tensor.permute(1, 2, 0)
    plt.imshow(im)
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_one_hot_label(label):
    label = label[0, :, :]
    label = label.type(torch.LongTensor)

    label_onehot = torch.nn.functional.one_hot(label, num_classes=10)
    label_onehot = label_onehot.type(torch.float32)

    num_classes = label_onehot.shape[-1]
    _, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(num_classes):
        channel = label_onehot[..., i].cpu().numpy()

        channel = channel * 255

        axes[i].imshow(channel, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(f"Channel {classes[i]}")
        axes[i].axis("off")

    for j in range(num_classes, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_results(image, prediction, ground_truth):
    im = image.permute(1, 2, 0).cpu()
    im_grayscale = np.dot(im[..., :3], [0.2989, 0.5870, 0.1140])

    _, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot original image
    axes[0].imshow(image.permute(1, 2, 0).cpu(), cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot ground truth mask
    axes[1].imshow(convert_label_to_image(ground_truth[0].cpu()), cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # Plot predicted mask
    axes[2].imshow(convert_label_to_image(prediction[0].cpu()), cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    #axes[3].imshow(im_grayscale, cmap="gray")
    #axes[3].imshow(convert_label_to_image(prediction[0].cpu()), alpha=0.25)
    #axes[3].set_title("Predicted Mask + Image")
    #axes[3].axis("off")

    patches = _compute_patches()
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()
