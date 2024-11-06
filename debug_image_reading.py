import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tifffile


def read_and_plot_channels():
    # Read the image using tifffile instead of PIL
    filepath = "calib_EBT3_103124001.tif"

    # Read with tifffile to preserve 16-bit depth
    img_array = tifffile.imread(filepath)

    print(f"\nTIFF file details:")
    print(f"Shape: {img_array.shape}")
    print(f"Data type: {img_array.dtype}")
    print(f"Value range: {img_array.min()} to {img_array.max()}")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('16-bit Channel Analysis', fontsize=16)

    # Plot original RGB image (scaled for display)
    display_img = (img_array / img_array.max() * 255).astype(np.uint8)
    axes[0, 0].imshow(display_img)
    axes[0, 0].set_title(f'Original RGB Image\nShape: {img_array.shape}, Type: {img_array.dtype}')
    axes[0, 0].axis('off')

    # Plot individual channels
    channel_names = ['Red', 'Green', 'Blue']
    for i in range(3):
        channel = img_array[:, :, i]
        # Scale for display but use original values for statistics
        display_channel = (channel / channel.max() * 255).astype(np.uint8)
        axes[(i + 1) // 2, (i + 1) % 2].imshow(display_channel, cmap='gray')
        axes[(i + 1) // 2, (i + 1) % 2].set_title(
            f'{channel_names[i]} Channel\n'
            f'Range: {channel.min()} to {channel.max()}\n'
            f'Mean: {channel.mean():.1f}'
        )
        axes[(i + 1) // 2, (i + 1) % 2].axis('off')

        # Print channel statistics
        print(f"\n{channel_names[i]} channel statistics:")
        print(f"Min: {channel.min()}")
        print(f"Max: {channel.max()}")
        print(f"Mean: {channel.mean():.1f}")
        print(f"Std: {channel.std():.1f}")

    plt.tight_layout()
    plt.savefig('channel_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot histograms of each channel
    plt.figure(figsize=(15, 5))

    for i, color in enumerate(['red', 'green', 'blue']):
        plt.subplot(1, 3, i + 1)
        plt.hist(img_array[:, :, i].ravel(), bins=256, color=color, alpha=0.7)
        plt.title(f'{channel_names[i]} Channel Histogram\nFull 16-bit range')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('channel_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print unique values in each channel to verify bit depth
    for i, name in enumerate(channel_names):
        unique_values = np.unique(img_array[:, :, i])
        print(f"\n{name} channel unique values:")
        print(f"Number of unique values: {len(unique_values)}")
        print(f"Min unique: {unique_values.min()}")
        print(f"Max unique: {unique_values.max()}")


if __name__ == "__main__":
    read_and_plot_channels()