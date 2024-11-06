import numpy as np
import cv2
import tifffile
from typing import Tuple


class ControlROISelector:
    def __init__(self, image: np.ndarray):
        self.original = image

        # Scale down for display while keeping aspect ratio
        scale = min(1000.0 / image.shape[1], 1000.0 / image.shape[0])
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        self.scale = scale

        # Simple linear scaling from 16-bit to 8-bit for display
        display_image = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(3):
            channel = cv2.resize((image[:, :, i] / 256).astype(np.uint8), (width, height))
            display_image[:, :, i] = channel

        self.display_image = display_image
        self.image = self.display_image.copy()
        self.roi = None
        self.drawing = False
        self.start_x = self.start_y = -1
        self.window_name = 'Select Control Area - Press SPACE when done, ESC to clear'

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.image = self.display_image.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.image.copy()
                cv2.rectangle(img_copy, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Convert coordinates back to original scale
            x1, y1 = min(self.start_x, x), min(self.start_y, y)
            x2, y2 = max(self.start_x, x), max(self.start_y, y)
            orig_x1 = int(x1 / self.scale)
            orig_y1 = int(y1 / self.scale)
            orig_x2 = int(x2 / self.scale)
            orig_y2 = int(y2 / self.scale)
            self.roi = (slice(orig_y1, orig_y2), slice(orig_x1, orig_x2))

            # Draw the rectangle on display image
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.image, "Control", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image)

    def select_roi(self) -> Tuple[slice, slice]:
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.image)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                if self.roi is not None:
                    self.roi = None
                    self.image = self.display_image.copy()
                    cv2.imshow(self.window_name, self.image)
            elif key == 32:  # SPACE
                if self.roi is None:
                    print("Please select an area before continuing.")
                    continue
                break

        cv2.destroyAllWindows()
        return self.roi


def measure_control(filepath: str) -> Tuple[float, float, float]:
    """
    Measure control values from an image.

    Args:
        filepath: Path to the TIFF file

    Returns:
        Tuple of (red, green, blue) channel values at dose=0
    """
    print("\nStarting control measurement...")
    print(f"Reading file: {filepath}")

    # Read image using tifffile to preserve 16-bit depth
    image = tifffile.imread(filepath)

    # Create ROI selector and get selection
    selector = ControlROISelector(image)
    print("\nSelect control area:")
    print("1. Draw rectangle with left mouse button")
    print("2. Press ESC to clear selection")
    print("3. Press SPACE when satisfied")
    roi = selector.select_roi()

    # Analyze selected region
    roi_data = image[roi[0], roi[1]]
    means = [np.mean(roi_data[:, :, i]) for i in range(3)]

    print("\nControl measurements (16-bit values):")
    for i, (color, value) in enumerate(zip(['Red', 'Green', 'Blue'], means)):
        print(f"{color} channel mean: {value:.1f}")

    return tuple(means)


if __name__ == "__main__":
    # Example usage
    filepath = input("Enter path to control image file: ")
    try:
        control_values = measure_control(filepath)
        print("\nFinal control values:")
        print(f"Red: {control_values[0]:.1f}")
        print(f"Green: {control_values[1]:.1f}")
        print(f"Blue: {control_values[2]:.1f}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")