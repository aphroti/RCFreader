import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
from typing import List, Tuple


def read_tiff(filepath: str) -> np.ndarray:
    """
    Read a TIFF file and convert to numpy array, preserving 16-bit depth.
    """
    image = tifffile.imread(filepath)
    print(f"Reading file: {filepath}")
    return image


class ROISelector:
    def __init__(self, image: np.ndarray, max_rois: int):
        self.original = image
        self.max_rois = max_rois

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
        self.rois = []
        self.drawing = False
        self.start_x = self.start_y = -1
        self.window_name = f'Select ROIs ({len(self.rois)}/{max_rois}) - Press SPACE when done, ESC to clear last'

    def update_window_title(self):
        cv2.setWindowTitle(self.window_name,
                           f'Select ROIs ({len(self.rois)}/{self.max_rois}) - Press SPACE when done, ESC to clear last')

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.rois) < self.max_rois:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.image = self.display_image.copy()
            # Redraw existing ROIs
            for i, roi in enumerate(self.rois, 1):
                y1, x1 = int(roi[0].start * self.scale), int(roi[1].start * self.scale)
                y2, x2 = int(roi[0].stop * self.scale), int(roi[1].stop * self.scale)
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.image, f"#{i}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.image.copy()
                cv2.rectangle(img_copy, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, img_copy)

        elif event == cv2.EVENT_LBUTTONUP and len(self.rois) < self.max_rois:
            self.drawing = False
            # Convert coordinates back to original scale
            x1, y1 = min(self.start_x, x), min(self.start_y, y)
            x2, y2 = max(self.start_x, x), max(self.start_y, y)
            orig_x1 = int(x1 / self.scale)
            orig_y1 = int(y1 / self.scale)
            orig_x2 = int(x2 / self.scale)
            orig_y2 = int(y2 / self.scale)
            self.rois.append((slice(orig_y1, orig_y2), slice(orig_x1, orig_x2)))

            # Draw the rectangle on display image
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.image, f"#{len(self.rois)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.image)

            self.update_window_title()

            if len(self.rois) == self.max_rois:
                print(f"\nMaximum number of ROIs ({self.max_rois}) reached!")
                print("Press SPACE to continue...")

    def select_rois(self) -> List[Tuple[slice, slice]]:
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.image)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                if self.rois:  # Remove last ROI if exists
                    self.rois.pop()
                    self.image = self.display_image.copy()
                    # Redraw remaining ROIs
                    for i, roi in enumerate(self.rois, 1):
                        y1, x1 = int(roi[0].start * self.scale), int(roi[1].start * self.scale)
                        y2, x2 = int(roi[0].stop * self.scale), int(roi[1].stop * self.scale)
                        cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(self.image, f"#{i}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow(self.window_name, self.image)
                    self.update_window_title()
            elif key == 32:  # SPACE
                if len(self.rois) < self.max_rois:
                    print(f"\nWarning: Only {len(self.rois)} regions selected out of {self.max_rois}")
                    response = input("Continue anyway? (y/n): ")
                    if response.lower() != 'y':
                        continue
                break

        cv2.destroyAllWindows()
        return self.rois


def analyze_film(filepath: str, num_rois: int, control_values: Tuple[float, float, float]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze radiochromatic film with manual ROI selection.

    Args:
        filepath: Path to the TIFF file
        num_rois: Number of ROIs to select (matching length of dose array)
        control_values: (r0, g0, b0) values from control measurement

    Returns:
        Tuple of arrays containing OD values for R,G,B channels and average OD
    """
    print("\nStarting analysis...")
    print(f"Reading file: {filepath}")
    image = read_tiff(filepath)

    # Unpack control values
    r0, g0, b0 = control_values
    T0_avg = (r0 + g0 + b0) / 3.0
    print(f"\nUsing control values (T0):")
    print(f"Red: {r0:.1f}")
    print(f"Green: {g0:.1f}")
    print(f"Blue: {b0:.1f}")
    print(f"Average T0: {T0_avg:.1f}")

    # Create ROI selector
    selector = ROISelector(image, num_rois)
    print(f"\nSelect {num_rois} regions of interest:")
    print("1. Draw rectangles with left mouse button")
    print("2. Press ESC to remove last selection")
    print("3. Press SPACE when finished")
    rois = selector.select_rois()

    # Initialize arrays for raw measurements
    r_raw, g_raw, b_raw = [], [], []

    # Process each ROI
    for i, roi in enumerate(rois, 1):
        print(f"\nProcessing ROI {i}...")
        roi_data = image[roi[0], roi[1]]
        means = [np.mean(roi_data[:, :, i]) for i in range(3)]
        r_raw.append(means[0])
        g_raw.append(means[1])
        b_raw.append(means[2])
        print(f"Raw values - R:{means[0]:.1f}, G:{means[1]:.1f}, B:{means[2]:.1f}")

    # Convert to numpy arrays
    r_raw = np.array(r_raw)
    g_raw = np.array(g_raw)
    b_raw = np.array(b_raw)

    # Calculate T (r0 - r_n, g0 - g_n, b0 - b_n)
    T_r = r0 - r_raw
    T_g = g0 - g_raw
    T_b = b0 - b_raw
    T_avg = (T_r + T_g + T_b) / 3.0

    print("\nT values (control - measurement):")
    for i in range(len(T_r)):
        print(f"ROI {i + 1}: R={T_r[i]:.1f}, G={T_g[i]:.1f}, B={T_b[i]:.1f}, Avg={T_avg[i]:.1f}")

    # Calculate T/T0
    ToverT0_r = T_r / r0
    ToverT0_g = T_g / g0
    ToverT0_b = T_b / b0
    ToverT0_avg = T_avg / T0_avg

    print("\nT/T0 values:")
    for i in range(len(ToverT0_r)):
        print(
            f"ROI {i + 1}: R={ToverT0_r[i]:.3f}, G={ToverT0_g[i]:.3f}, B={ToverT0_b[i]:.3f}, Avg={ToverT0_avg[i]:.3f}")

    # Calculate OD = -log10(T/T0)
    OD_r = -np.log10(ToverT0_r)
    OD_g = -np.log10(ToverT0_g)
    OD_b = -np.log10(ToverT0_b)
    OD_avg = -np.log10(ToverT0_avg)

    print("\nOptical Density (OD) values:")
    for i in range(len(OD_r)):
        print(f"ROI {i + 1}: R={OD_r[i]:.3f}, G={OD_g[i]:.3f}, B={OD_b[i]:.3f}, Avg={OD_avg[i]:.3f}")

    return (OD_r, OD_g, OD_b, OD_avg)


def plot_calibration(D: np.ndarray, OD_r: np.ndarray, OD_g: np.ndarray, OD_b: np.ndarray, OD_avg: np.ndarray):
    """
    Plot optical density calibration curves for all channels and average.

    Args:
        D: Dose values array
        OD_r, OD_g, OD_b: Optical density values for each channel
        OD_avg: Average optical density values
    """
    from scipy.optimize import curve_fit

    def fit_function(x, a, b, c):
        """Function to fit: y = a + b/(x-c)"""
        return a + b / (x - c)

    # Plot individual channels
    plt.figure(figsize=(12, 8))
    colors = ['red', 'green', 'blue']
    channel_names = ['Red', 'Green', 'Blue']
    values = [OD_r, OD_g, OD_b]

    x_smooth = np.linspace(D.min(), D.max(), 1000)

    for i, (color, name, y_data) in enumerate(zip(colors, channel_names, values)):
        try:
            p0 = [np.min(y_data), 1000, 0]
            popt, pcov = curve_fit(fit_function, D, y_data, p0=p0,
                                   bounds=([0, -np.inf, -np.inf],
                                           [np.inf, np.inf, D.min()]))

            a, b, c = popt
            plt.plot(D, y_data, 'o', color=color, label=name, markersize=8)
            y_fit = fit_function(x_smooth, a, b, c)
            plt.plot(x_smooth, y_fit, color=color, linestyle='-', alpha=0.7)

            eq_text = f"{name}: OD = {a:.3f} + {b:.3f}/(x - {c:.3f})"
            plt.text(0.02, 0.90 - 0.05 * i, eq_text,
                     transform=plt.gca().transAxes,
                     color=color, fontsize=10)

        except RuntimeError as e:
            print(f"Fitting failed for {name} channel: {e}")
            plt.plot(D, y_data, 'o', color=color, label=name, markersize=8)

    plt.xlabel('Dose (cGy)', fontsize=12)
    plt.ylabel('Optical Density (OD)', fontsize=12)
    plt.title('Film Calibration Curves - Individual Channels', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    y_min = min([np.min(v) for v in values])
    y_max = max([np.max(v) for v in values])
    plt.ylim(y_min * 0.9, y_max * 1.1)

    plt.text(0.02, 1.02, 'Fitted function: OD = a + b/(x-c)',
             transform=plt.gca().transAxes,
             fontsize=12, color='black')

    plt.tight_layout()
    plt.savefig('calibration_curves_OD.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot average OD
    plt.figure(figsize=(12, 8))
    try:
        p0 = [np.min(OD_avg), 1000, 0]
        popt, pcov = curve_fit(fit_function, D, OD_avg, p0=p0,
                               bounds=([0, -np.inf, -np.inf],
                                       [np.inf, np.inf, D.min()]))

        a, b, c = popt
        plt.plot(D, OD_avg, 'ko', label='Average', markersize=8)
        y_fit = fit_function(x_smooth, a, b, c)
        plt.plot(x_smooth, y_fit, 'k-', alpha=0.7)

        eq_text = f"Average: OD = {a:.3f} + {b:.3f}/(x - {c:.3f})"
        plt.text(0.02, 0.95, eq_text,
                 transform=plt.gca().transAxes,
                 color='black', fontsize=10)

    except RuntimeError as e:
        print(f"Fitting failed for average OD: {e}")
        plt.plot(D, OD_avg, 'ko', label='Average', markersize=8)

    plt.xlabel('Dose (cGy)', fontsize=12)
    plt.ylabel('Average Optical Density (OD)', fontsize=12)
    plt.title('Film Calibration Curve - Average of All Channels', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    plt.text(0.02, 1.02, 'Fitted function: OD = a + b/(x-c)',
             transform=plt.gca().transAxes,
             fontsize=12, color='black')

    plt.tight_layout()
    plt.savefig('calibration_curve_avg_OD.png', dpi=300, bbox_inches='tight')
    plt.show()