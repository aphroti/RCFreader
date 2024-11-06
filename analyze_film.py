import numpy as np
import os
from radiochromatic_analysis import analyze_film, plot_calibration
from control_measurement import measure_control
from typing import Tuple


def get_control_values(base_path: str) -> Tuple[float, float, float]:
    """
    Get control values from file or measure new ones.

    Args:
        base_path: Base directory path where control_measurement.txt should be

    Returns:
        Tuple of (red, green, blue) control values
    """
    control_file = os.path.join(base_path, "control_measurement.txt")
    control_image = os.path.join(base_path, "calib_EBT3_110324001.tif")

    # Try to read existing control values
    if os.path.exists(control_file):
        try:
            print("\nReading existing control values...")
            with open(control_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:  # Ensure we have all three values
                    r = float(lines[0].strip())
                    g = float(lines[1].strip())
                    b = float(lines[2].strip())
                    print(f"Found control values (T0) - R:{r:.1f}, G:{g:.1f}, B:{b:.1f}")

                    # Ask if user wants to use these values
                    response = input("\nUse existing control values? (y/n): ")
                    if response.lower() == 'y':
                        return (r, g, b)

                print("Will measure new control values...")
        except Exception as e:
            print(f"Error reading control file: {e}")
            print("Will measure new control values...")
    else:
        print("\nNo existing control measurements found.")
        print("Will measure new control values...")

    # Check if control image exists
    if not os.path.exists(control_image):
        raise FileNotFoundError(f"Control image not found: {control_image}\n"
                                f"Please ensure 'calib_EBT3_110324001.tif' is in the directory.")

    print(f"\nUsing control image: {control_image}")

    # Measure new control values
    control_values = measure_control(control_image)

    # Save to file
    try:
        print("\nSaving control values...")
        with open(control_file, 'w') as f:
            for value in control_values:
                f.write(f"{value}\n")
        print(f"Control values saved to {control_file}")
    except Exception as e:
        print(f"Warning: Could not save control values to file: {e}")

    return control_values


def main():
    # Set file paths relative to current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    calib_filepath = os.path.join(current_dir, "calib_EBT3_103124001.tif")

    # Define dose values array
    D = np.array([200, 250, 350, 500, 700, 1000, 1300, 1700, 2200])

    try:
        # Get control values first
        control_r, control_g, control_b = get_control_values(current_dir)
        print("\nUsing control values (T0):")
        print(f"Red: {control_r:.1f}")
        print(f"Green: {control_g:.1f}")
        print(f"Blue: {control_b:.1f}")

        print(f"\nAnalyzing calibration scan at: {calib_filepath}")

        # Analyze film and calculate optical density
        OD_r, OD_g, OD_b, OD_avg = analyze_film(calib_filepath, len(D), (control_r, control_g, control_b))

        # Print results
        print("\nFinal results:")
        print("\nDose values (cGy):")
        print(", ".join(f"{val:.1f}" for val in D))

        print("\nOptical Density values:")
        print("\nRed channel OD:")
        print(", ".join(f"{val:.3f}" for val in OD_r))

        print("\nGreen channel OD:")
        print(", ".join(f"{val:.3f}" for val in OD_g))

        print("\nBlue channel OD:")
        print(", ".join(f"{val:.3f}" for val in OD_b))

        # Save results to file
        results_file = os.path.join(current_dir, "calibration_results.txt")
        print(f"\nSaving results to {results_file}")
        with open(results_file, 'w') as f:
            f.write("Calibration Results\n")
            f.write("\nControl Values (T0):\n")
            f.write(f"Red: {control_r:.1f}\n")
            f.write(f"Green: {control_g:.1f}\n")
            f.write(f"Blue: {control_b:.1f}\n")

            f.write("\nDose (cGy)  Red OD    Green OD  Blue OD\n")
            for i in range(len(D)):
                f.write(f"{D[i]:8.1f}  {OD_r[i]:8.3f}  {OD_g[i]:8.3f}  {OD_b[i]:8.3f}\n")

        # Plot calibration curves
        plot_calibration(D, OD_r, OD_g, OD_b, OD_avg)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()