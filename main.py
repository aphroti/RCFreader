import np

# Define dose values
D = np.array([200, 250, 350, 500, 700, 1000, 1300, 1700, 2200])

# Analyze film
filepath = "path_to_your_tiff_file.tiff"  # Replace with actual path
Rd = analyze_film(filepath)

# Plot results
plot_results(Rd, D)