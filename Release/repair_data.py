import os

# --- CONFIGURATION ---
# The file you want to fix (full path or relative)
INPUT_FILE = "data/Aleyna_data/01_01_2026_20_12/gazeData_calibrated.txt" 

# The name of the new fixed file
OUTPUT_FILE = "gazeData_fixed.txt"

# HOW MUCH TO SHIFT?
# If text is HIGHER than dots -> You need to move dots UP.
# Since Y=1 is Top and Y=-1 is Bottom:
# ADD positive value (e.g., 0.15) to move dots UP.
# SUBTRACT value (e.g., -0.15) to move dots DOWN.
Y_OFFSET = 0.15  # Start with 0.15 and adjust as needed
# ---------------------

def repair_gaze_file():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"Processing {INPUT_FILE}...")
    print(f"Applying Y Offset: {Y_OFFSET}")

    with open(INPUT_FILE, 'r') as infile, open(OUTPUT_FILE, 'w') as outfile:
        count = 0
        for line in infile:
            if "Gaze point:" in line:
                try:
                    # Split the line to get coordinates
                    prefix, gaze_part = line.split("Gaze point: ")
                    coords_str = gaze_part.strip(" []\n")
                    x_str, y_str = coords_str.split(",")
                    
                    x = float(x_str)
                    y = float(y_str)

                    # --- APPLY CORRECTION ---
                    new_y = y + Y_OFFSET
                    # ------------------------

                    # Reconstruct the line
                    new_line = f"{prefix}Gaze point: [{x}, {new_y}]\n"
                    outfile.write(new_line)
                    count += 1
                except ValueError:
                    outfile.write(line) # Write broken lines as-is
            else:
                outfile.write(line) # Write header/other lines as-is

    print(f"Done! Fixed {count} lines.")
    print(f"Saved to: {OUTPUT_FILE}")
    print("Now, rename 'gazeData_fixed.txt' to 'gazeData_calibrated.txt' (backup the old one first!) and run Playback.")

if __name__ == "__main__":
    repair_gaze_file()