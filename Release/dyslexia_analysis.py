import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class GazeAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = self._load_data()
        self.fixations = []
        self.saccades = []
        self.metrics = {}

    def _load_data(self):
        """Parses the gazeData_calibrated.txt file."""
        data = []
        # Regex to parse: [2026-01-01 20:13:49.898] Gaze point: [-0.37..., -0.11...]
        pattern = re.compile(r'\[(.*?)\] Gaze point: \[(.*?), (.*?)\]')
        
        try:
            with open(self.file_path, 'r') as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        ts_str, x_str, y_str = match.groups()
                        # Parse timestamp
                        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()
                        data.append({'time': ts, 'x': float(x_str), 'y': float(y_str)})
            
            df = pd.DataFrame(data)
            # Normalize time to start at 0
            if not df.empty:
                df['time'] = df['time'] - df.iloc[0]['time']
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def detect_fixations(self, dispersion_threshold=0.05, min_duration=0.100):
        """
        I-DT (Dispersion-Threshold) Algorithm.
        Groups raw points into fixations if they stay within a small area (dispersion) 
        for a minimum time (min_duration).
        """
        if self.raw_data.empty:
            return

        points = self.raw_data.to_dict('records')
        fixations = []
        
        i = 0
        while i < len(points):
            j = i + 1
            while j < len(points):
                # Check duration condition
                duration = points[j]['time'] - points[i]['time']
                
                # Check dispersion condition (bounding box)
                window = points[i:j+1]
                xs = [p['x'] for p in window]
                ys = [p['y'] for p in window]
                dispersion = (max(xs) - min(xs)) + (max(ys) - min(ys))

                if dispersion > dispersion_threshold:
                    # Dispersion broken. Check if the previous window was a valid fixation
                    if duration >= min_duration:
                        # Save fixation (centroid of points)
                        fixations.append({
                            'start_time': points[i]['time'],
                            'end_time': points[j-1]['time'],
                            'duration': points[j-1]['time'] - points[i]['time'],
                            'x': np.mean([p['x'] for p in points[i:j]]),
                            'y': np.mean([p['y'] for p in points[i:j]]),
                            'count': j - i
                        })
                        i = j # Move window forward
                    else:
                        i += 1 # Advance slightly to find new window
                    break
                else:
                    j += 1
            else:
                # End of data
                break
        
        self.fixations = pd.DataFrame(fixations)
        self._detect_saccades()

    def _detect_saccades(self):
        """Calculates movements (saccades) between fixations."""
        if self.fixations.empty:
            return

        saccades = []
        for i in range(1, len(self.fixations)):
            prev = self.fixations.iloc[i-1]
            curr = self.fixations.iloc[i]
            
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            dist = np.sqrt(dx**2 + dy**2)
            
            # Classification:
            # Forward: Moved Right (dx > 0) AND didn't change line significantly
            # Regression: Moved Left (dx < 0) AND didn't change line significantly
            # Line Return: Moved significantly Down (dy < -0.2) and Left
            
            saccade_type = "noise"
            if abs(dy) < 0.15: # Staying on roughly same line
                if dx > 0.02: saccade_type = "forward"
                elif dx < -0.02: saccade_type = "regression"
            elif dy < -0.2: # Significant drop (coordinate system might be inverted, verify Y!)
                 # Assuming Y=1 is Top, Y=-1 is Bottom -> Line change is Y decreasing? 
                 # Wait, usually text is read Top to Bottom.
                 # Let's check raw data logic. If Y is relative:
                 # If moving DOWN lines, Y should change. Let's assume standard behavior.
                 saccade_type = "line_return"

            saccades.append({
                'from_idx': i-1,
                'to_idx': i,
                'dx': dx,
                'dy': dy,
                'distance': dist,
                'duration': curr['start_time'] - prev['end_time'],
                'type': saccade_type
            })
            
        self.saccades = pd.DataFrame(saccades)

    def calculate_metrics(self):
        """Calculates the 5 Core Metrics for the 'Difficulty Score'."""
        if self.fixations.empty or self.saccades.empty:
            return
        
        # 1. Avg Fixation Duration (Seconds)
        avg_fix_duration = self.fixations['duration'].mean()
        
        # 2. Regression Rate (Regressions / (Forward + Regressions))
        counts = self.saccades['type'].value_counts()
        n_reg = counts.get('regression', 0)
        n_fwd = counts.get('forward', 0)
        total_reading_moves = n_reg + n_fwd
        regression_rate = n_reg / total_reading_moves if total_reading_moves > 0 else 0
        
        # 3. Saccade Length Variability (Std Dev of Forward Saccades)
        fwd_saccades = self.saccades[self.saccades['type'] == 'forward']
        saccade_var = fwd_saccades['distance'].std() if not fwd_saccades.empty else 0

        # 4. Line Transition Noise (Approximated by Y-variance during "reading" phases)
        # We look at Y variability when we are supposedly reading a line (type='forward' or 'regression')
        reading_saccades = self.saccades[self.saccades['type'].isin(['forward', 'regression'])]
        line_noise = reading_saccades['dy'].abs().mean() if not reading_saccades.empty else 0

        # 5. Reading Speed Stability (Variance of fixations per second, or speed)
        # Let's use: Variance of (Distance / Time) for forward saccades
        if not fwd_saccades.empty:
            speeds = fwd_saccades['distance'] / (fwd_saccades['duration'] + 0.001) # Avoid div/0
            speed_stability = speeds.std() # Higher = More Unstable
        else:
            speed_stability = 0

        # --- COMPOSITE SCORE ---
        # Weights (can be tuned based on literature)
        w1, w2, w3, w4, w5 = 10, 20, 5, 10, 5
        score = (w1 * avg_fix_duration) + (w2 * regression_rate) + \
                (w3 * saccade_var) + (w4 * line_noise) + (w5 * speed_stability)

        self.metrics = {
            "Avg Fixation Duration (s)": avg_fix_duration,
            "Regression Rate": regression_rate,
            "Saccade Length Var": saccade_var,
            "Line Noise (Y-Jitter)": line_noise,
            "Speed Instability": speed_stability,
            "READING DIFFICULTY SCORE": score
        }
        return self.metrics

    def visualize(self):
        """Generates the 'HCI Gold' Visualizations."""
        if self.fixations.empty:
            print("No fixations to visualize.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # --- Plot 1: Scanpath (Spatial) ---
        ax1.set_title("Scanpath: Gaze Sequence & Regressions")
        ax1.set_xlabel("X (Horizontal Position)")
        ax1.set_ylabel("Y (Vertical Position)")
        
        # Invert Y if necessary (assuming Top-Left origin usually, but your coords are -1 to 1)
        # If -1 is bottom and 1 is top, standard reading goes Top->Bottom (1 -> -1)
        # Let's assume standard math coords for now.
        
        # Plot fixations as circles (size = duration)
        sizes = self.fixations['duration'] * 1000
        ax1.scatter(self.fixations['x'], self.fixations['y'], s=sizes, alpha=0.5, c='blue', label='Fixation')
        
        # Draw Arrows
        for _, row in self.saccades.iterrows():
            start = self.fixations.loc[row['from_idx']]
            end = self.fixations.loc[row['to_idx']]
            
            color = 'green' if row['type'] == 'forward' else \
                    'red' if row['type'] == 'regression' else 'gray'
            
            alpha = 0.8 if row['type'] == 'regression' else 0.3
            width = 0.005 if row['type'] == 'regression' else 0.002
            
            ax1.arrow(start['x'], start['y'], 
                      end['x'] - start['x'], end['y'] - start['y'], 
                      head_width=0.02, length_includes_head=True, 
                      fc=color, ec=color, alpha=alpha, width=width)

        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)

        # --- Plot 2: Timeline (Temporal) ---
        ax2.set_title("Reading Timeline: X-Position over Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("X Position (Left -> Right)")
        
        ax2.plot(self.raw_data['time'], self.raw_data['x'], color='lightgray', label='Raw Gaze', alpha=0.5)
        ax2.plot(self.fixations['end_time'], self.fixations['x'], 'o-', color='blue', label='Fixation Path')
        
        # Highlight Regressions on Timeline
        regressions = self.saccades[self.saccades['type'] == 'regression']
        for _, reg in regressions.iterrows():
            t_start = self.fixations.loc[reg['from_idx']]['end_time']
            t_end = self.fixations.loc[reg['to_idx']]['start_time']
            ax2.axvspan(t_start, t_end, color='red', alpha=0.2, label='Regression Event' if 'Regression Event' not in ax2.get_legend_handles_labels()[1] else "")

        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Change this to your actual file path
    FILE_PATH = "gazeData_calibrated.txt" 
    
    analyzer = GazeAnalyzer(FILE_PATH)
    analyzer.detect_fixations()
    metrics = analyzer.calculate_metrics()
    
    print("-" * 30)
    print("ANALYSIS RESULTS")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k:<25}: {v:.4f}")
    print("-" * 30)
    
    analyzer.visualize()