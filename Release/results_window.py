import os
import re
import numpy as np
import pandas as pd
from datetime import datetime

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QMessageBox, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from config import app_config

# --- ANALYSIS LOGIC ---
class GazeAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_data = self._load_data()
        self.fixations = pd.DataFrame()
        self.saccades = pd.DataFrame()

    def _load_data(self):
        data = []
        pattern = re.compile(r'\[(.*?)\] Gaze point: \[(.*?), (.*?)\]')
        try:
            with open(self.file_path, 'r') as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        ts_str, x_str, y_str = match.groups()
                        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f").timestamp()
                        data.append({'time': ts, 'x': float(x_str), 'y': float(y_str)})
            df = pd.DataFrame(data)
            if not df.empty:
                df['time'] = df['time'] - df.iloc[0]['time']
            return df
        except Exception:
            return pd.DataFrame()

    def run_analysis(self):
        if self.raw_data.empty: return None
        self._detect_fixations()
        self._detect_saccades()
        return self._calculate_metrics()

    def _detect_fixations(self, dispersion=0.05, duration_min=0.1):
        points = self.raw_data.to_dict('records')
        fixations = []
        i = 0
        while i < len(points):
            j = i + 1
            while j < len(points):
                dt = points[j]['time'] - points[i]['time']
                window = points[i:j+1]
                dx = max(p['x'] for p in window) - min(p['x'] for p in window)
                dy = max(p['y'] for p in window) - min(p['y'] for p in window)
                if (dx + dy) > dispersion:
                    if dt >= duration_min:
                        fixations.append({
                            'start': points[i]['time'],
                            'end': points[j-1]['time'],
                            'dur': dt,
                            'x': np.mean([p['x'] for p in window]),
                            'y': np.mean([p['y'] for p in window])
                        })
                        i = j
                    else:
                        i += 1
                    break
                else:
                    j += 1
            else:
                break
        self.fixations = pd.DataFrame(fixations)

    def _detect_saccades(self):
        if self.fixations.empty: return
        saccades = []
        for i in range(1, len(self.fixations)):
            prev, curr = self.fixations.iloc[i-1], self.fixations.iloc[i]
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            
            # Logic for saccade type
            sType = 'noise'
            if abs(dy) < 0.15:
                if dx > 0.02: sType = 'forward'
                elif dx < -0.02: sType = 'regression'
            elif dy < -0.2: sType = 'line_return' 
            
            saccades.append({'type': sType, 'dx': dx, 'dy': dy, 'dist': np.sqrt(dx**2+dy**2)})
        self.saccades = pd.DataFrame(saccades)

    def _calculate_metrics(self):
        if self.fixations.empty or self.saccades.empty: return None
        
        avg_fix = self.fixations['dur'].mean()
        counts = self.saccades['type'].value_counts()
        
        # Safe division
        total_reading_moves = counts.get('forward', 0) + counts.get('regression', 0)
        reg_rate = counts.get('regression', 0) / (total_reading_moves + 1e-6)
        
        # Std Dev of saccades
        fwd_saccades = self.saccades[self.saccades['type']=='forward']
        saccade_std = fwd_saccades['dist'].std() if not fwd_saccades.empty else 0
        
        # --- TUNED SCORING FORMULA ---
        score = (15 * avg_fix) + (20 * reg_rate) + (10 * saccade_std)
        
        # --- INTERPRETATION RANGES ---
        
        # 1. Regression Rate Logic
        if reg_rate < 0.15:
            reg_status = "Normal Range"
        elif reg_rate < 0.25:
            reg_status = "Moderate (Monitor)"
        else:
            reg_status = "High (Difficulty Indicator)"
        
        # Adding the reference line
        reg_desc = f"{reg_status}\n[Ref: Normal < 15% | High > 25%]"

        # 2. Fixation Logic
        if avg_fix < 0.22:
            fix_status = "Normal (Fast Processing)"
        elif avg_fix < 0.32:
            fix_status = "Moderate (Slower Decoding)"
        else:
            fix_status = "High (Processing Delay)"

        fix_desc = f"{fix_status}\n[Ref: Normal < 0.22s | High > 0.32s]"

        # 3. Score Logic
        if score < 5.0:
            score_status = "Low Risk (Fluent)"
        elif score < 7.0:
            score_status = "Moderate Risk"
        else:
            score_status = "High Risk"

        score_desc = f"{score_status}\n[Ref: Low < 5.0 | High > 7.0]"

        return {
            "Average Fixation": (avg_fix, "s", fix_desc),
            "Regression Rate": (reg_rate, "%", reg_desc),
            "Dyslexia Risk Score": (score, "", score_desc)
        }

# --- RESULTS WINDOW UI ---
class ResultsWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Results")
        self.resize(1400, 1000) # Start bigger
        self.initUI()
        self.analyze_current_session()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Header
        header = QHBoxLayout()
        self.title_label = QLabel("Session Analysis Results")
        self.title_label.setFont(QFont("Arial", 16, QFont.Bold))
        header.addWidget(self.title_label)
        header.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.setFixedSize(100, 40)
        self.close_btn.clicked.connect(self.close)
        header.addWidget(self.close_btn)
        layout.addLayout(header)

        # Metrics Section
        self.metrics_frame = QFrame()
        self.metrics_frame.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ddd;")
        self.metrics_layout = QHBoxLayout(self.metrics_frame)
        
        # STRETCH FACTOR 1: Metrics get small space
        layout.addWidget(self.metrics_frame, 1)

        # SPACE 1: Add vertical spacing between Metrics and Graphs
        layout.addSpacing(30)

        # Graph Canvas
        self.figure = Figure(figsize=(10, 12)) 
        self.canvas = FigureCanvas(self.figure)
        
        # STRETCH FACTOR 5: Graphs get big space
        layout.addWidget(self.canvas, 5)

    def analyze_current_session(self):
        directory = app_config.session_directory
        if not directory:
            self.title_label.setText("No Session Selected")
            return

        file_path = os.path.join(directory, 'gazeData_calibrated.txt')
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Error", "No calibrated data found in this session.")
            return

        analyzer = GazeAnalyzer(file_path)
        metrics = analyzer.run_analysis()

        if metrics:
            self.display_metrics(metrics)
            self.draw_graphs(analyzer)
            self.auto_save_results(metrics, directory)
        else:
             self.title_label.setText("Not enough data to analyze")

    def display_metrics(self, metrics):
        # Clear existing items safely
        while self.metrics_layout.count():
            child = self.metrics_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for k, (val, unit, desc) in metrics.items():
            if unit == "%":
                val_text = f"{val*100:.1f}%"
            else:
                val_text = f"{val:.4f} {unit}"

            # Create Card Container
            container = QFrame()
            container.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 8px;")
            vbox = QVBoxLayout(container)
            
            # Title
            lbl_title = QLabel(k)
            lbl_title.setFont(QFont("Arial", 11, QFont.Bold))
            lbl_title.setStyleSheet("color: #333; border: none;")
            lbl_title.setAlignment(Qt.AlignCenter)
            
            # Value
            lbl_val = QLabel(val_text)
            lbl_val.setFont(QFont("Arial", 18, QFont.Bold))
            lbl_val.setStyleSheet("color: #000; border: none;")
            lbl_val.setAlignment(Qt.AlignCenter)
            
            # Description (Status + Reference)
            lbl_desc = QLabel(desc)
            lbl_desc.setFont(QFont("Arial", 9))
            lbl_desc.setStyleSheet("color: #666; border: none;")
            lbl_desc.setAlignment(Qt.AlignCenter)
            lbl_desc.setWordWrap(True)

            vbox.addWidget(lbl_title)
            vbox.addWidget(lbl_val)
            vbox.addWidget(lbl_desc)
            vbox.addStretch()

            self.metrics_layout.addWidget(container)

    def draw_graphs(self, analyzer):
        self.figure.clear()
        
        # --- Scanpath Plot ---
        ax1 = self.figure.add_subplot(211)
        ax1.set_title("Scanpath (Spatial Reading Pattern)", fontweight='bold')
        ax1.invert_yaxis()
        
        if not analyzer.fixations.empty:
            ax1.scatter(analyzer.fixations['x'], analyzer.fixations['y'], 
                       s=analyzer.fixations['dur']*800, alpha=0.4, c='blue', label='Fixation (Size=Duration)')
            
            for i, row in analyzer.saccades.iterrows():
                start = analyzer.fixations.iloc[i]
                end = analyzer.fixations.iloc[i+1]
                color = 'red' if row['type'] == 'regression' else 'green'
                alpha = 0.5 if row['type'] == 'regression' else 0.15
                ax1.plot([start['x'], end['x']], [start['y'], end['y']], color=color, alpha=alpha, linewidth=1)
            
            ax1.legend(loc='upper right', fontsize='small')

        # --- Timeline Plot ---
        ax2 = self.figure.add_subplot(212)
        ax2.set_title("Reading Timeline (Rhythm & Stability)", fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Horizontal Position (Left → Right)")
        
        ax2.plot(analyzer.raw_data['time'], analyzer.raw_data['x'], color='gray', alpha=0.3, label='Raw Gaze')
        if not analyzer.fixations.empty:
            ax2.plot(analyzer.fixations['end'], analyzer.fixations['x'], 'o-', color='navy', markersize=3, linewidth=1, label='Fixations')
        
        ax2.legend(loc='upper left')
        
        # SPACE 2: Graph Spacing
        # 'h_pad' adds height padding between the two graphs (avoids label clipping)
        # 'pad' adds padding around the entire figure (avoids edge clipping)
        self.figure.tight_layout(pad=3.0, h_pad=4.0)
        self.canvas.draw()

    def auto_save_results(self, metrics, directory):
        file_path = os.path.join(directory, "analysis_results.txt")
        try:
            with open(file_path, "w") as f:
                f.write("=== DYSLEXIA SCREENING ANALYSIS ===\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 35 + "\n\n")
                
                for k, (val, unit, desc) in metrics.items():
                    if unit == "%":
                        val_str = f"{val*100:.2f}%"
                    else:
                        val_str = f"{val:.4f} {unit}"
                        
                    f.write(f"{k}: {val_str}\n")
                    f.write(f"   -> {desc.replace(chr(10), ' | ')}\n\n")
                
                f.write("=" * 35 + "\n")
                f.write("NOTE: This is a behavioral screening tool, not a medical diagnosis.\n")
            
            print(f"Results automatically saved to: {file_path}")
        except Exception as e:
            print(f"Failed to autosave: {e}")