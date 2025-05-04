# === frontend/main_window.py ===
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QFileDialog, QComboBox, QMessageBox, QFrame, QSizePolicy, QProgressBar, QTabWidget, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QSize, QTimer, QThread

from backend.segmentation.segmentation import simulate_segmentation
from backend.classification.classification import simulate_classification
from utils.utils import load_volume_data, apply_segmentation, display_slice

class MultiPlaneBrainTumorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroVision AI - Multi-Planar Tumor Analysis")
        self.setGeometry(100, 100, 1600, 1000)

        self.dark_palette = {
            'background': '#1e1e2e',
            'foreground': '#ffffff',
            'primary': '#6c5ce7',
            'secondary': '#a29bfe',
            'accent': '#fd79a8',
            'card': '#2d3436',
            'text': '#dfe6e9',
            'success': '#00b894',
            'warning': '#fdcb6e',
            'danger': '#d63031'
        }

        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {self.dark_palette['background']}; color: {self.dark_palette['foreground']}; }}
            QPushButton {{ background-color: {self.dark_palette['primary']}; color: white; border-radius: 5px; padding: 10px; }}
            QLabel {{ color: {self.dark_palette['text']}; font-size: 14px; }}
        """)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(350)
        self.sidebar.setStyleSheet(f"background-color: {self.dark_palette['card']}; border-radius: 10px;")
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar.setLayout(self.sidebar_layout)

        self.display_panel = QWidget()
        self.display_layout = QVBoxLayout(self.display_panel)
        self.display_panel.setLayout(self.display_layout)

        self.main_layout.addWidget(self.sidebar)
        self.main_layout.addWidget(self.display_panel)

        self.volume_data = None
        self.segmentation_masks = None
        self.current_slice = {'axial': 0, 'coronal': 0, 'sagittal': 0}

        self.init_sidebar()
        self.init_display()

    def init_sidebar(self):
        title = QLabel("NeuroVision AI")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {self.dark_palette['primary']};")
        self.sidebar_layout.addWidget(title)

        self.load_button = QPushButton("Load MRI Volume")
        self.load_button.clicked.connect(self.load_volume)
        self.sidebar_layout.addWidget(self.load_button)

        self.segment_button = QPushButton("Segment Tumor")
        self.segment_button.clicked.connect(self.segment_tumor)
        self.segment_button.setEnabled(False)
        self.sidebar_layout.addWidget(self.segment_button)

        self.classify_button = QPushButton("Classify Tumor")
        self.classify_button.clicked.connect(self.classify_tumor)
        self.classify_button.setEnabled(False)
        self.sidebar_layout.addWidget(self.classify_button)

        self.result_display = QLabel("No analysis performed")
        self.result_display.setWordWrap(True)
        self.sidebar_layout.addWidget(self.result_display)

        self.progress_bar = QProgressBar()
        self.sidebar_layout.addWidget(self.progress_bar)

    def init_display(self):
        self.axial_display = QLabel("Axial View")
        self.coronal_display = QLabel("Coronal View")
        self.sagittal_display = QLabel("Sagittal View")
        for display in [self.axial_display, self.coronal_display, self.sagittal_display]:
            display.setMinimumSize(300, 300)
            display.setStyleSheet("background-color: black;")
            display.setAlignment(Qt.AlignCenter)
            self.display_layout.addWidget(display)

    def load_volume(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open MRI Volume", "", "NIfTI Files (*.nii *.nii.gz);;Image Files (*.png *.jpg)")
        if file_name:
            self.volume_data = load_volume_data(file_name)
            self.current_slice = {k: v // 2 for k, v in zip(['axial', 'coronal', 'sagittal'], self.volume_data.shape)}
            self.update_views()
            self.segment_button.setEnabled(True)
            self.classify_button.setEnabled(True)

    def segment_tumor(self):
        self.segmentation_masks = simulate_segmentation(self.volume_data)
        self.update_views()

    def classify_tumor(self):
        tumor_types, probabilities = simulate_classification()
        result_text = "".join([f"{t}: {p*100:.2f}%\n" for t, p in zip(tumor_types, probabilities)])
        self.result_display.setText(result_text)

    def update_views(self):
        views = {
            'axial': self.volume_data[self.current_slice['axial'], :, :],
            'coronal': self.volume_data[:, self.current_slice['coronal'], :].T,
            'sagittal': self.volume_data[:, :, self.current_slice['sagittal']].T
        }
        for plane, view in views.items():
            if self.segmentation_masks is not None:
                mask = {
                    'axial': self.segmentation_masks[self.current_slice['axial'], :, :],
                    'coronal': self.segmentation_masks[:, self.current_slice['coronal'], :].T,
                    'sagittal': self.segmentation_masks[:, :, self.current_slice['sagittal']].T
                }[plane]
                view = apply_segmentation(view, mask)
            display_slice(view, getattr(self, f"{plane}_display"))