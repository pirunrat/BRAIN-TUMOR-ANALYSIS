# === frontend/sidebar_panel.py ===
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
                             QProgressBar, QFrame)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtWidgets import QFileDialog
import os
from .even_handlers import EventHandler


class SidebarPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.dark_palette = self.main_window.dark_palette
        self.event_handler = EventHandler(main_window)
        

    def init_sidebar(self):
        """Initialize sidebar widgets"""
        # Left panel for controls (sidebar)
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(350)
        self.sidebar.setStyleSheet(f"background-color: {self.dark_palette['card']}; border-radius: 10px;")
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.setContentsMargins(20, 20, 20, 20)
        self.sidebar_layout.setSpacing(20)
        self.sidebar.setLayout(self.sidebar_layout)
        
        self.main_window.main_layout.addWidget(self.sidebar)
        
        # App title
        title = QLabel("NeuroVision AI")
        title.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {self.dark_palette['primary']};")
        title.setAlignment(Qt.AlignCenter)
        self.sidebar_layout.addWidget(title)
        
        # Divider
        self.add_divider(self.sidebar_layout)
        
        # Load image section
        load_section = QLabel("Volume Input")
        load_section.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.sidebar_layout.addWidget(load_section)
        
        # Load image button
        self.load_button = QPushButton("Load MRI Volume")
        self.load_button.setIcon(QIcon.fromTheme("document-open"))
        self.load_button.setIconSize(QSize(20, 20))
        self.load_button.clicked.connect(self.load_volume)
        self.sidebar_layout.addWidget(self.load_button)
        
        # Image info label
        self.volume_info = QLabel("No volume loaded")
        self.volume_info.setAlignment(Qt.AlignCenter)
        self.volume_info.setStyleSheet("font-style: italic;")
        self.sidebar_layout.addWidget(self.volume_info)
        
        # Slice controls
        self.slice_control_group = QWidget()
        slice_layout = QVBoxLayout()
        self.slice_control_group.setLayout(slice_layout)
        
        # Axial slice control
        self.axial_slider_label = QLabel("Axial Slice: 0")
        slice_layout.addWidget(self.axial_slider_label)
        self.axial_slider = QProgressBar()
        self.axial_slider.setRange(0, 100)
        self.axial_slider.setValue(0)
        self.axial_slider.setTextVisible(False)
        self.axial_slider.mousePressEvent = lambda e: self.event_handler .handle_slice_click(e, 'axial')
        self.axial_slider.mouseMoveEvent = lambda e: self.event_handler .handle_slice_click(e, 'axial')
        slice_layout.addWidget(self.axial_slider)
        
        # Coronal slice control
        self.coronal_slider_label = QLabel("Coronal Slice: 0")
        slice_layout.addWidget(self.coronal_slider_label)
        self.coronal_slider = QProgressBar()
        self.coronal_slider.setRange(0, 100)
        self.coronal_slider.setValue(0)
        self.coronal_slider.setTextVisible(False)
        self.coronal_slider.mousePressEvent = lambda e: self.event_handler .handle_slice_click(e, 'coronal')
        self.coronal_slider.mouseMoveEvent = lambda e: self.event_handler .handle_slice_click(e, 'coronal')
        slice_layout.addWidget(self.coronal_slider)
        
        # Sagittal slice control
        self.sagittal_slider_label = QLabel("Sagittal Slice: 0")
        slice_layout.addWidget(self.sagittal_slider_label)
        self.sagittal_slider = QProgressBar()
        self.sagittal_slider.setRange(0, 100)
        self.sagittal_slider.setValue(0)
        self.sagittal_slider.setTextVisible(False)
        self.sagittal_slider.mousePressEvent = lambda e: self.event_handler .handle_slice_click(e, 'sagittal')
        self.sagittal_slider.mouseMoveEvent = lambda e: self.event_handler .handle_slice_click(e, 'sagittal')
        slice_layout.addWidget(self.sagittal_slider)
        
        self.sidebar_layout.addWidget(self.slice_control_group)
        self.slice_control_group.setVisible(False)
        
        # Divider
        self.add_divider(self.sidebar_layout)
        
        # Processing section
        process_section = QLabel("Analysis Tools")
        process_section.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.sidebar_layout.addWidget(process_section)
        
        # Model selection
        self.model_label = QLabel("AI Model:")
        self.sidebar_layout.addWidget(self.model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["3D U-Net", "DeepMedic", "Multi-Planar CNN"])
        self.sidebar_layout.addWidget(self.model_combo)
        
        # Segmentation button
        self.segment_button = QPushButton("Segment Tumor")
        self.segment_button.setIcon(QIcon.fromTheme("edit-select"))
        self.segment_button.setIconSize(QSize(20, 20))
        self.segment_button.clicked.connect(self.segment_tumor)
        self.segment_button.setEnabled(False)
        self.sidebar_layout.addWidget(self.segment_button)
        
        # Classification button
        self.classify_button = QPushButton("Classify Tumor")
        self.classify_button.setIcon(QIcon.fromTheme("dialog-information"))
        self.classify_button.setIconSize(QSize(20, 20))
        self.classify_button.clicked.connect(self.classify_tumor)
        self.classify_button.setEnabled(False)
        self.sidebar_layout.addWidget(self.classify_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.sidebar_layout.addWidget(self.progress_bar)
        
        # Divider
        self.add_divider(self.sidebar_layout)
        
        # Results section
        results_section = QLabel("Analysis Results")
        results_section.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.sidebar_layout.addWidget(results_section)
        
        self.result_display = QLabel("No analysis performed")
        self.result_display.setAlignment(Qt.AlignLeft)
        self.result_display.setStyleSheet("""
            background-color: #2d3436;
            padding: 15px;
            border-radius: 8px;
            font-size: 13px;
            border-left: 4px solid #6c5ce7;
        """)
        self.result_display.setWordWrap(True)
        self.sidebar_layout.addWidget(self.result_display)
        
        # Add stretch to push everything up
        self.sidebar_layout.addStretch()
        
        # Footer
        footer = QLabel("© 2023 NeuroVision AI")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #636e72; font-size: 12px;")
        self.sidebar_layout.addWidget(footer)

    def add_divider(self, layout):
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"color: {self.dark_palette['secondary']};")
        layout.addWidget(divider)

    def load_volume(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open MRI Volume", "", 
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;NIfTI Files (*.nii *.nii.gz);;All Files (*)", 
            options=options
        )
        if file_name:
            self.load_button.setEnabled(False)
            ext = os.path.splitext(file_name)[1].lower()
            if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                self.main_window.backend.load_image_2d(file_name)
                self.main_window.display_panel.display_2d_image(file_name)
            else:
                self.main_window.display_panel.view_tabs.setVisible(True)
                self.main_window.display_panel.view_tabs.setCurrentWidget(self.main_window.display_panel.multi_planar_tab)
                self.main_window.backend.load_volume(file_name)

    def segment_tumor(self):
        self.segment_button.setEnabled(False)
        self.classify_button.setEnabled(False)
        self.main_window.backend.segment_tumor()

    def classify_tumor(self):
        self.segment_button.setEnabled(False)
        self.classify_button.setEnabled(False)
        result = self.main_window.backend.classify_tumor()
        if result:
            self.display_classification_result(result)

    def display_classification_result(self, result):
        predicted_class = result["predicted_class"]
        probabilities = result["probabilities"]

        result_text = f"<b><font color='{self.dark_palette['primary']}'>Diagnosis:</font></b> {predicted_class}<br><br>"
        result_text += f"<b><font color='{self.dark_palette['primary']}'>Confidence Levels:</font></b><br>"

        for t, p in probabilities.items():
            color = self.dark_palette['success'] if t == predicted_class else self.dark_palette['text']
            result_text += f"<font color='{color}'>• {t}:</font> <b>{p*100:.1f}%</b><br>"

        self.result_display.setText(result_text)

    def reset_buttons(self):
        self.load_button.setEnabled(True)
        self.segment_button.setEnabled(True)
        self.classify_button.setEnabled(True)

  
