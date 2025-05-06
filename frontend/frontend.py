import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QComboBox, QMessageBox,
                             QFrame, QSizePolicy, QProgressBar, QTabWidget, QGridLayout)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QColor
from PyQt5.QtCore import Qt, QSize, QTimer, QThread
from backend.backend import Backend
from utils.utils import apply_segmentation, display_slice
from skimage import io, exposure
import os
import numpy as np
import cv2
from .Config import PALLETTE, STYLE_TEMPLATE


class MultiPlaneBrainTumorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroVision AI - Multi-Planar Tumor Analysis")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize backend
        self.backend = Backend()
        self.backend_thread = QThread()
        self.backend.moveToThread(self.backend_thread)
        self.backend_thread.start()
        
        # Connect backend signals
        self.backend.progress_updated.connect(self.update_progress)
        self.backend.processing_complete.connect(self.on_processing_complete)
        self.backend.error_occurred.connect(self.show_error)
        
        # Dark theme palette
        self.dark_palette = PALLETTE
        
        # Apply styles
        self.setup_styles()
        
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        
        # Initialize UI components
        self.init_sidebar()
        self.init_display()
        
        # Image data storage
        self.current_slice = {
            'axial': 0,
            'coronal': 0,
            'sagittal': 0
        }
    
    def setup_styles(self):
        """Setup the application styles"""
        stylesheet = STYLE_TEMPLATE.format(**self.dark_palette)
        self.setStyleSheet(stylesheet)

   
        
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
        
        self.main_layout.addWidget(self.sidebar)
        
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
        self.axial_slider.mousePressEvent = lambda e: self.handle_slice_click(e, 'axial')
        self.axial_slider.mouseMoveEvent = lambda e: self.handle_slice_click(e, 'axial')
        slice_layout.addWidget(self.axial_slider)
        
        # Coronal slice control
        self.coronal_slider_label = QLabel("Coronal Slice: 0")
        slice_layout.addWidget(self.coronal_slider_label)
        self.coronal_slider = QProgressBar()
        self.coronal_slider.setRange(0, 100)
        self.coronal_slider.setValue(0)
        self.coronal_slider.setTextVisible(False)
        self.coronal_slider.mousePressEvent = lambda e: self.handle_slice_click(e, 'coronal')
        self.coronal_slider.mouseMoveEvent = lambda e: self.handle_slice_click(e, 'coronal')
        slice_layout.addWidget(self.coronal_slider)
        
        # Sagittal slice control
        self.sagittal_slider_label = QLabel("Sagittal Slice: 0")
        slice_layout.addWidget(self.sagittal_slider_label)
        self.sagittal_slider = QProgressBar()
        self.sagittal_slider.setRange(0, 100)
        self.sagittal_slider.setValue(0)
        self.sagittal_slider.setTextVisible(False)
        self.sagittal_slider.mousePressEvent = lambda e: self.handle_slice_click(e, 'sagittal')
        self.sagittal_slider.mouseMoveEvent = lambda e: self.handle_slice_click(e, 'sagittal')
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
        
    

    def init_display(self):
        """Initialize display panel widgets"""
        # Right panel for display
        self.display_panel = QWidget()
        self.display_layout = QVBoxLayout()
        self.display_layout.setContentsMargins(20, 20, 20, 20)
        self.display_layout.setSpacing(20)
        self.display_panel.setLayout(self.display_layout)
        self.main_layout.addWidget(self.display_panel)

        # === Tab Widget ===
        self.view_tabs = QTabWidget()
        self.view_tabs.setStyleSheet("""
            QTabBar::tab { min-width: 120px; }
        """)
        self.display_layout.addWidget(self.view_tabs)

        # === Tab 1: Multi-Planar Views ===
        self.multi_planar_tab = QWidget()
        self.multi_planar_layout = QGridLayout()
        self.multi_planar_layout.setContentsMargins(10, 10, 10, 10)
        self.multi_planar_layout.setSpacing(15)
        self.multi_planar_tab.setLayout(self.multi_planar_layout)

        # Configure column stretch
        self.multi_planar_layout.setColumnStretch(0, 1)
        self.multi_planar_layout.setColumnStretch(1, 1)
        self.multi_planar_layout.setColumnStretch(2, 1)

        # Axial view
        self.axial_label = QLabel("Axial View")
        self.axial_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.axial_label.setAlignment(Qt.AlignCenter)
        self.multi_planar_layout.addWidget(self.axial_label, 0, 0)

        self.axial_display = QLabel()
        self.axial_display.setAlignment(Qt.AlignCenter)
        self.axial_display.setMinimumSize(300, 300)
        self.axial_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axial_display.setStyleSheet("background-color: black; border-radius: 5px;")
        self.multi_planar_layout.addWidget(self.axial_display, 1, 0)

        # Coronal view
        self.coronal_label = QLabel("Coronal View")
        self.coronal_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.coronal_label.setAlignment(Qt.AlignCenter)
        self.multi_planar_layout.addWidget(self.coronal_label, 0, 1)

        self.coronal_display = QLabel()
        self.coronal_display.setAlignment(Qt.AlignCenter)
        self.coronal_display.setMinimumSize(300, 300)
        self.coronal_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.coronal_display.setStyleSheet("background-color: black; border-radius: 5px;")
        self.multi_planar_layout.addWidget(self.coronal_display, 1, 1)

        # Sagittal view
        self.sagittal_label = QLabel("Sagittal View")
        self.sagittal_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.sagittal_label.setAlignment(Qt.AlignCenter)
        self.multi_planar_layout.addWidget(self.sagittal_label, 0, 2)

        self.sagittal_display = QLabel()
        self.sagittal_display.setAlignment(Qt.AlignCenter)
        self.sagittal_display.setMinimumSize(300, 300)
        self.sagittal_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sagittal_display.setStyleSheet("background-color: black; border-radius: 5px;")
        self.multi_planar_layout.addWidget(self.sagittal_display, 1, 2)

        # Add Multi-Planar tab
        self.view_tabs.addTab(self.multi_planar_tab, "Multi-Planar")

        # === Tab 2: Single Image View ===
        self.single_tab = QWidget()
        self.single_tab_layout = QVBoxLayout()
        self.single_tab.setLayout(self.single_tab_layout)

        self.single_display = QLabel()
        self.single_display.setAlignment(Qt.AlignCenter)
        self.single_display.setMinimumSize(600, 600)
        self.single_display.setStyleSheet("background-color: black; border-radius: 5px;")
        self.single_tab_layout.addWidget(self.single_display)

        self.view_tabs.addTab(self.single_tab, "Single View")

    def add_divider(self, layout):
        """Add a styled divider to the layout"""
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet(f"color: {self.dark_palette['secondary']};")
        layout.addWidget(divider)
        
    def load_volume(self):
        """Handle volume loading"""
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
                self.backend.load_image_2d(file_name)
                self.display_2d_image(file_name)
            else:
                self.view_tabs.setVisible(True)
                self.view_tabs.setCurrentWidget(self.multi_planar_tab)
                self.backend.load_volume(file_name)

    

   

    def display_2d_image(self, path):
        self.view_tabs.setCurrentWidget(self.single_tab)  # Switch to single view tab

        # Load and normalize grayscale image
        img = io.imread(path, as_gray=True)
        img = exposure.rescale_intensity(img)  # scale to [0, 1]

        # Convert to RGB
        img_rgb = np.stack([img] * 3, axis=-1)  # shape: (H, W, 3)

        # If segmentation mask exists
        if hasattr(self.backend, 'segmentation_masks') and self.backend.segmentation_masks is not None:
            mask = self.backend.segmentation_masks
            if mask.shape != img.shape:
                mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Create transparent red overlay
            red_overlay = np.zeros_like(img_rgb)
            red_overlay[..., 0] = 1.0  # Red channel

            alpha = 0.4  # Transparency level
            mask_bool = mask.astype(bool)
            img_rgb[mask_bool] = (1 - alpha) * img_rgb[mask_bool] + alpha * red_overlay[mask_bool]

        # Convert to 8-bit and QImage
        img_8bit = (img_rgb * 255).astype(np.uint8)
        qimg = QImage(img_8bit.data, img_8bit.shape[1], img_8bit.shape[0], img_8bit.shape[1] * 3, QImage.Format_RGB888)

        # Show image
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.single_display.width(),
            self.single_display.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.single_display.setPixmap(pixmap)
        self.volume_info.setText("Loaded 2D image")



    def handle_slice_click(self, event, plane):
        """Handle click on slice progress bars to change slice"""
        if not hasattr(self.backend, 'volume_data') or self.backend.volume_data is None:
            return
        
        # Calculate slice based on click position
        slider = getattr(self, f"{plane}_slider")
        pos = event.pos().x()
        slice_pos = int((pos / slider.width()) * slider.maximum())
        
        # Update current slice
        self.current_slice[plane] = slice_pos
        slider.setValue(slice_pos)
        getattr(self, f"{plane}_slider_label").setText(f"{plane.capitalize()} Slice: {slice_pos}")
        
        # Update views
        self.update_all_views()
    
    def update_all_views(self):
        """Update all three views (axial, coronal, sagittal)"""
        if not hasattr(self.backend, 'volume_data') or self.backend.volume_data is None:
            return
            
        # Get slices for each plane
        axial_slice = self.backend.volume_data[self.current_slice['axial'], :, :]
        coronal_slice = self.backend.volume_data[:, self.current_slice['coronal'], :]
        sagittal_slice = self.backend.volume_data[:, :, self.current_slice['sagittal']]
        
        # Transpose coronal and sagittal for better display
        coronal_slice = coronal_slice.T
        sagittal_slice = sagittal_slice.T
        
        # Apply segmentation if available
        if hasattr(self.backend, 'segmentation_masks') and self.backend.segmentation_masks is not None:
            axial_mask = self.backend.segmentation_masks[self.current_slice['axial'], :, :]
            coronal_mask = self.backend.segmentation_masks[:, self.current_slice['coronal'], :].T
            sagittal_mask = self.backend.segmentation_masks[:, :, self.current_slice['sagittal']].T
            
            axial_slice = apply_segmentation(axial_slice, axial_mask)
            coronal_slice = apply_segmentation(coronal_slice, coronal_mask)
            sagittal_slice = apply_segmentation(sagittal_slice, sagittal_mask)
        
        # Display each slice
        display_slice(axial_slice, self.axial_display)
        display_slice(coronal_slice, self.coronal_display)
        display_slice(sagittal_slice, self.sagittal_display)
    
    def segment_tumor(self):
        self.segment_button.setEnabled(False)
        self.classify_button.setEnabled(False)
        self.backend.segment_tumor()
    
    def classify_tumor(self):
        """Handle tumor classification"""
        self.segment_button.setEnabled(False)
        self.classify_button.setEnabled(False)
        result = self.backend.classify_tumor()
        if result:
            self.display_classification_result(result)
    
    def display_classification_result(self, result):
        """Display classification results"""
        predicted_class = result["predicted_class"]
        probabilities = result["probabilities"]
        
        # Format results
        result_text = f"<b><font color='{self.dark_palette['primary']}'>Diagnosis:</font></b> {predicted_class}<br><br>"
        result_text += f"<b><font color='{self.dark_palette['primary']}'>Confidence Levels:</font></b><br>"
        
        for t, p in probabilities.items():
            color = self.dark_palette['success'] if t == predicted_class else self.dark_palette['text']
            result_text += f"<font color='{color}'>• {t}:</font> <b>{p*100:.1f}%</b><br>"
        
        self.result_display.setText(result_text)
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    
    

    def on_processing_complete(self):
        """Handle completion of backend processing"""
        if hasattr(self.backend, 'volume_data') and self.backend.volume_data is not None:
            dims = self.backend.volume_data.shape

            if len(dims) == 3:
                # This is a 3D volume (e.g., NIfTI)
                self.current_slice = {
                    'axial': dims[0] // 2,
                    'coronal': dims[1] // 2,
                    'sagittal': dims[2] // 2
                }

                # Set sliders
                self.axial_slider.setMaximum(dims[0] - 1)
                self.coronal_slider.setMaximum(dims[1] - 1)
                self.sagittal_slider.setMaximum(dims[2] - 1)

                self.axial_slider.setValue(self.current_slice['axial'])
                self.coronal_slider.setValue(self.current_slice['coronal'])
                self.sagittal_slider.setValue(self.current_slice['sagittal'])

                # Show appropriate UI
                self.slice_control_group.setVisible(True)
                self.view_tabs.setVisible(True)
                self.view_tabs.setCurrentWidget(self.multi_planar_tab)

                self.volume_info.setText(f"Loaded: {dims[2]}×{dims[1]}×{dims[0]} volume\nVoxel size: {dims[2]}×{dims[1]}×{dims[0]}")
                self.update_all_views()

            elif len(dims) == 2:
                # This is a 2D image (e.g., JPG/PNG)
                self.slice_control_group.setVisible(False)
                self.view_tabs.setVisible(True)
                self.view_tabs.setCurrentWidget(self.single_tab)

                self.volume_info.setText(f"Loaded 2D image: {dims[1]}×{dims[0]}")
                self.display_2d_image(self.backend.image_2d_path)

            else:
                self.show_error("Unsupported image format or dimension.")

        self.load_button.setEnabled(True)
        self.segment_button.setEnabled(True)
        self.classify_button.setEnabled(True)
        QTimer.singleShot(1000, lambda: self.progress_bar.setValue(0))


    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.load_button.setEnabled(True)
        self.segment_button.setEnabled(True)
        self.classify_button.setEnabled(True)
        self.progress_bar.setValue(0)

