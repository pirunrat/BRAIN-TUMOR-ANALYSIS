# frontend/display_panel.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QTabWidget, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from skimage import io, exposure
import cv2
from utils.utils import apply_segmentation, display_slice
from .slice_viewer import SliceViewer

class DisplayPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        

    def init_display(self):
        """Initialize display panel widgets"""
        # Right panel for display
        self.display_panel = QWidget()
        self.display_layout = QVBoxLayout()
        self.display_layout.setContentsMargins(20, 20, 20, 20)
        self.display_layout.setSpacing(20)
        self.display_panel.setLayout(self.display_layout)
        self.main_window.main_layout.addWidget(self.display_panel)

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

      

        # Add Multi-Planar tab
        self.view_tabs.addTab(self.multi_planar_tab, "Multi-Planar")

        # Replace QLabel with SliceViewer for each plane
        self.axial_view = SliceViewer("axial", self.main_window.sidebar.event_handler)
        self.coronal_view = SliceViewer("coronal", self.main_window.sidebar.event_handler)
        self.sagittal_view = SliceViewer("sagittal", self.main_window.sidebar.event_handler)

        for view in [self.axial_view, self.coronal_view, self.sagittal_view]:
            view.setAlignment(Qt.AlignCenter)
            view.setMinimumSize(300, 300)
            view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            view.setStyleSheet("background-color: black; border-radius: 5px;")

        self.multi_planar_layout.addWidget(self.axial_view, 1, 0)
        self.multi_planar_layout.addWidget(self.coronal_view, 1, 1)
        self.multi_planar_layout.addWidget(self.sagittal_view, 1, 2)


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

    def create_display_widget(self):
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(300, 300)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label.setStyleSheet("background-color: black; border-radius: 5px;")
        return label

    def update_all_views(self):
        backend = self.main_window.backend
        current = self.main_window.current_slice

        if not hasattr(backend, 'volume_data') or backend.volume_data is None:
            return

        axial_slice = backend.volume_data[current['axial'], :, :]
        coronal_slice = backend.volume_data[:, current['coronal'], :].T
        sagittal_slice = backend.volume_data[:, :, current['sagittal']].T

        if hasattr(backend, 'segmentation_masks') and backend.segmentation_masks is not None:
            axial_mask = backend.segmentation_masks[current['axial'], :, :]
            coronal_mask = backend.segmentation_masks[:, current['coronal'], :].T
            sagittal_mask = backend.segmentation_masks[:, :, current['sagittal']].T

            axial_slice = apply_segmentation(axial_slice, axial_mask)
            coronal_slice = apply_segmentation(coronal_slice, coronal_mask)
            sagittal_slice = apply_segmentation(sagittal_slice, sagittal_mask)

        # display_slice(axial_slice, self.axial_display)
        # display_slice(coronal_slice, self.coronal_display)
        # display_slice(sagittal_slice, self.sagittal_display)

        display_slice(axial_slice, self.axial_view)
        display_slice(coronal_slice, self.coronal_view)
        display_slice(sagittal_slice, self.sagittal_view)
    
    def convert_to_uint8(self, img):
        if img.dtype != np.uint8:
            # Normalize to [0, 255] and convert to uint8
            img = (255 * (img - np.min(img)) / (np.max(img) - np.min(img))).astype(np.uint8)
        return img
    
    def image_normalization(self, img):
        img_max = np.max(img)
        img_min = np.min(img)
        img_range = img_max - img_min
        if img_range > 0:
            img = (img - img_min)/(img_range) 
        else:
            img = np.zeros_like(img)
        return img.astype(np.float32)
    
    def display_2d_image(self, path):
        self.view_tabs.setCurrentWidget(self.single_tab)

        img = io.imread(path, as_gray=True)
        img = self.convert_to_uint8(img)
        img = self.image_normalization(img)
        img_rgb = np.stack([img] * 3, axis=-1)

        backend = self.main_window.backend
        if hasattr(backend, 'segmentation_masks') and backend.segmentation_masks is not None:
            mask = backend.segmentation_masks
            if mask.shape != img.shape:
                mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            red_overlay = np.zeros_like(img_rgb)
            red_overlay[..., 0] = 1.0
            alpha = 0.4
            mask_bool = mask.astype(bool)
            img_rgb[mask_bool] = (1 - alpha) * img_rgb[mask_bool] + alpha * red_overlay[mask_bool]

        img_8bit = (img_rgb * 255).astype(np.uint8)
        qimg = QImage(img_8bit.data, img_8bit.shape[1], img_8bit.shape[0], img_8bit.shape[1] * 3, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg).scaled(
            self.single_display.width(),
            self.single_display.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.single_display.setPixmap(pixmap)
        self.main_window.sidebar.volume_info.setText("Loaded 2D image")
