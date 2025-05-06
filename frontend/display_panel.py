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

        # # Axial view
        # self.axial_label = QLabel("Axial View")
        # self.axial_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        # self.axial_label.setAlignment(Qt.AlignCenter)
        # self.multi_planar_layout.addWidget(self.axial_label, 0, 0)

        # self.axial_display = QLabel()
        # self.axial_display.setAlignment(Qt.AlignCenter)
        # self.axial_display.setMinimumSize(300, 300)
        # self.axial_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.axial_display.setStyleSheet("background-color: black; border-radius: 5px;")
        # self.multi_planar_layout.addWidget(self.axial_display, 1, 0)

        # # Coronal view
        # self.coronal_label = QLabel("Coronal View")
        # self.coronal_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        # self.coronal_label.setAlignment(Qt.AlignCenter)
        # self.multi_planar_layout.addWidget(self.coronal_label, 0, 1)

        # self.coronal_display = QLabel()
        # self.coronal_display.setAlignment(Qt.AlignCenter)
        # self.coronal_display.setMinimumSize(300, 300)
        # self.coronal_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.coronal_display.setStyleSheet("background-color: black; border-radius: 5px;")
        # self.multi_planar_layout.addWidget(self.coronal_display, 1, 1)

        # # Sagittal view
        # self.sagittal_label = QLabel("Sagittal View")
        # self.sagittal_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        # self.sagittal_label.setAlignment(Qt.AlignCenter)
        # self.multi_planar_layout.addWidget(self.sagittal_label, 0, 2)

        # self.sagittal_display = QLabel()
        # self.sagittal_display.setAlignment(Qt.AlignCenter)
        # self.sagittal_display.setMinimumSize(300, 300)
        # self.sagittal_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.sagittal_display.setStyleSheet("background-color: black; border-radius: 5px;")
        # self.multi_planar_layout.addWidget(self.sagittal_display, 1, 2)

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

    def display_2d_image(self, path):
        self.view_tabs.setCurrentWidget(self.single_tab)

        img = io.imread(path, as_gray=True)
        img = exposure.rescale_intensity(img)
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
