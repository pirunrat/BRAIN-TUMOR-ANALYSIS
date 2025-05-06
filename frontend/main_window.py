from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QMessageBox
from frontend.style import setup_styles
from backend.backend import Backend
from PyQt5.QtCore import Qt, QSize, QTimer, QThread
from .Config import PALLETTE
from .sidebar import SidebarPanel
from .display_panel import DisplayPanel
from utils.utils import apply_segmentation, display_slice


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
        self.sidebar = SidebarPanel(self)
        self.display_panel = DisplayPanel(self)
        self.sidebar.init_sidebar()
        self.display_panel.init_display()
        
        # Image data storage
        self.current_slice = {
            'axial': 0,
            'coronal': 0,
            'sagittal': 0
        }
    def setup_styles(self):
        setup_styles(self, self.dark_palette)
    
    def update_progress(self, value):
        """Update progress bar"""
        self.sidebar.progress_bar.setValue(value)


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
                self.sidebar.axial_slider.setMaximum(dims[0] - 1)
                self.sidebar.coronal_slider.setMaximum(dims[1] - 1)
                self.sidebar.sagittal_slider.setMaximum(dims[2] - 1)

                self.sidebar.axial_slider.setValue(self.current_slice['axial'])
                self.sidebar.coronal_slider.setValue(self.current_slice['coronal'])
                self.sidebar.sagittal_slider.setValue(self.current_slice['sagittal'])

                # Show appropriate UI
                self.sidebar.slice_control_group.setVisible(True)
                self.display_panel.view_tabs.setVisible(True)
                self.display_panel.view_tabs.setCurrentWidget(self.display_panel.multi_planar_tab)

                self.sidebar.volume_info.setText(f"Loaded: {dims[2]}×{dims[1]}×{dims[0]} volume\nVoxel size: {dims[2]}×{dims[1]}×{dims[0]}")
                self.update_all_views()

            elif len(dims) == 2:
                # This is a 2D image (e.g., JPG/PNG)
                self.sidebar.slice_control_group.setVisible(False)
                self.display_panel.view_tabs.setVisible(True)
                self.display_panel.view_tabs.setCurrentWidget(self.display_panel.single_tab)

                self.sidebar.volume_info.setText(f"Loaded 2D image: {dims[1]}×{dims[0]}")
                self.display_panel.display_2d_image(self.backend.image_2d_path)

            else:
                self.show_error("Unsupported image format or dimension.")

        self.sidebar.load_button.setEnabled(True)
        self.sidebar.segment_button.setEnabled(True)
        self.sidebar.classify_button.setEnabled(True)
        QTimer.singleShot(1000, lambda:  self.sidebar.progress_bar.setValue(0))
    

    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Error", message)
        self.sidebar.load_button.setEnabled(True)
        self.sidebar.segment_button.setEnabled(True)
        self.sidebar.classify_button.setEnabled(True)
        self.sidebar.progress_bar.setValue(0)
    

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
        display_slice(axial_slice, self.display_panel.axial_view)
        display_slice(coronal_slice, self.display_panel.coronal_view)
        display_slice(sagittal_slice, self.display_panel.sagittal_view)
