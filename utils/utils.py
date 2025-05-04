# import numpy as np
# from skimage import exposure
# from PyQt5.QtGui import QImage, QPixmap
# from PyQt5.QtCore import Qt
# import cv2

# def apply_segmentation(image_slice, mask_slice):
#     # Resize the mask to match the image size
#     mask_resized = cv2.resize(mask_slice.astype(np.uint8), (image_slice.shape[1], image_slice.shape[0]), interpolation=cv2.INTER_NEAREST)

#     # Convert grayscale image to RGB if necessary
#     if image_slice.ndim == 2:
#         img_rgb = np.stack([image_slice]*3, axis=-1)
#     else:
#         img_rgb = image_slice

#     # Apply mask overlay (example: red channel highlights segmentation)
#     overlay = img_rgb.copy()
#     overlay[..., 0] = img_rgb[..., 0] * 0.7 + mask_resized * 0.3 * 255  # Red overlay
#     overlay = overlay.astype(np.uint8)

#     return overlay

# def display_slice(slice_data, display_widget):
#     """Display a single slice in the specified widget"""
#     if len(slice_data.shape) == 2:  # Grayscale
#         # Convert to 8-bit
#         slice_8bit = (exposure.rescale_intensity(slice_data) * 255).astype(np.uint8)
#         slice_8bit = np.ascontiguousarray(slice_8bit)
        
#         # Create QImage from numpy array
#         height, width = slice_8bit.shape
#         bytes_per_line = width  # For 8-bit grayscale
#         qimage = QImage(slice_8bit.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        
#         # Scale to fit display
#         pixmap = QPixmap.fromImage(qimage)
#         pixmap = pixmap.scaled(
#             display_widget.width(), 
#             display_widget.height(), 
#             Qt.KeepAspectRatio,
#             Qt.SmoothTransformation
#         )
        
#         display_widget.setPixmap(pixmap)
#     elif len(slice_data.shape) == 3:  # RGB
#         # Convert to 8-bit
#         slice_8bit = (exposure.rescale_intensity(slice_data) * 255).astype(np.uint8)
#         slice_8bit = np.ascontiguousarray(slice_8bit)
        
#         # Create QImage from numpy array
#         height, width, channels = slice_8bit.shape
#         bytes_per_line = 3 * width  # For RGB
#         qimage = QImage(slice_8bit.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
#         # Scale to fit display
#         pixmap = QPixmap.fromImage(qimage)
#         pixmap = pixmap.scaled(
#             display_widget.width(), 
#             display_widget.height(), 
#             Qt.KeepAspectRatio,
#             Qt.SmoothTransformation
#         )
        
#         display_widget.setPixmap(pixmap)



import numpy as np
from skimage import exposure
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2

import numpy as np

def apply_segmentation(image_slice, mask_slice, alpha=0.5):
    """
    Overlay the segmentation mask on top of the grayscale image.
    Red is used to highlight the segmentation.
    """
    if image_slice.ndim == 2:
        image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
        img_rgb = np.stack([image_slice]*3, axis=-1)  # Convert grayscale to RGB
    else:
        img_rgb = image_slice.copy()  # Already RGB

    # Ensure mask matches image shape
    if mask_slice.shape != image_slice.shape:
        from cv2 import resize, INTER_NEAREST
        mask_slice = resize(mask_slice.astype(np.uint8), (image_slice.shape[1], image_slice.shape[0]), interpolation=INTER_NEAREST)

    # Apply red overlay
    red = np.array([1.0, 0.0, 0.0])
    overlay = img_rgb.copy()
    overlay[mask_slice > 0] = (1 - alpha) * overlay[mask_slice > 0] + alpha * red

    return (overlay * 255).astype(np.uint8)

def display_slice(slice_data, display_widget):
    """Display a single slice in the specified widget"""
    slice_8bit = (exposure.rescale_intensity(slice_data) * 255).astype(np.uint8)
    slice_8bit = np.ascontiguousarray(slice_8bit)

    if slice_8bit.ndim == 2:  # Grayscale
        height, width = slice_8bit.shape
        qimage = QImage(slice_8bit.data, width, height, width, QImage.Format_Grayscale8)
    elif slice_8bit.ndim == 3:  # RGB
        height, width, _ = slice_8bit.shape
        qimage = QImage(slice_8bit.data, width, height, 3 * width, QImage.Format_RGB888)
    else:
        raise ValueError("Unsupported image shape for display")

    pixmap = QPixmap.fromImage(qimage).scaled(
        display_widget.width(),
        display_widget.height(),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )
    display_widget.setPixmap(pixmap)
