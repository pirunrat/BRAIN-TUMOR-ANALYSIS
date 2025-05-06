import numpy as np
from skimage import exposure
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
from cv2 import resize, INTER_NEAREST

import numpy as np




def apply_segmentation(image_slice, mask_slice, alpha=0.5):
    """
    Overlay a transparent red mask on top of a grayscale image, preserving original pixel intensities.
    Only the mask region is tinted red.
    """
    import numpy as np
    from cv2 import resize, INTER_NEAREST

    # Ensure grayscale image is 2D
    if image_slice.ndim == 3 and image_slice.shape[-1] == 1:
        image_slice = image_slice[..., 0]

    # Ensure uint8 image
    if image_slice.dtype != np.uint8:
        image = image_slice.astype(np.float32)
        image = (255 * (image - image.min()) / (image.max() - image.min() + 1e-8)).astype(np.uint8)
    else:
        image = image_slice.copy()

    # Convert grayscale to RGB
    img_rgb = np.stack([image]*3, axis=-1)  # shape: (H, W, 3), dtype=uint8

    # Resize mask if needed
    if mask_slice.shape != image.shape:
        mask_slice = resize(mask_slice.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=INTER_NEAREST)

    # Apply transparent red only on mask
    mask_bool = mask_slice > 0
    img_rgb = img_rgb.astype(np.float32) / 255.0  # convert to [0, 1] float

    red_overlay = np.zeros_like(img_rgb)
    red_overlay[..., 0] = 1.0  # red channel

    img_rgb[mask_bool] = (1 - alpha) * img_rgb[mask_bool] + alpha * red_overlay[mask_bool]

    return (img_rgb * 255).astype(np.uint8)





def display_slice(slice_img, target_label):
    """
    Display a 2D slice (grayscale or RGB) on the given QLabel.
    """
    import numpy as np

    # Check if image is grayscale or RGB
    if slice_img.ndim == 2:
        # Grayscale → convert to RGB for uniformity
        img_rgb = np.stack([slice_img] * 3, axis=-1)
    elif slice_img.ndim == 3 and slice_img.shape[2] == 3:
        img_rgb = slice_img
    else:
        raise ValueError("Unsupported image format")

    # Ensure uint8
    if img_rgb.dtype != np.uint8:
        img_rgb = (img_rgb * 255).astype(np.uint8)

    h, w, _ = img_rgb.shape
    qimg = QImage(img_rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)

    pixmap = QPixmap.fromImage(qimg).scaled(
        target_label.width(),
        target_label.height(),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )
    target_label.setPixmap(pixmap)
