import os
import numpy as np
import nibabel as nib
from skimage import io, exposure
from PyQt5.QtCore import QObject, pyqtSignal
import torch
from torch import nn
from backend.segmentation.segmentation import Segmentor
from backend.classification.classification import Classifier
import matplotlib.pyplot as plt

class Backend(QObject):
    progress_updated = pyqtSignal(int)
    processing_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.volume_data = None
        self.segmentation_masks = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize and load both segmentation and classification models"""
        try:
            # Initialize models
            self.segmentor = self._load_segmentation_model()
            #self.classifier = self._load_classification_model()
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize models: {str(e)}")
            raise

    def _load_segmentation_model(self):
        """Load the segmentation model with pretrained weights"""
        try:
            # Get absolute path to model file
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'segmentation',
                'SegFormer_Distilled_New_at_0.9475_patch13_response.pth'
            )
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            # Create model architecture (replace with your actual model class)
            model = Segmentor()
            
            # Load pretrained weights
            checkpoint = torch.load(model_path, map_location='cuda')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to load segmentation model: {str(e)}")
            raise

    
    def load_image_2d(self, file_path):
        try:
            from skimage import io, exposure
            import numpy as np
            import cv2

            img = io.imread(file_path, as_gray=True)
            img = exposure.rescale_intensity(img)

            # Resize if smaller than required by model
            min_size = 128
            h, w = img.shape
            if h < min_size or w < min_size:
                img = cv2.resize(img, (max(w, min_size), max(h, min_size)), interpolation=cv2.INTER_LINEAR)

            self.volume_data = img.astype(np.float32)
            self.image_2d_path = file_path
            self.segmentation_masks = None
            self.processing_complete.emit()
        except Exception as e:
            self.error_occurred.emit(f"Failed to load 2D image: {str(e)}")


    def load_volume(self, file_name):
        """Load either 2D image or 3D volume"""
        try:
            self.progress_updated.emit(10)
            
            if file_name.lower().endswith(('.nii', '.nii.gz')):
                # 3D NIfTI volume loading (existing code)
                img = nib.load(file_name)
                self.volume_data = img.get_fdata()
                self.is_3d = True  # Flag for 3D data
                # ... rest of your NIfTI loading code ...
                
            elif file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 2D image loading
                img = io.imread(file_name, as_gray=True)
                self.volume_data = np.expand_dims(img, axis=0)  # Add z-dimension [1, H, W]
                self.is_3d = False  # Flag for 2D data
                self.progress_updated.emit(100)
                
            # Normalize and convert to float32 (common for both)
            self.volume_data = self.volume_data.astype(np.float32)
            self.volume_data = (self.volume_data - np.min(self.volume_data)) / \
                            (np.max(self.volume_data) - np.min(self.volume_data))
            
            self.segmentation_masks = None
            self.processing_complete.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to load image:\n{str(e)}")

   

    def segment_tumor(self):
        if self.volume_data is None:
            self.error_occurred.emit("No volume data loaded")
            return

        try:
            self.progress_updated.emit(10)
            import torch
            import cv2
            import numpy as np

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.segmentor = self.segmentor.to(device)

            # Handle 3D volume [D, H, W]
            if len(self.volume_data.shape) == 3:
                masks = []
                num_slices = self.volume_data.shape[0]

                for i in range(num_slices):
                    slice_data = self.volume_data[i]

                    # Resize slice if too small
                    h, w = slice_data.shape
                    if h < 64 or w < 64:
                        slice_data = cv2.resize(slice_data, (128, 128), interpolation=cv2.INTER_LINEAR)

                    input_tensor = torch.from_numpy(slice_data).float()
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
                    input_tensor = input_tensor.repeat(1, 3, 1, 1)  # Convert to 3 channels

                    with torch.no_grad():
                        output, _ = self.segmentor(input_tensor)
                        mask = (output > 0.5).float()
                        masks.append(mask.squeeze().cpu().numpy())

                    self.progress_updated.emit(10 + int((i+1)/num_slices * 90))

                self.segmentation_masks = np.stack(masks)  # [D, H, W]

            # Handle 2D image [H, W]
            elif len(self.volume_data.shape) == 2:
                slice_data = self.volume_data

                # Resize if too small
                h, w = slice_data.shape
                if h < 64 or w < 64:
                    slice_data = cv2.resize(slice_data, (128, 128), interpolation=cv2.INTER_LINEAR)

                input_tensor = torch.from_numpy(slice_data).float()
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
                input_tensor = input_tensor.repeat(1, 3, 1, 1)

                with torch.no_grad():
                    output, _ = self.segmentor(input_tensor)
                    mask = (output > 0.5).float().squeeze().cpu().numpy()
                    mask = cv2.resize(mask.astype(np.uint8), (slice_data.shape[1], slice_data.shape[0]), interpolation=cv2.INTER_NEAREST)
                    self.segmentation_masks = mask
                    # plt.imshow(mask, cmap='gray')
                    # plt.show()
                    
            else:
                self.error_occurred.emit("Unsupported input dimensions for segmentation.")
                return

            self.progress_updated.emit(100)
            self.processing_complete.emit()

        except Exception as e:
            self.error_occurred.emit(f"Segmentation failed:\n{str(e)}")


    