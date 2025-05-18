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
from skimage import io, exposure
import numpy as np
import cv2


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
            self.classifier = self._load_classification_model()
            
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
                'SegFormer_New_Arch_at_0.9400.pth'
            )
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            # Create model architecture (replace with your actual model class)
            model = Segmentor()
            
            # Load pretrained weights
            checkpoint = torch.load(model_path, map_location='cuda')
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model.eval()
            
            return model
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to load segmentation model: {str(e)}")
            raise
    
    def _load_classification_model(self):
        """Load the segmentation model with pretrained weights"""
        try:
            # Get absolute path to model file
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'classification',
                'BTC_Model_at_0.9571_FeatAndOut.pth'
            )
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            # Create model architecture (replace with your actual model class)
            model = Classifier()
            
            # Load pretrained weights
            checkpoint = torch.load(model_path, map_location='cuda')
            model.load_state_dict(checkpoint, strict=True)
            model.eval()
            
            return model
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to load segmentation model: {str(e)}")
            raise

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

    def load_image_2d(self, file_path):
        try:

            img = io.imread(file_path, as_gray=True)
            img = self.convert_to_uint8(img)
            img = self.image_normalization(img)
         
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

   
    def classify_tumor(self):
        if self.volume_data is None:
            self.error_occurred.emit("No volume data available for classification.")
            return None

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.classifier = self.classifier.to(device)

            input_data = self.volume_data

            # If 3D, take the center slice
            if len(input_data.shape) == 3:
                d = input_data.shape[0] // 2
                input_data = input_data[d]

            # Resize to 128x128 if too small
            if input_data.shape[0] < 128 or input_data.shape[1] < 128:
                input_data = cv2.resize(input_data, (128, 128), interpolation=cv2.INTER_LINEAR)

            tensor = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
            tensor = tensor.repeat(1, 3, 1, 1).to(device)  # [1, 3, H, W]

            with torch.no_grad():
                outputs = self.classifier(tensor)  # <-- match your notebook

                probs = torch.softmax(outputs, dim=1).cpu().squeeze(0)  # Shape: [num_classes]
                #print(f'Probabilities shape : {probs.shape}')
                pred_class_idx = int(torch.argmax(probs).item())

            class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
            result = {
                "predicted_class": class_names[pred_class_idx],
                "probabilities": {cls: float(probs[i].item()) for i, cls in enumerate(class_names)}
            }

            return result

        except Exception as e:
            self.error_occurred.emit(f"Classification failed:\n{str(e)}")
            return None


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
                        output = torch.sigmoid(output)
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
                    output = torch.sigmoid(output)
                    mask = (output > 0.5).float().squeeze().cpu().numpy()
                    mask = cv2.resize(mask.astype(np.uint8), (slice_data.shape[1], slice_data.shape[0]), interpolation=cv2.INTER_NEAREST)
                    self.segmentation_masks = mask
                    plt.imshow(mask, cmap='gray')
                    plt.show()
                    
            else:
                self.error_occurred.emit("Unsupported input dimensions for segmentation.")
                return

            self.progress_updated.emit(100)
            self.processing_complete.emit()

        except Exception as e:
            self.error_occurred.emit(f"Segmentation failed:\n{str(e)}")


    