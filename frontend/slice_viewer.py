# === frontend/slice_viewer.py ===
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt

class SliceViewer(QLabel):
    def __init__(self, plane, event_handler, parent=None):
        super().__init__(parent)
        self.plane = plane
        self.event_handler = event_handler
        self.setAlignment(Qt.AlignCenter)

    def wheelEvent(self, event):
        direction = 1 if event.angleDelta().y() > 0 else -1
        self.event_handler.scroll_slice(self.plane, direction)
