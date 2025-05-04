# === main.py ===
import sys
from PyQt5.QtWidgets import QApplication
from frontend.frontend import MultiPlaneBrainTumorApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MultiPlaneBrainTumorApp()
    window.show()
    sys.exit(app.exec_())

