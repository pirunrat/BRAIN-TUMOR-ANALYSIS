# === frontend/style.py ===
from .Config import STYLE_TEMPLATE

def setup_styles(app_instance, palette):
    app_instance.setStyleSheet(STYLE_TEMPLATE.format(**palette))