





PALLETTE = {
            'background': '#1e1e2e',
            'foreground': '#ffffff',
            'primary': '#6c5ce7',
            'secondary': '#a29bfe',
            'accent': '#fd79a8',
            'card': '#2d3436',
            'text': '#dfe6e9',
            'success': '#00b894',
            'warning': '#fdcb6e',
            'danger': '#d63031'
        }


# style_config.py

STYLE_TEMPLATE = """
QMainWindow {{
    background-color: {background};
    color: {foreground};
}}
QPushButton {{
    background-color: {primary};
    color: white;
    border: none;
    padding: 10px;
    border-radius: 5px;
    font-size: 14px;
    min-width: 120px;
}}
QPushButton:hover {{
    background-color: {secondary};
}}
QPushButton:disabled {{
    background-color: #636e72;
    color: #b2bec3;
}}
QLabel {{
    color: {text};
    font-size: 14px;
}}
QComboBox {{
    background-color: {card};
    color: {text};
    border: 1px solid {secondary};
    padding: 5px;
    border-radius: 4px;
    min-width: 200px;
}}
QProgressBar {{
    border: 1px solid {secondary};
    border-radius: 5px;
    text-align: center;
    background: {card};
}}
QProgressBar::chunk {{
    background-color: {primary};
}}
QTabWidget::pane {{
    border: 1px solid {secondary};
    border-radius: 5px;
    background: {card};
}}
QTabBar::tab {{
    background: {card};
    color: {text};
    padding: 8px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    border: 1px solid {secondary};
}}
QTabBar::tab:selected {{
    background: {primary};
    color: white;
}}
"""
