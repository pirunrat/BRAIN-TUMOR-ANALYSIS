

# frontend/event_handler.py
class EventHandler:
    def __init__(self, main_window):
        self.main_window = main_window

    def handle_slice_click(self, event, plane):
        """Update the selected slice and refresh the corresponding view."""
        backend = self.main_window.backend
        if not hasattr(backend, 'volume_data') or backend.volume_data is None:
            return

        slider = getattr(self.main_window.sidebar, f"{plane}_slider")
        label = getattr(self.main_window.sidebar, f"{plane}_slider_label")

        pos = event.pos().x()
        slice_pos = int((pos / slider.width()) * slider.maximum())

        # Update state
        self.main_window.current_slice[plane] = slice_pos
        slider.setValue(slice_pos)
        label.setText(f"{plane.capitalize()} Slice: {slice_pos}")

        self.main_window.display_panel.update_all_views()
