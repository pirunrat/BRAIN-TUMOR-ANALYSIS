

# frontend/event_handler.py
class EventHandler:
    def __init__(self, main_window):
        self.main_window = main_window
    
    def _get_axis(self, plane):
        """Return axis index for a given plane name."""
        return {
            'sagittal': 0,  # x
            'coronal': 1,   # y
            'axial': 2      # z
        }[plane]
    
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
    

    
    def scroll_slice(self, plane, direction):
        backend = self.main_window.backend
        if not hasattr(backend, 'volume_data') or backend.volume_data is None:
            return

        max_slice = backend.volume_data.shape[self._get_axis(plane)] - 1
        current = self.main_window.current_slice[plane]
        new_slice = max(0, min(max_slice, current + direction))

        if new_slice != current:
            self.main_window.current_slice[plane] = new_slice

            slider = getattr(self.main_window.sidebar, f"{plane}_slider")
            label = getattr(self.main_window.sidebar, f"{plane}_slider_label")

            slider.setValue(new_slice)
            label.setText(f"{plane.capitalize()} Slice: {new_slice}")
            self.main_window.display_panel.update_all_views()

