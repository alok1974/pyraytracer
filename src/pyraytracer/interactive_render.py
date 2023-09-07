import sys
import tkinter as tk

from PIL import Image, ImageTk


class InteractiveRenderWindow:
    def __init__(self, image: Image.Image, close_after_rendering: bool = False) -> None:
        # Create the main tkinter window
        self._image = image
        self._close_after_rendering = close_after_rendering

        self._root = tk.Tk()
        self._root.title("Interactive Renderer")

        self._root.protocol("WM_DELETE_WINDOW", self._disable_close)

        # Convert the PIL image to a PhotoImage and display it
        self._tk_image = ImageTk.PhotoImage(self._image)

        # Use a Canvas instead of a Label to display the image
        self._canvas = tk.Canvas(self._root, width=self._image.width, height=self._image.height)
        self._canvas.pack()
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_image)

        self._root.after(1, self._center_window)

    def _center_window(self) -> None:
        # Get the screen width and height
        screen_width = self._root.winfo_screenwidth()
        screen_height = self._root.winfo_screenheight()

        # Get the window width and height
        window_width = self._root.winfo_width()
        window_height = self._root.winfo_height()

        # Calculate the x and y coordinates to center the window
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)

        # Set the window's position
        self._root.geometry(f'{window_width}x{window_height}+{x}+{y}')

    def _disable_close(self):
        ...

    def update_pixels(self, new_image: Image.Image) -> None:
        # Replace the current image with the new one
        self._image = new_image

        # Convert the updated PIL image to a PhotoImage and update the display
        self._tk_image = ImageTk.PhotoImage(self._image)

        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_image)

    def run(self) -> None:
        self._root.mainloop()

    def on_rendering_complete(self) -> None:
        # Re-enable the close button
        self._root.protocol("WM_DELETE_WINDOW", self._root.destroy)

        # Automatically close the window
        if self._close_after_rendering:
            self._root.destroy()
            sys.exit()
