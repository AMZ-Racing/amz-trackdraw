import tkinter as tk
from fs_trackdraw import FSTrackDraw

# ---------------------------- MAIN GUI FUNCTION ---------------------------- #
def main_gui():
    root = tk.Tk()
    root.title("Trackdraw")
    app = FSTrackDraw(root)
    root.mainloop()

if __name__ == "__main__":
    main_gui()