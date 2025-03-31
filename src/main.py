import tkinter as tk
from amz_trackdraw import AMZTrackDraw

# ---------------------------- MAIN GUI FUNCTION ---------------------------- #
def main_gui():
    root = tk.Tk()
    root.title("AMZ Trackdraw")
    app = AMZTrackDraw(root)
    root.mainloop()

if __name__ == "__main__":
    main_gui()