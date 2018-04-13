from tkinter import *
import random


def white_click_function(event):
    print("White clicked")


def black_click_function(event):
    print("Black clicked")


def color_to_string(color):
    ret_str = "#"
    for comp in color:
        hex_value = hex(comp)[2:]
        if len(hex_value) == 1:
            ret_str += "0" + hex_value
        else:
            ret_str += hex_value
    return ret_str


def random_rgb():
    ret_col = []
    for _ in range(3):
        ret_col.append(random.randint(0, 255))
    return ret_col


window = Tk()
window.title("My machine learning gui")

black_label = Label(window, text="Black")
black_label.grid(row=0, column=0)

white_label = Label(window, text="White")
white_label.grid(row=0, column=1)

color_1 = Label(window, text="black", background=color_to_string(random_rgb()), height=20, width=20, fg="black")
color_1.grid(row=1, column=0)
color_1.bind("<Button-1>", black_click_function)

color_2 = Label(window, text="white", background=color_to_string(random_rgb()), height=20, width=20, fg="white")
color_2.grid(row=1, column=1)
color_2.bind("<Button-1>", white_click_function)

window.mainloop()