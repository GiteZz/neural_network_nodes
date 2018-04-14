from tkinter import *
import random
import _pickle


tuple_list = []

def white_click_function(event):
    tuple_list.append((rgb_scale(current_rgb),[1,0]))
    new_color_label()
    print("White clicked")


def black_click_function(event):
    tuple_list.append((rgb_scale(current_rgb), [0,1]))
    new_color_label()
    print("Black clicked")

def new_color_label():
    global current_rgb
    current_rgb = random_rgb()

    color_white.config(bg=color_to_string(current_rgb))
    color_black.config(bg=color_to_string(current_rgb))

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


def rgb_scale(rgb):
    rgb_new = [0]*3
    for i in range(3):
        rgb_new[i] = rgb[i] / 255
    return rgb_new

def save_list(event):
    print("saving...")
    with open('outfile', 'wb') as fp:
        _pickle.dump(tuple_list, fp)

with open('outfile', 'rb') as fp:
    tuple_list = _pickle.load(fp)

current_rgb = random_rgb()
text_size = 20

window = Tk()
window.title("My machine learning gui")

black_label = Label(window, text="Black")
black_label.grid(row=0, column=0)

white_label = Label(window, text="White")
white_label.grid(row=0, column=2)

save_button = Label(window, text="Save", background="white", fg="black", font=(None, text_size))
save_button.grid(row=0, column=1)
save_button.bind("<Button-1>", save_list)

color_black = Label(window, text="black", background=color_to_string(current_rgb), height=20, width=20, fg="black", font=(None, text_size))
color_black.grid(row=1, column=0)
color_black.bind("<Button-1>", black_click_function)

color_white = Label(window, text="white", background=color_to_string(current_rgb), height=20, width=20, fg="white", font=(None, text_size))
color_white.grid(row=1, column=2)
color_white.bind("<Button-1>", white_click_function)

window.mainloop()