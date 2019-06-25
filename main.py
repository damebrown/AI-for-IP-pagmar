import math
from tkinter import *
from tkinter.font import Font
from tkinter import Label
# import parsing
import sys
import choose_model

OVER = "C:\\Users\user\PycharmProjects\AIFIP_pagmar\display texts\over"
STAV = "C:\\Users\user\PycharmProjects\AIFIP_pagmar\display texts\stav"
ISLAND = "C:\\Users\user\PycharmProjects\AIFIP_pagmar\display texts\island"


def get_poem():
    poem = e1.get(1.0, END).encode('utf-8')
    return poem.split()


def start():
    dy = 350
    first_label['text'] = "How did"
    first_label.place(x = dx, y = dy, anchor = N)
    dy += 50
    e1.place(x = dx, y = dy, anchor = N)
    dy += 50
    sec_label['text'] = "influence"
    sec_label.place(x = dx, y = dy, anchor = N)
    dy += 50
    gobutton['text'] = "?"
    gobutton['command'] = get
    gobutton['font'] = ("Courier", 30, 'bold')
    gobutton.place(x = dx, y = dy, anchor = N)


def get():
    first_label['text'] = 'Thinking...'
    text = get_poem()
    e1.place_forget()
    sec_label.place_forget()
    res = query(text)
    if not res:
        first_label['text'] = 'Well, never!'
        gobutton['text'] = "Wanna try again?"
        gobutton['command'] = start
        gobutton['font'] = ("Courier", 22, 'bold')
        gobutton.place(x = dx, y = 450, anchor = N)
        return
    gobutton.place_forget()
    le = len(res)
    f3 = Font(family = "Courier", size = 24, weight = 'bold')
    h = root.winfo_height()
    first_label.place_forget()
    if le == 2:
        if res[1] in [12, 1, 2]:
            season = "Winter"
        elif res[1] in [3, 4, 5]:
            season = "Spring"
        elif res[1] in [6, 7, 8]:
            season = "Summer"
        elif res[1] in [9, 10, 11]:
            season = "Fall"
        label = Label(root, text = "Mmm... probably around\n" + str(season) + "\n of the year\n" + str(res[0]),
                      font = f3,
                      foreground = color)
        label.place(x = dx, y = h / 2 - 100, anchor = N)
        gobutton['text'] = "Wanna try again?"
        gobutton['command'] = start
        gobutton['font'] = ("Courier", 22, 'bold')
        gobutton.place(x = dx, y = 480, anchor = N)
        return


def query(poem):
    paths = [OVER, STAV, ISLAND]
    names = ["OVER", "STAV", "ISLAND"]
    for i, path in enumerate(paths):
        file = open(path, 'r')
        lines = file.readlines()
        if lines[0].startswith(poem[0]):
            res = choose_model.when(names[i])
            if not res:
                return 0
            return res[0] + 2000, int(res[1])


root = Tk()
x = 600
y = 800
root.geometry('{}x{}'.format(y, x))
frame = Frame(root)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
frame.place()
frame.winfo_toplevel().title("When did I write this poem?")
color = 'Dodger Blue'
rel = "ridge"
root.configure(bg = color)
im = "C:\\Users\user\PycharmProjects\AIFIP_pagmar\\notebook.gif"
background_image = PhotoImage(file = im)
background_label = Label(root, image = background_image, borderwidth = 800, highlightthickness = 0, width = 400)
background_label.place(x = 40, y = 10)
background_label.pack()
root.bind('<Escape>', sys.exit)

dx = root.winfo_screenwidth() / 2 - 5
dy = root.winfo_screenheight() / 4
f = Font(family = "Courier", size = 24, weight = 'bold')
first_label = Label(root, text = 'When the heck did I write the following:', font = f, foreground = color,
                    borderwidth = 2,
                    relief = rel)
first_label.place(x = dx, y = dy, anchor = N)
f1 = Font(family = "Courier", size = 18)
e1 = Text(root, height = 14, width = 55)
e1.tag_configure("center", justify = CENTER)
dy += 60
e1.place(x = dx, y = dy, anchor = N)
dy += 245
sec_label = Label(root, text = 'poem?', font = f, foreground = color, borderwidth = 2, relief = rel)
sec_label.place(x = dx, y = dy, anchor = N)

gobutton = Button(root, text = "?", fg = color, font = ("Courier", 30, 'bold'), command = get, bd = 6)
dy += 90
gobutton.place(x = dx, y = dy, anchor = N)
root.mainloop()
