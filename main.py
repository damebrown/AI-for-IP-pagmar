import math
from tkinter import *
from tkinter.font import Font
from tkinter import Label
# import parsing
import sys
import choose_model


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
    if res == -1:
        first_label['text'] = 'He didn\'t!'
        start()
    if res == -2:
        first_label['text'] = 'Something went wrong'
        start()
    else:
        le = len(res)
        f3 = Font(family = "Courier", size = 24)
        h = root.winfo_height()
        first_label.place_forget()
        if le == 2:
            label = Label(root, text = res[0] + " influenced " + res[1] + " directly!", font = f3, foreground = color)
            label.place(x = dx, y = h / 2, anchor = N)
            return
        dy = math.floor(h / 2) - (math.floor(le / 2) * 50)
        for i in range(1, len(res)):
            if i == 1:
                label = Label(root, text = res[i - 1] + " influenced " + res[i] + "...", font = f3, foreground = color)
                label.place(x = dx, y = dy, anchor = N)
            else:
                if i == le - 1:
                    label = Label(root, text = "And " + res[i - 1] + " influenced " + res[i] + "!", font = f3,
                                  foreground = color)
                    label.place(x = dx, y = dy, anchor = N)
                else:
                    label = Label(root, text = "And " + res[i - 1] + " influenced " + res[i] + "...", font = f3,
                                  foreground = color)
                    label.place(x = dx, y = dy, anchor = N)
            dy += 50


OVER = "C:\\Users\user\PycharmProjects\AIFIP_pagmar\display texts\over"
STAV = "C:\\Users\user\PycharmProjects\AIFIP_pagmar\display texts\stav"
ISLAND = "C:\\Users\user\PycharmProjects\AIFIP_pagmar\display texts\island"


def query(poem):
    paths = [OVER, STAV, ISLAND]
    names = ["OVER", "STAV", "ISLAND"]
    for i, path in enumerate(paths):
        file = open(path, 'r')
        lines = file.readlines()
        if lines[0].startswith(poem[0]):
            res = choose_model.when(names[i])
    return


# if __name__ == "__main__":
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
# background_label = Label(root, image = background_image)
background_label.place(x = 40, y = 10)
background_label.pack()
root.bind('<Escape>', sys.exit)

# dx = 955
# dy = 220
dx = root.winfo_screenwidth() / 2
dy = root.winfo_screenheight() / 4
f = Font(family = "Courier", size = 24, weight = 'bold')
first_label = Label(root, text = 'When the heck did I write the following:', font = f, foreground = color,
                    borderwidth = 2,
                    relief = rel)
first_label.place(x = dx, y = dy, anchor = N)
f1 = Font(family = "Courier", size = 18)
# e1 = Entry(root, width = 40, height = 160, justify = CENTER, font = f1)
e1 = Text(root, height = 14, width = 60)
e1.tag_configure("center", justify = CENTER)
# dy += 130
dy += 60
e1.place(x = dx, y = dy, anchor = N)
# dy += 370
dy += 250
sec_label = Label(root, text = 'poem', font = f, foreground = color, borderwidth = 2, relief = rel)
sec_label.place(x = dx, y = dy, anchor = N)

gobutton = Button(root, text = "?", fg = color, font = ("Courier", 30, 'bold'), command = get, bd = 6)
dy += 50
gobutton.place(x = dx, y = dy, anchor = N)

root.mainloop()
