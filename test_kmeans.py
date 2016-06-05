import tkinter as tk
import clustering.kmeans as km
import numpy as np
import random


def collectCoords(event):
    print(event.x, event.y)
    ca.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill="green")
    data.append([event.x, event.y])


def cluster():
    c, indice, cost = km.kmeans(np.array(data), 3)

    type = np.shape(np.unique(indice))[0]

    color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(0, type)]

    # print '#%.2X%.2X%.2X' % (color1[indice[0] % type], color2[indice[0] % type], color3[indice[0] % type])
    i = 0
    for d in data:
        ca.create_oval(d[0] - 5, d[1] - 5, d[0] + 5, d[1] + 5,
                       fill='#%.2X%.2X%.2X' % (
                           color[indice[i]][0], color[indice[i]][1], color[indice[i]][2]))
        i += 1

    print(indice)


def clear():
    del data[:]
    ca.delete("all")


top = tk.Tk()

top.geometry('500x700')
top.title('get data')

data = []
ca = tk.Canvas(top, width=500, height=500)
ca.bind('<Button-1>', collectCoords)
ca.pack()

btn = tk.Button(top, text="cluster", height=1, width=20, command=cluster)
btn.pack()
btn2 = tk.Button(top, text="clear", height=1, width=40, command=clear)
btn2.pack()

top.mainloop()