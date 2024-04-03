import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk


def getResult(output, root):
    root.destroy()
    outWin = tk.Tk()
    outWin.title("OAR")
    outWin.state('zoomed')
    try:
        getPath = filedialog.askopenfile(mode='r')
        imgPath = (str(os.path.abspath(getPath.name)))
        lbl, conf, img = output(imgPath)
        dispImg = Image.fromarray(img)
        dispImgTK = ImageTk.PhotoImage(image=dispImg)
        tk.Label(outWin, image=dispImgTK).pack(anchor="center", pady=50)
        resultText = ("This image is of the letter " + lbl +
                      ", with a " + conf + "% confidence")
        result = tk.Label(outWin, text=resultText)
        result.pack(anchor="center")
        btnText = "Open Main Menu"
        againBtn = ttk.Button(outWin, text=btnText,
                              command=lambda: displayNew(output, outWin))
        againBtn.pack(anchor='center', pady=25)
        exitBtn = ttk.Button(outWin, text="Close", command=outWin.destroy)
        exitBtn.pack(anchor='center', pady=50)
    except AttributeError:
        errText = "No file was chosen"
        messagebox.showerror("File Error", errText)
        btnText = "Open Main Menu"
        againBtn = ttk.Button(outWin, text=btnText,
                              command=lambda: displayNew(output, outWin))
        againBtn.pack(anchor='center', pady=25)
        exitBtn = ttk.Button(outWin, text="Close", command=outWin.destroy)
        exitBtn.pack(anchor='center', pady=50)
    outWin.mainloop()


def displayNew(output, root):
    root.destroy()
    dispNew = tk.Tk()
    dispNew.title("OAR")
    dispNew.state('zoomed')

    oarLogo = tk.PhotoImage(file="OAR_logo.png")
    logo = ttk.Label(dispNew, image=oarLogo)
    logo.place(x=75, y=0, relwidth=1, relheight=1)

    init = "Please Enter an Image to be Classified"
    appBtn = ttk.Button(dispNew, text=init,
                        command=lambda: getResult(output, dispNew))
    appBtn.pack(anchor='center', pady=180)

    quitBtn = ttk.Button(dispNew, text="Quit", command=dispNew.destroy)
    quitBtn.pack(anchor='center')

    dispNew.mainloop()


def display(output):
    disp = tk.Tk()
    disp.title("OAR")
    disp.state('zoomed')

    oarLogo = tk.PhotoImage(file="OAR_logo.png")
    logo = ttk.Label(disp, image=oarLogo)
    logo.place(x=75, y=0, relwidth=1, relheight=1)

    init = "Please Enter an Image to be Classified"
    appBtn = ttk.Button(disp, text=init,
                        command=lambda: getResult(output, disp))
    appBtn.pack(anchor='center', pady=180)

    quitBtn = ttk.Button(disp, text="Quit", command=disp.destroy)
    quitBtn.pack(anchor='center')

    disp.mainloop()
