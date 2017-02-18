from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)
#from tkinter import *		# python 3
#from tkinter import Ttk	# python 3

import sys
from Tkinter import *		# python 2.7
import ttk			# python 2.7
import threading
import time

import vnavs_mqtt

class ImageGrabber(vnavs_mqtt.mqtt_node):
  def __init__(self):
    super().__init__(Subscriptions=['helmsman/pic_ready'], Blocking=True)

  def rmsg_helmsman_pic_ready(self, msg):
    self.picfn = msg

class App(threading.Thread):
    def __init__(self, tk_root):
        super().__init__()
        self.root = tk_root
        mqttc = ImageGrabber()
        mqttc.Connect()
        threading.Thread.__init__(self)
        self.lastfn = ""

    def run(self):
        loop_active = True
        while loop_active:
            user_input = "XX"
            if user_input == "exit":
                loop_active = False
                self.root.quit()
                self.root.update()
            else:
                if self.lastfn != self.mqttc.picfn:
                  self.lastfn = self.mqttc.picfn
                  label = Label(self.root, text=self.lastfn)
                  label.pack()

#ROOT = Tk()
#LABEL = Label(ROOT, text="Hello, world!")
#LABEL.pack()
#ROOT.mainloop()

    
class Movie(threading.Thread):
#class Movie():
    def __init__(self):
        super().__init__()
        self.start()
    def run(self):
        self.root = Tk()
        self.root.title("Feet to Meters")

        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        mainframe.columnconfigure(0, weight=1)
        mainframe.rowconfigure(0, weight=1)

        feet = StringVar()
        meters = StringVar()

        feet_entry = ttk.Entry(mainframe, width=7, textvariable=feet)
        feet_entry.grid(column=2, row=1, sticky=(W, E))

        ttk.Label(mainframe, textvariable=meters).grid(column=2, row=2, sticky=(W, E))
        ttk.Button(mainframe, text="Calculate", command=self.calculate).grid(column=3, row=3, sticky=W)

        ttk.Label(mainframe, text="feet").grid(column=3, row=1, sticky=W)
        ttk.Label(mainframe, text="is equivalent to").grid(column=1, row=2, sticky=E)
        ttk.Label(mainframe, text="meters").grid(column=3, row=2, sticky=W)

        for child in mainframe.winfo_children():
            child.grid_configure(padx=5, pady=5)
        feet_entry.focus()
        self.root.bind('<Return>', self.calculate)
        self.root.mainloop()

    def calculate(*args):
        try:
            value = float(feet.get())
            meters.set((0.3048 * value * 10000.0 + 0.5)/10000.0)
        except ValueError:
            pass

m = Movie()
while True:
  time.sleep(2)
  print("HI")
#APP = App(m.root)
#APP.start()
