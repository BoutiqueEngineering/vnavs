from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range,
                      zip, round, input, int, pow, object)
from tkinter import *		# python 3
from tkinter import ttk	# python 3

#from Tkinter import *		# python 2.7
#import ttk			# python 2.7

import sys
import os
from PIL import ImageTk, Image

import threading
import time

import vnavs_mqtt

bot_path = "/Volumes/pi/projects/vnavs"

class Movie(vnavs_mqtt.mqtt_node):
    def __init__(self):
        super().__init__(Subscriptions=['helmsman/pic_ready'], Blocking=True, BlockingTimeoutSecs=0.1)
        self.tk_is_initialized = False
        self.lastfn = ""
        self.Connect()			# This starts the mqtt client in another thread
        self.tk_root = Tk()
        self.tk_root.title("Feet to Meters")

        self.f1_feet = StringVar()
        self.f1_meters = StringVar()
        self.f1_fname = StringVar()
        self.f1_fname.set('fname')

        mainframe = self.tk_root

        # row 0
        ttk.Label(mainframe, text='Feet').grid(column=0, row=0, sticky=W)
        self.f1_feet_entry = ttk.Entry(mainframe, width=7, textvariable=self.f1_feet)
        self.f1_feet_entry.grid(column=1, row=0, sticky=(W, E))

        # row 1
        ttk.Button(mainframe, text="Calculate", command=self.calculate).grid(column=0, columnspan=2, row=1, sticky=W)

        # row 2
        ttk.Label(mainframe, text="meters").grid(column=0, row=2, sticky=W)
        ttk.Label(mainframe, textvariable=self.f1_meters).grid(column=1, row=2, sticky=(W, E))

        # row 3
        self.f1_label1 = ttk.Label(mainframe, textvariable=self.f1_fname)
        self.f1_label1.grid(columnspan=2, sticky=W)

        # row 4
        fn = "temp/single.jpg"
        path = os.path.join(bot_path, fn)
        self.img_pil = Image.open(path)
        self.img_tk = ImageTk.PhotoImage(self.img_pil)
        self.f1_img = ttk.Label(mainframe, image = self.img_tk)
        self.f1_img.grid(column=0, columnspan=2, row=4, sticky=W)

        #for child in mainframe.winfo_children():
        #    child.grid_configure(padx=5, pady=5)

        self.f1_feet_entry.focus()
        self.tk_root.bind('<Return>', self.calculate)

    def rmsg_helmsman_pic_ready(self, msg):
        if not self.tk_is_initialized:
            return
        print("PIC", msg)
        self.picfn = msg
        self.f1_fname.set(self.picfn)
        path = os.path.join(bot_path, self.picfn)
        self.img_pil = Image.open(path)
        self.img_tk = ImageTk.PhotoImage(self.img_pil)
        self.f1_img.configure(image = self.img_tk)

    def calculate(self, *args):
        try:
            value = float(self.f1_feet.get())
            self.f1_meters.set((0.3048 * value * 10000.0 + 0.5)/10000.0)
        except ValueError:
            pass

    def mainloop(self):
        self.tk_is_initialized = True
        while True:
          # rmsg_helmsman_pic_ready is getting called asyncronously
          self.CheckMqtt()						# this has a short timeout
          self.tk_root.update()
        # when tk is destroyed by close window, self.Disconnect()	# stop mqtt client loop

m = Movie()
m.mainloop()
