import timeit

from PIL import Image, ImageFilter
#Read image

import opencv_test

def Transform(px):
  m = min(px)
  r = px[0]
  if r < 128:
    # not enough red to be be yellow or white
      return (0,0,0)
  b = px[2]
  if b > m:
    # blue isn't the smallest component, so call this white
    return (255,255,255)
  return (255,255,0)		# yellow

def Quantitize():
  q_steps = 5
  q_steps = 10
  q_steps = 2
  q_div = (256 * 3) // q_steps
  q_color_step = 255 // q_steps
  im_data = im.getdata()
  new_data = []
  for (r, g, b)  in im_data:
    pxv = r + g + b
    pxv = r + r + g + b
    pxq = pxv // q_div
    c = pxq * q_color_step
    if c > 255:
      c = 255
    new_data.append((c, c, c))
  im.putdata(new_data) 

def MakeArray():
  im_data = im.getdata()
  new_data = []
  for px in im_data:
    new_data.append(Transform(px))
  im.putdata(new_data) 

def DirectUpdate():
  im_data = im.load()
  for x in range(im_width):
    for y in range(im_height):
      im_data[x, y] = Transform(im_data[x, y])

"""
start_time = timeit.default_timer()
MakeArray()
elapsed = timeit.default_timer() - start_time
print(elapsed)

start_time = timeit.default_timer()
DirectUpdate()
elapsed = timeit.default_timer() - start_time
print(elapsed)
"""

fn = 'opencv_1'
source_path = fn + '_s.jpg'
balance_path = fn + '_balanced.jpg'
dest_path = fn + '_colorkey.jpg'
opencv_test.ColorBalance(source_path, balance_path)

im = Image.open(balance_path)
(im_width, im_height) = im.size
#im.show()

#Quantitize()
MakeArray()
#im.show()
im.save(dest_path, "JPEG")
