import os, cv2, numpy as np
import math
import time
from scipy import weave
import sys


# automatically set threshold using technique from 
# http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
# just saw URL, and have seen it before, so that's re-assuring that I like it
def auto_canny(img_to_canny, auto_canny_sigma):
    img_to_canny_median = np.median(img_to_canny)
    lower_canny_thresh = int(max(0, (1 - auto_canny_sigma) * img_to_canny_median ))
    upper_canny_thresh = int(max(255, (1 + auto_canny_sigma) * img_to_canny_median ))
    return cv2.Canny(img_to_canny,lower_canny_thresh,upper_canny_thresh)


def apply_mask(channel, mask, fill_value):
    masked = np.ma.array(channel, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(channel, low_value, high_value):
    low_mask = channel < low_value
    channel = apply_mask(channel, low_mask, low_value)

    high_mask = channel > high_value
    channel = apply_mask(channel, high_mask, high_value)

    return channel

def simplest_cb(img, percentile):
    # Separately for each channel,
    # If the intensity is in the bottom X percentile, increase to the highest value in
    # that percentile group. This eliminates low values in this channel.
    # If the intensity is in the top X percentile, reduce to the lowest value in
    # that percentile group. This eliminates high values in this channel.
    #
    # The percentile parameter is the integer percentage that you want included
    # in this compression. The upper and lower percentiles are each 1/2 of this number.
    #assert img.shape[2] == 3
    assert percentile > 0 and percentile < 100

    half_percentile = percentile / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        # channels should be BGR
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentileile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)		# sort this channel (R, G or B) by intensity

        n_cols = flat.shape[0]
        # I added the int(). Floor retuns a float. Flat doesn't want a float. This probably was written
        # for python 3 which does some of these conversions differently.
        low_val  = flat[int(math.floor(n_cols * half_percentile))]
        high_val = flat[int(math.ceil( n_cols * (1.0 - half_percentile)))]

        print "Lowval: ", low_val
        print "Highval: ", high_val

        # saturate below the low percentileile and above the high percentileile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

def ColorKey(img):
    channels = cv2.split(img)
    print("CK channels:", len(channels))
    threshold = 200

    out_channels = []
    for channel in channels:
        # This really only works for one channel
        mask = channel <= threshold
        thresholded = apply_mask(channel, mask, 0)
        mask = channel > threshold
        thresholded = apply_mask(thresholded, mask, 255)
        # scale the channel
        #normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        normalized = thresholded
        out_channels.append(normalized)

    return cv2.merge(out_channels)

def _thinningIteration(im, iter):
	I, M = im, np.zeros(im.shape, np.uint8)
	expr = """
	for (int i = 1; i < NI[0]-1; i++) {
		for (int j = 1; j < NI[1]-1; j++) {
			int p2 = I2(i-1, j);
			int p3 = I2(i-1, j+1);
			int p4 = I2(i, j+1);
			int p5 = I2(i+1, j+1);
			int p6 = I2(i+1, j);
			int p7 = I2(i+1, j-1);
			int p8 = I2(i, j-1);
			int p9 = I2(i-1, j-1);
			int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
			         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
			         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
			         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
			if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
				M2(i,j) = 1;
			}
		}
	} 
	"""

	weave.inline(expr, ["I", "iter", "M"])
	return (I & ~M)


def thinning(src):
	dst = src.copy() / 255
	prev = np.zeros(src.shape[:2], np.uint8)
	diff = None

	while True:
		dst = _thinningIteration(dst, 0)
		dst = _thinningIteration(dst, 1)
		diff = np.absolute(dst - prev)
		prev = dst.copy()
		if np.sum(diff) == 0:
			break

	return dst * 255

def thinning_example(src):
        # This is just kept for doccumented example from thinning code
        # https://github.com/bsdnoobz/zhang-suen-thinning/blob/master/thinning.py
	#src = cv2.imread("kanji.png")
	#if src == None:
	#	sys.exit()
	bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	_, bw2 = cv2.threshold(bw, 10, 255, cv2.THRESH_BINARY)
	bw2 = thinning(bw2)
        return bw2
	cv2.imshow("src", bw)
	cv2.imshow("thinning", bw2)
	cv2.waitKey()



def ColorBalance(spath, dpath):
  # from https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
  # possibly from this Stanford C++ code: https://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html
  img = cv2.imread(spath)
  cb_img = simplest_cb(img, 20)
  cv2.imwrite(dpath, cb_img)

def LineFind(fpath):
  # load img
  imgOrg = cv2.imread(fpath + '_s.jpg')
  #img = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
  #[420:,:]
  img = simplest_cb(imgOrg, 20)
  #img = imgOrg

  # crop
  height, width, channels = img.shape
  c_x = 250
  c_y = 375
  c_w = 425
  print("Crop: (%d, %d) start (%d, %d) width %d" % (width, height, c_x, c_y, c_w))
  img = img[c_y:height, c_x:c_x+c_w]

  # bw img
  bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  bw_img = cv2.blurr(bw_img.copy(), (5,5))

  # histogram equalization
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  claheimg = clahe.apply(bw_img)

  # canny edge detection
  bw_edged = auto_canny(bw_img, 0.33)
  cont2, contours, hierarchy = cv2.findContours(bw_edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  countoured_img = cv2.drawContours(img.copy(), contours, -1, (0,255,0), 1)

  # the histogram equalization definitely makes everythign super sharp, but possibly
  # too sharp 
  clahe_edged = auto_canny(claheimg, 0.33)

  blurred_edges = cv2.GaussianBlur(clahe_edged, (21,21), 0)

  find_corners = auto_canny(clahe.apply(blurred_edges), 0.33)

  DrawGrid(imgOrg)
  cv2.imwrite(fpath + '_DA.jpg', imgOrg)
  cv2.imwrite(fpath + '_DB.jpg', img)
  cv2.imwrite(fpath + '_DC.jpg', bw_img)
  cv2.imwrite(fpath + '_DD.jpg', countoured_img)
  cv2.imwrite(fpath + '_DE.jpg', find_corners)
  cv2.imwrite(fpath + '_DT.jpg', thinning_example(img))

def HoughLines(img, gray):
  lined_img = img.copy()
  edges = cv2.Canny(gray.copy() ,100,200,apertureSize = 3)	# app size is 3, 5 or 7
  #edges = auto_canny(gray.copy(), 0.33)

  minLineLength = 30
  maxLineGap = 5
  maxLineGap = 1
  maxLineGap = 10
  rho = 30
  rho = 90
  rho = 1
  theta = np.pi / 180
  threshold = 15
  threshold = 1
  lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength,maxLineGap)
  if lines is not None:
    print("lineCt:", len(lines))
    for x in range(0, len(lines)):
      for x1,y1,x2,y2 in lines[x]:
        cv2.line(lined_img,(x1,y1),(x2,y2),(0,255,0),2)
  return edges, lined_img

def DrawContourLines(img, contours, color):
  h, w, channels = img.shape
  origin_x = int(w/2)
  origin_y = 0
  horizon_x = origin_x
  horizon_y = h
  tiny = []
  vertical = []
  horizontal = []
  for this_c in contours:
    rect = cv2.minAreaRect(this_c)
    # rect: center (x,y), (width, height), angle of rotation
    print("R", rect)
    line = MakeMapLine(rect, w, h)
    cv2.line(img, line[0], line[1], color ,2)
  return img

def ContourLines(img, gray, Drawlines=False, DrawBoth=False):
  # canny edge detection
  bw_edged = auto_canny(gray, 0.33)
  #cont2, contours, hierarchy = cv2.findContours(bw_edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  cont2, contours, hierarchy = cv2.findContours(bw_edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_TC89_L1)
  if len(img.shape) == 2:
    cropped_height, cropped_width = img.shape
  else:
    cropped_height, cropped_width, cropped_channels = img.shape
  (tiny, vertical, horizontal) = CreateContours(contours, cropped_width, cropped_height)
  if Drawlines or DrawBoth:
    contoured_img = DrawContourLines(img.copy(), tiny, (128,128,0))
    contoured_img = DrawContourLines(contoured_img, vertical, (128,128,0))
    contoured_img = DrawContourLines(contoured_img, horizontal, (128,128,0))
  else:
    contoured_img = img.copy()
  if (not Drawlines) or DrawBoth:
    contoured_img = cv2.drawContours(contoured_img.copy(), tiny, -1, (128,0,128), 1)
    contoured_img = cv2.drawContours(contoured_img.copy(), vertical, -1, (0,255,0), 1)
    contoured_img = cv2.drawContours(contoured_img.copy(), horizontal, -1, (255,0,0), 1)
  DumpContours(contours)
  return bw_edged, contoured_img

def MyLineFind(fpath, Crop=(250, 450)):
  imgOrg = cv2.imread(fpath + '_s.jpg')
  imgBal = simplest_cb(imgOrg, 20)
  #imgBal = imgOrg

  # crop
  height, width, channels = imgBal.shape
  c_x = Crop[0]
  c_y = 400
  c_w = Crop[1]
  print("Crop: (%d, %d) start (%d, %d) width %d" % (width, height, c_x, c_y, c_w))
  imgCrop = imgBal[c_y:height, c_x:c_x+c_w]
  #imgCrop = imgBal

  # bw img
  bw_img = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
  #bw_img = cv2.blur(bw_img.copy(), (3,3))
  #bw_img = simplest_cb(bw_img.copy(), 90)
  bw_img = ColorKey(bw_img)
  bw_img = cv2.blur(bw_img.copy(), (5,5))

  # convert to binary
  #(thresh, bw_img) = cv2.threshold(bw_img.copy(), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  #(thresh, bw_img) = cv2.threshold(bw_img.copy(), 128, 255, cv2.THRESH_BINARY)

  #bw_edged, lined_img = ContourLines(imgCrop, bw_img, DrawBoth=True)
  bw_edged, lined_img = HoughLines(imgCrop, bw_img)

  DrawGrid(imgOrg)
  cv2.imwrite(fpath + '_DA.jpg', imgOrg)
  cv2.imwrite(fpath + '_DB.jpg', imgBal)
  cv2.imwrite(fpath + '_DC.jpg', imgCrop)
  cv2.imwrite(fpath + '_DD.jpg', bw_img)
  cv2.imwrite(fpath + '_DE.jpg', bw_edged)
  cv2.imwrite(fpath + '_DF.jpg', lined_img)

def MakeMapLine(cvRect, w, h):
  # Box2D: center (x,y), (width, height), angle of rotation
  origin_x = int(w/2)
  origin_y = 0
  horizon_x = origin_x
  horizon_y = h
  box_x = cvRect[0][0]
  box_y = cvRect[0][1]
  box_w = cvRect[1][0]
  box_h = cvRect[1][1]
  hyp = box_h / 2
  box_r = math.radians(cvRect[2])
  print("deg",cvRect[2], "rad", box_r, math.sin(box_r), math.cos(box_r))
  y_offset = int(hyp * math.cos(box_r))
  x_offset = int(hyp * math.sin(box_r))
  return ((int(box_x + x_offset), int(box_y + y_offset)), (int(box_x - x_offset), int(box_y - y_offset)))

def CreateContours(src, w, h):
  origin_x = int(w/2)
  origin_y = 0
  horizon_x = origin_x
  horizon_y = h
  tiny = []
  vertical = []
  horizontal = []
  for this_c in src:
    if len(this_c) < 4:
      # not enough vertices
      #tiny.append(this_c)
      continue
    brec = cv2.boundingRect(this_c)
    #if brec[1] < 30:
    if brec[1] < 0:
      # Too far up in frame, ignore till we get closer.
      # This needs to be smarter because it catches lines that begin at the 
      # top of the frame but continue into the relevant part of hte frame.
      tiny.append(this_c)
      continue
    print("B", brec, "Vertices", len(this_c))
    rect = cv2.minAreaRect(this_c)
    # rect: center (x,y), (width, height), angle of rotation
    print("R", rect)
    print(MakeMapLine(rect, w, h)) 
    x = rect[0][0]
    y = rect[0][1]
    w = rect[1][0]
    h = rect[1][1]
    r = abs(rect[2])
    if r > 75:
      (w, h) = (h, w)
    if w > (h * 1.5):
      print("Horizontal")
      horizontal.append(this_c)
      continue
    print("Vertical")
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #vertical.append(box)
    vertical.append(this_c)
  #
  print(len(tiny), len(vertical), len(horizontal))
  return (tiny, vertical, horizontal)

def DumpContours(contours):
  c_l = len(contours)
  print("Contours len: ", c_l)
  for ix, this_c in enumerate(contours):
    for iy, this_vertex in enumerate(this_c):
      print("C[%d-%d] %s" % (ix, iy, this_vertex))

def FindVertices(contour):
  ul = contour[0]
  ur = contour[0]
  ll = contour[0]
  rr = contour[0]
  for ix, this_v in enumerate(contour):
    if this_v[0] < ul[0]:
      pass

def DrawGrid(img):
  grid_incr = 25
  height, width, channels = img.shape
  for x in range(grid_incr, width, grid_incr):
    cv2.line(img,(x, 0),(x, width),(255,0,0), 1)
  for y in range(grid_incr, height, grid_incr):
    cv2.line(img,(0,y),(width,y),(255,0,0), 1)

if __name__ == '__main__':
  start_time = time.clock()
  MyLineFind('opencv_3', Crop=(350,150))
  #MyLineFind('opencv_1')
  stop_time = time.clock()
  print("Elapsed Time:", (stop_time - start_time))

