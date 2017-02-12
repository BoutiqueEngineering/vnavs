import os, cv2, numpy as np
import math


# automatically set threshold using technique from 
# http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
# just saw URL, and have seen it before, so that's re-assuring that I like it
def auto_canny(img_to_canny, auto_canny_sigma):
    img_to_canny_median = np.median(img_to_canny)
    lower_canny_thresh = int(max(0, (1 - auto_canny_sigma) * img_to_canny_median ))
    upper_canny_thresh = int(max(255, (1 + auto_canny_sigma) * img_to_canny_median ))
    return cv2.Canny(img_to_canny,lower_canny_thresh,upper_canny_thresh)


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]
        # I added the int(). Floor retuns a float. Flat doesn't want a float. This probably was written
        # for python 3 which does some of these conversions differently.
        low_val  = flat[int(math.floor(n_cols * half_percent))]
        high_val = flat[int(math.ceil( n_cols * (1.0 - half_percent)))]

        print "Lowval: ", low_val
        print "Highval: ", high_val

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

def ColorBalance(fpath):
  # from https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
  # possibly from this Stanford C++ code: https://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html
  img = cv2.imread(fpath)
  cb_img = simplest_cb(img, 20)
  cv2.imwrite('cb.jpg', cb_img)

def LineFind(fpath):
  # load img
  imgOrg = cv2.imread(fpath + '_s.jpg')
  #img = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2RGB)
  #[420:,:]
  img = simplest_cb(imgOrg, 20)
  #img = imgOrg

  # bw img
  bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

  cv2.imwrite(fpath + '_DA.jpg', imgOrg)
  cv2.imwrite(fpath + '_DB.jpg', img)
  cv2.imwrite(fpath + '_DC.jpg', bw_img)
  cv2.imwrite(fpath + '_DD.jpg', countoured_img)
  cv2.imwrite(fpath + '_DE.jpg', find_corners)

#ColorBalance('imgs/R3_12.jpg')
LineFind('opencv_1')

