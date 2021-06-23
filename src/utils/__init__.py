import cv2
import os
import re
import fnmatch
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from collections import defaultdict
from glob import glob

def imshow(name, frame, flags=cv2.WINDOW_AUTOSIZE):
  cv2.namedWindow(name, flags)
  cv2.imshow(name, frame)
  cv2.waitKey()
  cv2.destroyAllWindows()

def random_color():
  """
  Generate a random color
  """
  color = list(np.random.choice(range(256), size=3))
  return (int(color[0]), int(color[1]), int(color[2]))

def perspective_transform(image, corners):
  def order_corner_points(corners):
    # Separate corners into individual points
    # Index 0 - top-right
    #       1 - top-left
    #       2 - bottom-left
    #       3 - bottom-right
    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
    return (top_l, top_r, bottom_r, bottom_l)

  # Order points in clockwise order
  ordered_corners = order_corner_points(corners)
  top_l, top_r, bottom_r, bottom_l = ordered_corners

  # Determine width of new image which is the max distance between 
  # (bottom right and bottom left) or (top right and top left) x-coordinates
  width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
  width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
  width = max(int(width_A), int(width_B))

  # Determine height of new image which is the max distance between 
  # (top right and bottom right) or (top left and bottom left) y-coordinates
  height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
  height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
  height = max(int(height_A), int(height_B))

  # Construct new points to obtain top-down view of image in 
  # top_r, top_l, bottom_l, bottom_r order
  dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                  [0, height - 1]], dtype = "float32")

  # Convert to Numpy format
  ordered_corners = np.array(ordered_corners, dtype="float32")

  # Find perspective transform matrix
  matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

  # Return the transformed image
  return cv2.warpPerspective(image, matrix, (width, height))

def rotate_image(image, angle):
  # Grab the dimensions of the image and then determine the center
  (h, w) = image.shape[:2]
  (cX, cY) = (w / 2, h / 2)

  # grab the rotation matrix (applying the negative of the
  # angle to rotate clockwise), then grab the sine and cosine
  # (i.e., the rotation components of the matrix)
  M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])

  # Compute the new bounding dimensions of the image
  nW = int((h * sin) + (w * cos))
  nH = int((h * cos) + (w * sin))

  # Adjust the rotation matrix to take into account translation
  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY

  # Perform the actual rotation and return the image
  return cv2.warpAffine(image, M, (nW, nH))

# Canny edge detection
def canny_edge(img, sigma=0.33):
  v = np.median(img)
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edges = cv2.Canny(img, lower, upper)
  return edges

# Hough line detection
def hough_line(edges, min_line_length=100, max_line_gap=10):
  lines = cv2.HoughLines(edges, 1, np.pi / 180, 125, min_line_length, max_line_gap)
  lines = np.reshape(lines, (-1, 2))
  return lines

# Separate line into horizontal and vertical
def h_v_lines(lines):
  h_lines, v_lines = [], []
  for rho, theta in lines:
    if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
      v_lines.append([rho, theta])
    else:
      h_lines.append([rho, theta])
  return h_lines, v_lines

# Find the intersections of the lines
def line_intersections(h_lines, v_lines):
  points = []
  for r_h, t_h in h_lines:
    for r_v, t_v in v_lines:
      a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
      b = np.array([r_h, r_v])
      inter_point = np.linalg.solve(a, b)
      points.append(inter_point)
  return np.array(points)

# Hierarchical cluster (by euclidean distance) intersection points
def cluster_points(points):
  dists = spatial.distance.pdist(points)
  single_linkage = cluster.hierarchy.single(dists)
  flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
  cluster_dict = defaultdict(list)
  for i in range(len(flat_clusters)):
    cluster_dict[flat_clusters[i]].append(points[i])
  cluster_values = cluster_dict.values()
  clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
  return sorted(round_iterable(list(clusters)), key=lambda k: [k[1], k[0]])

def round_iterable(iterable):
  return list(map(lambda item: (round(item[0]), round(item[1])), iterable))

def draw_chessboard_mapping(img, matrix):
  if len(matrix) == 9:
    mapping_img = img.copy()
    colors = [(140,0,236), (145,45,102), (166,84,0), (239,174,0), (81,166,0), (63,198,141), (0,242,255), (29,148,247), (36,28,237)]
    for (idx, points) in enumerate(matrix):
      # draw horizontal line
      cv2.line(mapping_img, points[0], points[-1], colors[idx], 2)

      # draw diagonal line
      if idx > 0:
        cv2.line(mapping_img, matrix[idx - 1][-1], points[0], colors[idx], 1)

      # draw points
      for point in points:
        cv2.circle(mapping_img, point, 5, colors[idx], -1)

    return mapping_img
  else:
    return np.array([])

def intersect_area(react1=None, react2=None):
  """
  Calcule area of intersect between two rectangles

  @params
  ---------
    `react1` [x1, y1, x2, y1]
    `react2` [x1, y1, x2, y1]

  @return
  ---------
  None if rectangles don't intersect
  """
  dx = min(react1[2], react2[2]) - max(react1[0], react2[0])
  dy = min(react1[3], react2[3]) - max(react1[1], react2[1])
  if (dx>=0) and (dy>=0):
    return dx*dy
  return None

def listdir(path, pattern):
  """
  List recursive files in dir with ignore case
  """
  regex = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
  return [os.path.abspath(name) for name in glob(path) if regex.match(name)]

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=2,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.
  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.
  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.
  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box (each to be shown on its
      own line).
    use_normalized_coordinates: If True (default), treat coordinates ymin, xmin,
      ymax, xmax as relative to the image.  Otherwise treat coordinates as
      absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)
  
  font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill='black',
              font=font)
    text_bottom -= text_height - 2 * margin