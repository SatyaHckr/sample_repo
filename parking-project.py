import cv2
# import pytesseract
from pyzbar.pyzbar import decode
import numpy as np
# pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def calculate_distance(box,sp):
  # Calculate the distance using the Euclidean distance formula
  distance = ((box[0] - sp[0])**2 + (box[1] - sp[1])**2)**0.5
  return distance

lst =['1234','1123','4515','4555']


def get_num(object):
  cap = cv2.VideoCapture(0)
  a = True

  while a==True:
      ret, frame = cap.read()

      # Convert the frame to grayscale for better QR code detection
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # Use pyzbar to decode QR codes
      qr_codes = decode(gray)

      for qr_code in qr_codes:
          data = qr_code.data.decode('utf-8')

          # Check if the decoded data is numeric
          if data.isnumeric():
              cv2.putText(frame, f'Numeric Data: {data}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

              if data in object:
                 continue
              else:
                object.append(str(data))
              # Optionally, you can use the data variable for further processing

      cv2.imshow('Webcam Feed', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()

#read image to find rectangles
empty_boxes = []

def rect(image):

  # image = cv2.imread('slot.jpeg')

  # Convert the image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Apply GaussianBlur to reduce noise and help contour detection
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)

  # Use Canny edge detector to find edges in the image
  edges = cv2.Canny(blurred, 50, 150)

  # Find contours in the edged image
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Iterate through the contours
  for contour in contours:
      # Approximate the contour to a polygon
      epsilon = 0.04 * cv2.arcLength(contour, True)
      approx = cv2.approxPolyDP(contour, epsilon, True)

      # If the polygon has four vertices, it's likely a rectangle
      if len(approx) == 4:
            # Calculate the centroid of the rectangle
          M = cv2.moments(approx)
          cX = int(M["m10"] / M["m00"])
          cY = int(M["m01"] / M["m00"])

            # Draw a dot at the center
          cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

          empty_boxes.append([cX,cY])

num_slot = {}

def find_closest_empty_box(image, start_point, number_plate):

  # Find the closest empty box to the starting point
  closest_empty_box = min(empty_boxes, key=lambda box: calculate_distance(box, start_point))

  cv2.circle(image, (closest_empty_box[0], closest_empty_box[1]), 5, (0, 255, 0), -1)
  # Write the number plate on the image
  cv2.putText(image, number_plate, (closest_empty_box[0]-45, closest_empty_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  num_slot[number_plate]=closest_empty_box

  print(closest_empty_box)

  empty_boxes.remove(closest_empty_box)

  print(empty_boxes)

  # Display the image with the closest empty box and the number plate
  cv2.imshow('Allocation',image)

  cv2.waitKey(0)

  cv2.destroyAllWindows()


remove_lst = []

def remove_cars():
   
   get_num(remove_lst)

   print(remove_lst)

   new_img = cv2.imread('slot.jpeg')

   for i in remove_lst:
      if i in num_slot:
         cv2.circle(new_img, (num_slot[i][0], num_slot[i][1]), 5, (0, 0, 255), -1)
         num_slot.pop(i)

   key = list(num_slot.keys())
   a=0
   for i in num_slot:

    cv2.circle(new_img, (num_slot[i][0], num_slot[i][1]), 5, (0, 255, 0), -1)


    cv2.putText(new_img, str(key[a]), (num_slot[i][0] -45, num_slot[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(key[a])
    a += 1

   cv2.imshow('New Allocation', new_img)

   cv2.waitKey(0)

   cv2.destroyAllWindows()

   

# reads the given image
img = cv2.imread("slot.jpeg")

# scans the QR to get car number
# get_num(lst)

# start point cooridnated
sp = [25,540]

#gets the middle point coordinated of empty boxes
rect(img)

# finds and allocate the closest box to number
for i in lst:
  find_closest_empty_box(img,sp,i)

remove_cars()

print(num_slot)
