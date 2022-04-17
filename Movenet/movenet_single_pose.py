import tensorflow as tf
import numpy as np 
import cv2

image_path = "C:/Users/ARTH/Desktop/Computer_Vision/movenet/jojo.jpeg"
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)

input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, 192, 192)

model_path = "C:/Users/ARTH/Desktop/Computer_Vision/movenet/movenet_lightning_fp16.tflite"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

input_image = tf.cast(input_image, dtype=tf.uint8)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
interpreter.invoke()
keypoints = interpreter.get_tensor(output_details[0]['index'])

width = 640
height = 640

KEYPOINT_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7),
    (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)]

input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, width, height)
input_image = tf.cast(input_image, dtype=tf.uint8)

image_np = np.squeeze(input_image.numpy(), axis=0)
image_np = cv2.resize(image_np, (width, height))
image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
#print(keypoints[])
for keypoint in keypoints[0][0]:
    x = int(keypoint[1] * width)
    y = int(keypoint[0] * height)

    cv2.circle(image_np, (x, y), 4, (0, 0, 255), -1)


print('hello')
x1=keypoints[0][0][5][1]#left shld
x2=keypoints[0][0][6][1]#right shld
y1=keypoints[0][0][5][0]#left shld
y2=keypoints[0][0][6][0]#right shld
print('dist:')
print(x1,y1)
print(x2,y2)
val=((y2-y1)**2+(x2-x1)**2)**(1/2)
print(val*640)

print('hello')
x3=keypoints[0][0][1][1]#left eye
x4=keypoints[0][0][2][1]#right eye
y3=keypoints[0][0][1][0]#left eye
y4=keypoints[0][0][2][0]#right eye
print('dist:')
print(x3,y3)
print(x4,y4)
val2=((y3-y4)**2+(x3-x4)**2)**(1/2)
print(val2*640)


for edge in KEYPOINT_EDGES:
    
    x1 = int(keypoints[0][0][edge[0]][1] * width)
    y1 = int(keypoints[0][0][edge[0]][0] * height)

    x2 = int(keypoints[0][0][edge[1]][1] * width)
    y2 = int(keypoints[0][0][edge[1]][0] * height)

    cv2.line(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("pose estimation", image_np)
cv2.imwrite(r'D:\capstone\working project\blazepose\mobenet.png',image_np)
        
cv2.waitKey()
