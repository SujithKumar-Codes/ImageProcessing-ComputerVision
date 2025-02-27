import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "batman.jpg"
image = cv2.imread(image_path)

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the Image
resized = cv2.resize(image, (300, 300))

# Rotate the Image (90 degrees clockwise)
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Flip the Image (Horizontally)
flipped = cv2.flip(image, 1)

# Blur the Image
blurred = cv2.GaussianBlur(image, (15, 15), 0)

# Edge Detection using Canny
edges = cv2.Canny(image, 50, 150)

# Thresholding (Binary)
_, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive Thresholding
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Dilation
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# Erosion
eroded = cv2.erode(edges, kernel, iterations=1)

# Morphological Gradient
gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

# Sobel Edge Detection (X & Y)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

# Laplacian Edge Detection
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Convert to HSV Color Space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert to LAB Color Space
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Histogram Equalization
equalized = cv2.equalizeHist(gray)

# Draw a Rectangle
rectangle = image.copy()
cv2.rectangle(rectangle, (50, 50), (250, 250), (0, 255, 0), 3)

# Draw a Circle
circle = image.copy()
cv2.circle(circle, (150, 150), 50, (255, 0, 0), -1)

# Draw a Line
line = image.copy()
cv2.line(line, (50, 50), (250, 250), (0, 0, 255), 3)


# Prepare images for display
images = [gray, resized, rotated, flipped, blurred, edges, thresholded, adaptive_thresh, dilated, eroded,
          gradient, sobelx, sobely, laplacian, hsv, lab, equalized, rectangle, circle, line]

titles = ["Grayscale", "Resized", "Rotated", "Flipped", "Blurred", "Edges", "Thresholded", "Adaptive Thresh",
          "Dilated", "Eroded", "Gradient", "Sobel X", "Sobel Y", "Laplacian", "HSV", "LAB", "Equalized", 
          "Rectangle", "Circle", "Line"]

# Plot images
fig, axes = plt.subplots(5, 4, figsize=(15, 15))
axes = axes.ravel()

for i in range(20):
    if len(images[i].shape) == 2:  # Grayscale images
        axes[i].imshow(images[i], cmap='gray')
    else:  # Color images
        axes[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    axes[i].set_title(titles[i])
    axes[i].axis("off")

plt.tight_layout()
plt.show()