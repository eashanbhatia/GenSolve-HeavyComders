import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []

    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []

        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)

        path_XYs.append(XYs)

    return path_XYs

def draw_shapes_on_image(paths_XYs, img_size=(500, 500)):
    img = np.zeros(img_size, dtype=np.uint8)
    for XYs in paths_XYs:
        for XY in XYs:
            pts = np.array(XY, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=255, thickness=2)
    return img

def detect_and_segment_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    
    segmented_shapes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        shape = mask[y:y+h, x:x+w]
        segmented_shapes.append((shape, (x, y, w, h)))
    
    return segmented_shapes

def complete_shape_with_inpainting(shape, original_img, bbox):
    # Extract the region of interest from the original image
    x, y, w, h = bbox
    roi = original_img[y:y+h, x:x+w]

    # Ensure shape is properly sized and binary
    shape_resized = cv2.resize(shape, (w, h))
    
    # Create a single-channel mask where the shape exists
    mask = np.zeros_like(roi[:, :, 0])  # Single channel
    mask[shape_resized == 255] = 255

    # Perform inpainting
    inpainted_shape = cv2.inpaint(roi, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return inpainted_shape


def regularize_shape(completed_shape):
    # Regularization can include smoothing, erosion, or dilation to perfect the shape
    kernel = np.ones((3, 3), np.uint8)
    regularized_shape = cv2.morphologyEx(completed_shape, cv2.MORPH_CLOSE, kernel)
    return regularized_shape

def plot_image(img, title=''):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    # Example CSV file path
    csv_path = './occlusion2.csv'

    # Read the CSV file
    paths_XYs = read_csv(csv_path)

    # Draw shapes on a binary image
    img = draw_shapes_on_image(paths_XYs)
    original_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Plot the original binary image
    plot_image(original_img, title='Original Doodle')

    # Detect and segment contours
    segmented_shapes = detect_and_segment_contours(img)

    for i, (shape, bbox) in enumerate(segmented_shapes):
        completed_shape = complete_shape_with_inpainting(shape, original_img, bbox)
        regularized_shape = regularize_shape(completed_shape)
        plot_image(regularized_shape, title=f'Regularized and Completed Shape {i+1}')

if __name__ == "__main__":
    main()
