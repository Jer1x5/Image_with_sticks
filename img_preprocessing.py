import cv2
import numpy as np

def preprocess_image(image_path, target_size=(256, 256)):
    # Read the image
    img = cv2.imread(image_path)

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get canny edges
    edges = cv2.Canny(gray, 1, 50)

    # apply morphology close to ensure they are closed
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # get contours
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # filter contours to keep only large ones
    result = img.copy()
    i = 1
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        if perimeter > 500:
            cv2.drawContours(result, c, -1, (0, 0, 255), 1)
            contour_img = np.ones_like(gray, dtype=np.uint8)*255
            cv2.drawContours(contour_img, c, -1, (0, 0, 0), 1)
            # cv2.imwrite("short_title_contour_{0}.jpg".format(i), contour_img)
            i = i + 1

    # Resize the image to the target size
    contour_image_resized = cv2.resize(contour_img, target_size)

    new_img = []
    for row in contour_image_resized:
        new_row = []
        for pixel in row:
            if pixel != 255:
                new_row.append(0)
            else:
                new_row.append(255)
        new_img.append(new_row.copy())

    return np.array(new_img, dtype='uint8')

def bresenham_line(x0, y0, x1, y1):
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    switched = False
    if x0 > x1:
        switched = True
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    if y0 < y1:
        ystep = 1
    else:
        ystep = -1

    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y0

    line = []
    for x in range(x0, x1 + 1):
        if steep:
            line.append((y,x))
        else:
            line.append((x,y))

        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
    if switched:
        line.reverse()
    return line

# Example usage
if __name__ == "__main__":
    image_path = "circle.jpeg"  # Provide the path to your image
    result_image = preprocess_image(image_path)

    # пример построения линии по двум координатам
    x1, y1 = 20, 10
    x2, y2 = 0, 0
    print(bresenham_line(x1,y1, x2, y2))

    # Display the result
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
