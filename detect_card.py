import cv2
import numpy as np

def detect_card_boxes(image,resize_height):


    orig = image.copy()
    height, width = image.shape[:2]
    
    # Resize for faster processing
    
    scale = resize_height / height
    image = cv2.resize(image, (int(width * scale), resize_height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    gray = cv2.equalizeHist(gray)

    # Edge detection
    edged = cv2.Canny(gray, 100, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 50
    max_area = 0.8 * image.shape[0] * image.shape[1]
    card_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Approximate the contour to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Card has 4 corners
        if len(approx) == 4:
            card_contours.append(approx)

    #return card_contours

    # Draw the detected contours
    output = image.copy()
    cv2.drawContours(output, card_contours, -1, (0, 255, 0), 3)

    # Show results
    cv2.imshow("Original Image", image)
    cv2.imshow("Detected Cards", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    image_path = "data_for_testing/test.jpg"  # Replace with your image path
    image = cv2.imread(image_path)
    detect_card_boxes(image,1000)