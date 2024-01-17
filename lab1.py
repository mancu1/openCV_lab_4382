import cv2
import numpy as np


def draw_triangle():
    width, height = 800, 600
    image = np.zeros((height, width, 3), dtype=np.uint8)

    pt1 = (200, 300)
    pt2 = (400, 100)
    pt3 = (600, 300)

    cv2.line(image, pt1, pt2, (0, 0, 255), 2)
    cv2.line(image, pt2, pt3, (0, 0, 255), 2)
    cv2.line(image, pt3, pt1, (0, 0, 255), 2)

    cv2.imwrite("triangle.png", image)


drawing = False
# Начальные и конечные координаты для прямоугольника
x1, y1, x2, y2 = -1, -1, -1, -1




def select_and_save_area():


    # Загрузка изображения
    image = cv2.imread("leaf.jpg")

    # Функция обработки события мыши
    def mouse_event(event, x, y, flags, param):
        global x1, y1, x2, y2, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x1, y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                x2, y2 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x2, y2 = x, y
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Select Area", image)

    # Создание окна для изображения и установка обработчика события мыши
    cv2.namedWindow("Select Area")
    cv2.setMouseCallback("Select Area", mouse_event)

    while True:
        cv2.imshow("Select Area", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Если нажата клавиша "s", сохраняем выделенную область
            if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
                roi = image[y1:y2, x1:x2]
                cv2.imwrite("selected_area.png", roi)
                print("Выделенная область сохранена в selected_area.png")
            else:
                print("Выделите область, прежде чем сохранить")
        elif key == 27:
            # Если нажата клавиша ESC, закрываем окно
            break

    cv2.destroyAllWindows()


def merge_images():
    image1 = cv2.imread("leaf2.jpg")
    image2 = cv2.imread("leaf.jpg")

    if image1.shape != image2.shape:
        raise ValueError("Изображения должны иметь одинаковые размеры")

    result_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)
    cv2.imwrite("merged_image.png", result_image)
