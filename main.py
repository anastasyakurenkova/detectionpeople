import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO

# Выбор источника видео - файл или веб-камера
def get_video_source():
    root = tk.Tk()
    root.withdraw()  # Скрыть основное окно

    answer = messagebox.askyesno("Источник", "Хотите использовать веб-камеру?")
    if answer:
        return 0  # веб-камера обычно имеет индекс 0
    file_path = filedialog.askopenfilename(title="Выберите видео файл")
    return file_path

# Выбор места для сохранения видео
def get_save_path():
    root = tk.Tk()
    root.withdraw()  # Скрыть основное окно
    save_path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                             filetypes=[("MP4 files", "*.mp4")],
                                             title="Сохранить видео как")
    return save_path

def main():
    video_source = get_video_source()
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_source)

    # Получаем параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_path = get_save_path()

    # Настройка видео записи
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обнаружение объектов
        results = model(frame, conf=0.5)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf.item()
                cls = box.cls.item()

                if cls == 0:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Person: {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Запись кадра в видеофайл
        out.write(frame)

        # Показываем кадр
        cv2.imshow("Output", frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


