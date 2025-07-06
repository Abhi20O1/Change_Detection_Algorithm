import os
import cv2
import numpy as np

def get_image_pairs(input_dir):
    files = os.listdir(input_dir)
    before_files = [f for f in files if '~2' not in f]
    pairs = []
    for bf in before_files:
        base_name = bf.replace('.jpg', '')
        after_name = base_name + '~2.jpg'
        if after_name in files:
            pairs.append((os.path.join(input_dir, bf), os.path.join(input_dir, after_name)))
    return pairs

def highlight_changes(before_path, after_path, output_path):
    before = cv2.imread(before_path)
    after = cv2.imread(after_path)

    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_before, gray_after)

    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(after, (x, y), (x + w, y + h), (0, 0, 255), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, after)

def process_folder(input_dir, output_dir):
    pairs = get_image_pairs(input_dir)
    for before_path, after_path in pairs:
        base_name = os.path.basename(before_path).replace('.jpg', '')
        output_path = os.path.join(output_dir, base_name + '_diff.jpg')
        highlight_changes(before_path, after_path, output_path)
        print(f"[INFO] Processed {base_name}")

input_folder = 'input-images'
output_folder = 'output-images'
process_folder(input_folder, output_folder)
