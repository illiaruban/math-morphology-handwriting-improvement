import os
import time
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from hw_morphology_algorithm import handwriting_enhancement_algorithm
from stack_path import stack_path_opening
from talbot_algorithm import talbot_algorithm

DATASET_DIR = "./bentham_dataset"
IMAGE_LIMIT = 50
MAX_WIDTH = 600

L = 15
DARKEN_VALUE = 45

def load_image_paths(dataset_dir, limit):
    paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                paths.append(os.path.join(root, file))
    return sorted(paths)[:limit]

def resize_if_needed(img, max_width):
    h, w = img.shape

    if w <= max_width:
        return img

    scale = max_width / w
    new_h = int(h * scale)

    return cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)

def measure_time(func):
    start = time.perf_counter()
    func()
    end = time.perf_counter()
    return end - start

def main():
    image_paths = load_image_paths(DATASET_DIR, IMAGE_LIMIT)

    if len(image_paths) == 0:
        raise Exception("Немає зображень у bentham_dataset")

    warm_img = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)

    if warm_img is None:
        raise Exception("Не вдалося прочитати перше зображення")

    warm_img = resize_if_needed(warm_img, MAX_WIDTH)

    print("Warm-up...")

    handwriting_enhancement_algorithm(
        warm_img,
        L=L,
        darken_value=DARKEN_VALUE,
        blur=True
    )

    stack_path_opening(
        warm_img,
        L=L
    )

    talbot_algorithm(
        warm_img,
        L=L,
        reduce_gray_step=2,
        dark_text=True
    )

    results = []

    for i, path in enumerate(image_paths, 1):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = resize_if_needed(img, MAX_WIDTH)

        print(f"[{i}/{len(image_paths)}] {os.path.basename(path)}")

        new_time = measure_time(
            lambda: handwriting_enhancement_algorithm(
                img,
                L=L,
                darken_value=DARKEN_VALUE,
                blur=True
            )
        )

        stack_time = measure_time(
            lambda: stack_path_opening(
                img,
                L=L
            )
        )

        talbot_time = measure_time(
            lambda: talbot_algorithm(
                img,
                L=L,
                reduce_gray_step=2,
                dark_text=True
            )
        )

        results.append({
            "image": os.path.basename(path),
            "proposed_algorithm": new_time,
            "stack_path_opening": stack_time,
            "talbot_algorithm": talbot_time
        })

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)

    mean_times = df[[
        "proposed_algorithm",
        "stack_path_opening",
        "talbot_algorithm"
    ]].mean()

    labels = [
        "Запропонований алгоритм",
        "Стекове шляхове розкриття",
        "Алгоритм Талбота"
    ]

    values = mean_times.values

    plt.figure(figsize=(8, 5))

    bars = plt.bar(labels, values)

    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{y:.3f}",
            ha="center",
            va="bottom"
        )

    plt.title("Порівняння часу виконання алгоритмів")
    plt.ylabel("Час виконання, с")

    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("final_bar_chart.png", dpi=300)
    plt.show()

    print("\nСередній час виконання:")
    print(mean_times)

if __name__ == "__main__":
    main()