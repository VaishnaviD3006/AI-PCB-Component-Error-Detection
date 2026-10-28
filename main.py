"""
AI-Based Component Placement Error Detection in PCB Assembly
Baseline prototype: reference/test PCB image comparison using OpenCV.

Important:
- This baseline is image processing, not a trained AI model.
- It highlights visual differences for human review; it does not prove a component defect.
- Use well-aligned images with similar lighting for meaningful comparisons.
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {path}. Use a valid JPG or PNG file."
        )
    return image


def compare_pcb_images(reference_path: str, test_path: str,
                       pixel_threshold: int = 30,
                       min_contour_area: int = 100):
    reference = load_image(reference_path)
    test = load_image(test_path)

    # Match dimensions only; this does not correct rotation or perspective.
    test = cv2.resize(test, (reference.shape[1], reference.shape[0]))

    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    ref_gray = cv2.GaussianBlur(ref_gray, (5, 5), 0)
    test_gray = cv2.GaussianBlur(test_gray, (5, 5), 0)

    difference = cv2.absdiff(ref_gray, test_gray)
    _, mask = cv2.threshold(
        difference, pixel_threshold, 255, cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = test.copy()
    regions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            regions.append({
                "x": int(x), "y": int(y), "width": int(w),
                "height": int(h), "area": float(area)
            })

    return reference, test, mask, result, regions


def save_and_show(reference, mask, result, regions, output_dir="results"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / "difference_mask.png"), mask)
    cv2.imwrite(str(out / "highlighted_result.png"), result)

    print("Status:", "CHECK - review highlighted regions" if regions
          else "PASS - no significant image differences detected")
    print("Highlighted regions:", len(regions))
    print("Note: PASS does not guarantee that the PCB is defect-free.")

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
    plt.title("Reference PCB")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Difference Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Regions for Human Review")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out / "comparison_output.png", dpi=160, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    reference_path = input("Path to correct reference PCB image: ").strip()
    test_path = input("Path to PCB image to inspect: ").strip()
    reference, test, mask, result, regions = compare_pcb_images(
        reference_path, test_path
    )
    save_and_show(reference, mask, result, regions)
