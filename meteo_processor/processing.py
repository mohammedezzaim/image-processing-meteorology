import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from fpdf import FPDF

def apply_histogram_equalization(app):
    """Appliquer l'égalisation d'histogramme"""
    if not app.current_images:
        return

    image_path = os.path.join(app.data_folder, app.current_images[app.current_index])
    image = cv2.imread(image_path)

    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    app.display_processed_result(result, "Égalisation d'histogramme")
    app.processing_results["histogram_eq"] = result

def apply_gaussian_filter(app):
    """Appliquer un filtrage gaussien"""
    if not app.current_images:
        return

    image_path = os.path.join(app.data_folder, app.current_images[app.current_index])
    image = cv2.imread(image_path)

    result = cv2.GaussianBlur(image, (15, 15), 0)

    app.display_processed_result(result, "Filtrage gaussien")
    app.processing_results["gaussian_filter"] = result

def apply_canny_edges(app):
    """Appliquer la détection de contours Canny"""
    if not app.current_images:
        return

    image_path = os.path.join(app.data_folder, app.current_images[app.current_index])
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    app.display_processed_result(result, "Détection de contours Canny")
    app.processing_results["canny_edges"] = result

def apply_morphological_ops(app):
    """Appliquer des opérations morphologiques"""
    if not app.current_images:
        return

    image_path = os.path.join(app.data_folder, app.current_images[app.current_index])
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    result = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)

    app.display_processed_result(result, "Opérations morphologiques")
    app.processing_results["morphological"] = result

def apply_cloud_extraction(app):
    """Extraire les nuages de l'image"""
    if not app.current_images:
        return

    image_path = os.path.join(app.data_folder, app.current_images[app.current_index])
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    app.display_processed_result(result, "Extraction de nuages")
    app.processing_results["cloud_extraction"] = result

def apply_advanced_cloud_analysis(app):
    """Analyse avancée des nuages avec calcul de caractéristiques"""
    image_path = os.path.join(app.data_folder, app.current_images[app.current_index])
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, app.cloud_threshold.get(), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    app.cloud_properties = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        app.cloud_properties.append({
            'id': i+1,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity
        })

        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(result, f"{i+1}", (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    app.display_processed_result(result, "Analyse avancée des nuages")
    app.processing_results["advanced_cloud_analysis"] = result
    app.update_cloud_properties_visualization()

def apply_cyclone_detection(app):
    """Détecter les formations cycloniques"""
    if not app.current_images:
        return

    image_path = os.path.join(app.data_folder, app.current_images[app.current_index])
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                              param1=50, param2=30, minRadius=30, maxRadius=150)

    result = image.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(result, (x, y), r, (0, 0, 255), 3)
            cv2.circle(result, (x, y), 2, (0, 0, 255), -1)

    app.display_processed_result(result, "Détection de cyclones")
    app.processing_results["cyclone_detection"] = result

def apply_precipitation_analysis(app):
    """Analyser les zones de précipitations"""
    if not app.current_images:
        return

    image_path = os.path.join(app.data_folder, app.current_images[app.current_index])
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, light_precip = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    _, moderate_precip = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    _, heavy_precip = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    result = image.copy()
    result[light_precip == 255] = [0, 255, 255]
    result[moderate_precip == 255] = [0, 165, 255]
    result[heavy_precip == 255] = [0, 0, 255]

    app.display_processed_result(result, "Analyse des précipitations")
    app.processing_results["precipitation"] = result

def apply_optical_flow(app):
    """Calculer le flot optique pour détecter le mouvement"""
    if not app.current_images:
        return

    current_path = os.path.join(app.data_folder, app.current_images[app.current_index])
    current_image = cv2.imread(current_path)

    next_index = (app.current_index + 1) % len(app.current_images)
    next_path = os.path.join(app.data_folder, app.current_images[next_index])
    next_image = cv2.imread(next_path)

    gray1 = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                       pyr_scale=0.5, levels=3, winsize=15,
                                       iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    step = 20
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x]
            if np.sqrt(dx*dx + dy*dy) > app.optical_flow_threshold.get():
                cv2.arrowedLine(result, (x, y), (int(x + dx*3), int(y + dy*3)),
                               (255, 255, 255), 1, tipLength=0.3)

    app.display_processed_result(result, "Flot optique")
    app.processing_results["optical_flow"] = result

def apply_homomorphic_filter(app):
    """Applique un filtrage homomorphique pour améliorer le contraste"""
    if not app.current_images:
        return

    image_path = os.path.join(app.data_folder, app.current_images[app.current_index])
    image = cv2.imread(image_path)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    img_log = np.log1p(gray.astype(np.float32))
    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    gamma_l = 0.5
    gamma_h = 1.5
    c = 1.0
    d0 = 30.0

    H = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow)**2 + (j - ccol)**2)
            H[i,j] = (gamma_h - gamma_l) * (1 - np.exp(-c * (distance**2 / d0**2))) + gamma_l

    filtered = dft_shift * H
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    result = np.expm1(img_back)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result = result.astype(np.uint8)

    if len(image.shape) == 3:
        result = cv2.merge([result, result, result])

    app.display_processed_result(result, "Filtrage Homomorphique")
    app.processing_results["homomorphic_filter"] = result

def apply_multi_threshold_segmentation(app):
    """Appliquer une segmentation multi-seuils"""
    if not app.current_images:
        return

    image_path = os.path.join(app.data_folder, app.current_images[app.current_index])
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresholds = [50, 100, 150, 200]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    result = np.zeros_like(image)

    for i, thresh in enumerate(thresholds):
        if i == 0:
            mask = gray <= thresh
        else:
            mask = (gray > thresholds[i-1]) & (gray <= thresh)
        result[mask] = colors[i]

    mask = gray > thresholds[-1]
    result[mask] = colors[-1]

    app.display_processed_result(result, "Segmentation multi-seuils")
    app.processing_results["multi_threshold"] = result

def cloud_tracking(app):
    """Analyse le mouvement des nuages sur plusieurs images"""
    images = []
    for i in range(min(5, len(app.current_images))):
        img_path = os.path.join(app.data_folder, app.current_images[i])
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gray)

    flows = []
    for i in range(len(images)-1):
        flow = cv2.calcOpticalFlowFarneback(images[i], images[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)

    mean_movement = np.mean([np.sqrt(flow[...,0]**2 + flow[...,1]**2) for flow in flows])
    app.temporal_analysis_results = {
        'num_images': len(images),
        'mean_movement': mean_movement,
        'dominant_direction': calculate_dominant_direction(flows)
    }

def cyclone_tracking(app):
    """Suivi des cyclones sur plusieurs images"""
    cyclone_positions = []
    for i in range(min(5, len(app.current_images))):
        img_path = os.path.join(app.data_folder, app.current_images[i])
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                                  param1=50, param2=30, minRadius=30, maxRadius=150)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            main_cyclone = max(circles, key=lambda x: x[2])
            cyclone_positions.append((main_cyclone[0], main_cyclone[1]))

    if len(cyclone_positions) > 1:
        movements = []
        for i in range(len(cyclone_positions)-1):
            x1, y1 = cyclone_positions[i]
            x2, y2 = cyclone_positions[i+1]
            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx**2 + dy**2)
            movements.append(distance)

        mean_movement = np.mean(movements)
        direction = calculate_direction_between_points(cyclone_positions[0], cyclone_positions[-1])

        app.temporal_analysis_results['cyclone_tracking'] = {
            'num_images': len(cyclone_positions),
            'mean_movement': mean_movement,
            'direction': direction
        }

def calculate_dominant_direction(flows):
    """Calcule la direction dominante du mouvement"""
    if not flows:
        return "Non disponible"

    angles = []
    for flow in flows:
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        angles.extend(ang)

    mean_angle = np.mean(angles)
    degrees = np.degrees(mean_angle) % 360

    if 22.5 <= degrees < 67.5:
        return "Nord-Est"
    elif 67.5 <= degrees < 112.5:
        return "Est"
    elif 112.5 <= degrees < 157.5:
        return "Sud-Est"
    elif 157.5 <= degrees < 202.5:
        return "Sud"
    elif 202.5 <= degrees < 247.5:
        return "Sud-Ouest"
    elif 247.5 <= degrees < 292.5:
        return "Ouest"
    elif 292.5 <= degrees < 337.5:
        return "Nord-Ouest"
    else:
        return "Nord"

def calculate_direction_between_points(point1, point2):
    """Calcule la direction entre deux points"""
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(dy, dx)) % 360

    if 22.5 <= angle < 67.5:
        return "Nord-Est"
    elif 67.5 <= angle < 112.5:
        return "Est"
    elif 112.5 <= angle < 157.5:
        return "Sud-Est"
    elif 157.5 <= angle < 202.5:
        return "Sud"
    elif 202.5 <= angle < 247.5:
        return "Sud-Ouest"
    elif 247.5 <= angle < 292.5:
        return "Ouest"
    elif 292.5 <= angle < 337.5:
        return "Nord-Ouest"
    else:
        return "Nord"