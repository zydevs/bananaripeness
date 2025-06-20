from flask import Flask, render_template, request, redirect, session, url_for
from skimage.feature import graycomatrix, graycoprops
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import glob  

app = Flask(__name__)
app.secret_key = 'secret_key_for_session'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- HALAMAN --------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_post():
    file = request.files.get('image')
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['filename'] = filename
        return redirect(url_for('preview'))
    return redirect(url_for('upload_page'))

@app.route('/preview')
def preview():
    filename = session.get('filename')
    if not filename:
        return redirect(url_for('upload_page'))
    file_url = url_for('static', filename='uploads/' + filename)
    return render_template('preview.html', file_url=file_url)

@app.route('/check', methods=['GET'])
@app.route('/check')
def check():
    import glob  # Pastikan glob diimport

    main_filename = session.get('filename')
    if not main_filename:
        return redirect(url_for('upload_page'))

    main_path = os.path.join(app.config['UPLOAD_FOLDER'], main_filename)

    # Ambil semua file .jpg, .jpeg, .png dari static/matching/
    matching_folder = os.path.join('static', 'matching')
    template_paths = (
        glob.glob(os.path.join(matching_folder, '*.jpg')) +
        glob.glob(os.path.join(matching_folder, '*.jpeg')) +
        glob.glob(os.path.join(matching_folder, '*.png'))
    )

    best_match_val = float('-inf')
    best_detected_img = None
    best_detections = 0

    for template_path in template_paths:
        detected_name, num_detections, match_val = run_template_matching(main_path, template_path)

        if match_val > best_match_val and detected_name is not None:
            best_match_val = match_val
            best_detected_img = detected_name
            best_detections = num_detections

    if best_detected_img is None:
        return "Gagal membaca gambar atau tidak ada template yang cocok.", 500

    return render_template(
        'check.html',
        detected_img=url_for('static', filename='uploads/' + best_detected_img),
        num_detections=best_detections,
        max_val=round(best_match_val, 3)
    )


@app.route('/process')
def process():
    filename = session.get('filename')
    if not filename:
        return redirect(url_for('upload_page'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        return "Gagal membaca gambar.", 400

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_name = 'rgb_' + filename
    rgb_path = os.path.join(app.config['UPLOAD_FOLDER'], rgb_name)
    cv2.imwrite(rgb_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab_name = 'lab_' + filename
    lab_path = os.path.join(app.config['UPLOAD_FOLDER'], lab_name)
    cv2.imwrite(lab_path, cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR))

    pixel_values = img_lab.reshape((-1, 3)).astype(np.float32)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(pixel_values)
    segmented_img = kmeans.cluster_centers_[labels].reshape(img_lab.shape).astype(np.uint8)

    seg_name = 'segment_' + filename
    seg_path = os.path.join(app.config['UPLOAD_FOLDER'], seg_name)
    cv2.imwrite(seg_path, cv2.cvtColor(segmented_img, cv2.COLOR_LAB2BGR))

    segmented_l_channel = cv2.split(segmented_img)[0]

    try:
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(segmented_l_channel, distances=distances, angles=angles, symmetric=True, normed=True)

        contrast = np.mean(graycoprops(glcm, 'contrast'))
        correlation = np.mean(graycoprops(glcm, 'correlation'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    except Exception as e:
        return f"Ekstraksi fitur gagal: {str(e)}", 500

    class_features = {
        "Matang": np.array([86.45, 0.9575, 0.7324, 0.9819]),
        "Setengah Matang": np.array([192.06, 0.9483, 0.6363, 0.9671]),
        "Mentah": np.array([158.48, 0.9528, 0.6602, 0.9769]),
        "Busuk": np.array([343.8895, 0.9653, 0.7575, 0.9656])
    }

    current_features = np.array([contrast, correlation, energy, homogeneity])
    min_distance = float('inf')
    predicted_class = None

    for class_name, feature_vector in class_features.items():
        distance = np.linalg.norm(current_features - feature_vector)
        if distance < min_distance:
            min_distance = distance
            predicted_class = class_name

    maturity = f"Pisang {predicted_class}"

    return render_template(
        'output.html',
        rgb_img=url_for('static', filename='uploads/' + rgb_name),
        lab_img=url_for('static', filename='uploads/' + lab_name),
        segment_img=url_for('static', filename='uploads/' + seg_name),
        maturity_result=maturity,
        contrast=round(contrast, 4),
        correlation=round(correlation, 4),
        energy=round(energy, 4),
        homogeneity=round(homogeneity, 4)
    )

# -------------------- FUNGSI MATCHING TEMPLATE --------------------

def allowed_file(filename):
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def run_template_matching(main_image_path, template_path, threshold=0.5, method=cv2.TM_CCOEFF_NORMED):
    main_image = cv2.imread(main_image_path)
    template = cv2.imread(template_path)

    if main_image is None or template is None:
        return None, 0, 0

    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Cek ukuran: template tidak boleh lebih besar dari gambar utama
    if template_gray.shape[0] > main_gray.shape[0] or template_gray.shape[1] > main_gray.shape[1]:
        print(f"Template {template_path} Gambar tidak cocok, dilewati.")
        return None, 0, 0

    w, h = template_gray.shape[::-1]
    res = cv2.matchTemplate(main_gray, template_gray, method)
    loc = np.where(res >= threshold)

    detected = main_image.copy()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(detected, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

    detected_name = 'detected_' + os.path.basename(main_image_path)
    detected_path = os.path.join(app.config['UPLOAD_FOLDER'], detected_name)
    cv2.imwrite(detected_path, detected)

    num_detections = len(list(zip(*loc[::-1])))
    max_val = np.max(res)

    return detected_name, num_detections, max_val

# -------------------- RUN --------------------
if __name__ == '__main__':
    app.run(debug=True)
