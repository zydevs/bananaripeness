from flask import Flask, render_template, request, redirect, session, url_for
from skimage.feature import graycomatrix, graycoprops
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)
app.secret_key = 'secret_key_for_session'

# Folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Halaman tentang aplikasi
@app.route('/about')
def about():
    return render_template('about.html')

# Halaman Template Matching
@app.route('/check')
def check():
    return render_template('check.html')

# Halaman unggah gambar
@app.route('/upload')
def upload_page():
    return render_template('upload.html')

# Proses unggah gambar
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

# Validasi ekstensi file
def allowed_file(filename):
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Menampilkan gambar yang sudah diunggah
@app.route('/preview')
def preview():
    filename = session.get('filename')
    if not filename:
        return redirect(url_for('upload_page'))

    file_url = url_for('static', filename='uploads/' + filename)
    return render_template('preview.html', file_url=file_url)

# Proses segmentasi dan klasifikasi
@app.route('/process')
def process():
    filename = session.get('filename')
    if not filename:
        return redirect(url_for('upload_page'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        return "Gagal membaca gambar.", 400

    # Konversi BGR ke RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb_name = 'rgb_' + filename
    rgb_path = os.path.join(app.config['UPLOAD_FOLDER'], rgb_name)
    cv2.imwrite(rgb_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Konversi ke ruang warna LAB
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    lab_name = 'lab_' + filename
    lab_path = os.path.join(app.config['UPLOAD_FOLDER'], lab_name)
    cv2.imwrite(lab_path, cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR))

    # Segmentasi dengan K-Means
    pixel_values = img_lab.reshape((-1, 3)).astype(np.float32)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(pixel_values)
    segmented_img = kmeans.cluster_centers_[labels].reshape(img_lab.shape).astype(np.uint8)

    seg_name = 'segment_' + filename
    seg_path = os.path.join(app.config['UPLOAD_FOLDER'], seg_name)
    cv2.imwrite(seg_path, cv2.cvtColor(segmented_img, cv2.COLOR_LAB2BGR))

    # Ekstraksi fitur tekstur menggunakan GLCM
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

    # Nilai referensi fitur untuk klasifikasi
    class_features = {
        "Matang": np.array([86.45, 0.9575, 0.7324, 0.9819]),
        "Setengah Matang": np.array([192.06, 0.9483, 0.6363, 0.9671]),
        "Mentah": np.array([158.48, 0.9528, 0.6602, 0.9769]),
        "Busuk": np.array([343.8895, 0.9653, 0.7575, 0.9656])
    }

    current_features = np.array([contrast, correlation, energy, homogeneity])
    min_distance = float('inf')
    predicted_class = None

    # Klasifikasi berdasarkan jarak Euclidean
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

if __name__ == '__main__':
    app.run(debug=True)
