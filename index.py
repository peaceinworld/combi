import os
from flask import Flask, render_template, request, redirect, abort, flash, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import os
from _datetime import datetime

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


YUKLEME_KLASORU = 'static/yuklemeler'
UZANTILAR = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = YUKLEME_KLASORU
app.secret_key = "Flask_Dosya_Yukleme_Ornegi"



################### Uzantı Kontrolü #########################

def uzanti_kontrol(dosyaadi):
   return '.' in dosyaadi and \
   dosyaadi.rsplit('.', 1)[1].lower() in UZANTILAR

#############################################################



# Ana Sayfa
@app.route("/")
def anaSayfa():
    return render_template("index.html")


# Form ile dosya yükleme sayfası
@app.route('/dosyayukleme')
def dosyayukleme():
   return render_template("dosyayukleme.html")


# Form ile dosya yükleme sayfası - Sonuç
@app.route('/dosyayukleme/<string:dosya>')
def dosyayuklemesonuc(dosya):
   return render_template("dosyayukleme.html", dosya=dosya)



# Form ile dosya yükleme işlemi
@app.route('/dosyayukle', methods=['POST'])
def dosyayukle():
	
   if request.method == 'POST':
					
		# formdan dosya gelip gelmediğini kontrol edelim
      if 'dosya' not in request.files:
         flash('Dosya seçilmedi')
         return redirect('dosyayukleme')         
							
		# kullanıcı dosya seçmemiş ve tarayıcı boş isim göndermiş mi
      dosya = request.files['dosya']					
      if dosya.filename == '':
         flash('Dosya seçilmedi')
         return redirect('dosyayukleme')
					
		# gelen dosyayı güvenlik önlemlerinden geçir
      if dosya and uzanti_kontrol(dosya.filename):
         now = datetime.now() # current date and time
         date_time = now.strftime("%m%d%Y%H%M%S")
         dosyaadi = date_time+".png"
         dosya.save(os.path.join(app.config['UPLOAD_FOLDER'], dosyaadi))
         #return redirect(url_for('dosyayukleme',dosya=dosyaadi))
         return redirect('dosyayukleme/' + dosyaadi)
      else:
         flash('İzin verilmeyen dosya uzantısı')
         return redirect('dosyayukleme')
							
   else:
      abort(401)      
@app.route('/kontrol',methods = ['POST'])
def kontrol():
    if request.method == 'POST':
        CWD_PATH = os.getcwd()
        yuklenen = os.listdir("static/yuklemeler")[0]
        MODEL_NAME = 'new_graph'
        IMAGE_NAME = yuklenen
        # Grab path to current working directory
        

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
        
        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
        
        # Path to image
        PATH_TO_IMAGE = os.path.join(CWD_PATH,'static','yuklemeler',IMAGE_NAME)
        # Number of classes the object detector can identify
        NUM_CLASSES = 4
        
        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        
        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
            sess = tf.Session(graph=detection_graph)
        
        # Define input and output tensors (i.e. data) for the object detection classifier
        
        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        
        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        
        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        
        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(PATH_TO_IMAGE)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0)
        
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        classprint = np.squeeze(classes).astype(np.int32)
        son = classprint[0]
        if son == 1:
            return redirect("https://www.demirdokum.com.tr/urunler/nitromix-duvar-tipi-hermetik-yogusmal-kombi-13952.html")
        elif son == 2:
            return redirect("https://www.demirdokum.com.tr/urunler/atromix-duvar-tipi-yogusmal-kombi-27328.html")
        else:
            return redirect('kontrol/' + str(son))
        

if __name__ == "__main__":
    app.run(debug=True)