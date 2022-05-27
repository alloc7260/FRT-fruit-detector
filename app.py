from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import CognitiveServicesCredentials
from msrest.authentication import ApiKeyCredentials
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import numpy as np
import os
from flask import *
from PIL import Image

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/save', methods = ['POST'])
def save():
    try:
        blob = request.files['file']
        im = Image.open(blob)
        im.save("./static/produce.jpg")
        prediction_endpoint = 'https://cvdemotestd2-prediction.cognitiveservices.azure.com/'
        prediction_key = '520cd02cf0d649fe985ed25fc453f948'
        project_id = '970790a7-c161-461b-a9b4-e1463139ac39'
        model_name = 'Iteration1'
        print([prediction_endpoint,prediction_key,project_id,model_name])
        prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        predictor = CustomVisionPredictionClient(prediction_endpoint, prediction_credentials)
        image_file = './static/produce.jpg'
        print('Started Detecting objects in', image_file)
        image = Image.open(image_file)
        h, w, ch = np.array(image).shape
        with open(image_file,'rb') as image_data :
            results = predictor.detect_image(project_id,model_name,image_data)
        fig = plt.figure()
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        lineWidth = int(w/200)
        color = 'magenta'
        for prediction in results.predictions :
            if (prediction.probability*100)>90 :
                left = prediction.bounding_box.left * w
                top = prediction.bounding_box.top * h
                height = prediction.bounding_box.height * h
                width= prediction.bounding_box.width * w
                points = ((left,top),(left+width,top),(left+width,top+height),(left,top+height),(left,top))
                draw.line(points,fill=color,width=lineWidth)
                plt.annotate(prediction.tag_name+": {0:.2f}%".format(prediction.probability),(left+5,top-10))
        plt.imshow(image)
        outputfile = './static/output.jpg'
        fig.savefig(outputfile)
        print('Results saved in ',outputfile)
        return render_template('output.html')
    except TypeError:
        pass

if __name__ == '__main__':
    app.run()