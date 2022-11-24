from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from flask import Flask, render_template, request
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

image_file = './static/produce.jpg'
outputfile = './static/output.jpg'
prediction_endpoint = 'https://cvdemotestd2-prediction.cognitiveservices.azure.com/'
project_id = '970790a7-c161-461b-a9b4-e1463139ac39'
prediction_key = '520cd02cf0d649fe985ed25fc453f948'
model_name = 'Iteration1'

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(prediction_endpoint, prediction_credentials)

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/save', methods = ['POST'])
def save():
    blob = request.files['file']
    Image.open(blob).save(image_file)

    with open(image_file,'rb') as image_data :
        results = predictor.detect_image(project_id,model_name,image_data)

    image = Image.open(image_file)
    h, w, ch = np.array(image).shape
    draw = ImageDraw.Draw(image)
    fig = plt.figure()
    plt.axis('off')
    for prediction in results.predictions :
        if (prediction.probability*100)>90 :
            left = prediction.bounding_box.left * w
            top = prediction.bounding_box.top * h
            height = prediction.bounding_box.height * h
            width= prediction.bounding_box.width * w
            points = ((left,top),(left+width,top),(left+width,top+height),(left,top+height),(left,top))
            draw.line(points,fill='magenta',width=int(w/200))
            plt.annotate(prediction.tag_name+":{0}%".format(int(prediction.probability*100)),(left,top))
    plt.imshow(image)
    fig.savefig(outputfile)
    print('Results saved in ',outputfile)

    return render_template('output.html')

if __name__ == '__main__':
    app.run()