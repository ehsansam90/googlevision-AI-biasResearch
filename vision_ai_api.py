import os
import io
from google.cloud import vision
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision-ai-368602-ab4a0385730f.json'

def detect_labels(path):
    client = vision.ImageAnnotatorClient()
    # The name of the image file to annotate
    file_name = os.path.abspath(path)

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    resultDict = {}

    for label in labels:
        if label.description == "Research":
            score = label.score
            resultDict['research'] = score
        if label.description == "Science":
            science = label.score
            resultDict['science'] = science

    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return likelihood_name[safe.adult], likelihood_name[safe.medical], likelihood_name[safe.spoof], likelihood_name[safe.violence], likelihood_name[safe.racy], resultDict


Folder = "White Female Researcher"
dir_list = os.listdir("Pilot Sample/"+Folder)
researches = []
image_names = []
output = pd.DataFrame()
count = 0
for image in dir_list:
    research_score = detect_labels("Pilot Sample/"+Folder+"/"+image)
    print(count)
    output = output.append(research_score, ignore_index=True)
    image_names.append(image)
    count+=1
output.index = image_names
#     researches.append(research_score)
#
# df = pd.DataFrame({"image names":image_names, "Research":researches},
#                   columns = ["image names","Research"])
print(output.head())
output.to_csv('PilotResults_LabelDetection/WhiteFemaleResearcher_LabelDetectionResults.csv')