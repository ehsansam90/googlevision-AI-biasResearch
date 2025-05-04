import os
import io
from google.cloud import vision
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'visionai-379417-544db22509ed.json'

def detect_labels(path):
    client = vision.ImageAnnotatorClient()
    # The name of the image file to annotate
    file_name = os.path.abspath(path)

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image,max_results=50)
    labels = response.label_annotations

    resultDict = {}

    scores = []

    for label in labels:
        scores.append(label.score)
        if label.description == "Research":
            score = label.score
            resultDict['research'] = score
        if label.description == "Science":
            science = label.score
            resultDict['science'] = science
    print(scores)
    print(len(scores))
    return resultDict


for Folder in ['Asian Male Researcher','Black Female Researcher','Black Male Researcher','Hispanic Female Researcher','Hispanic Male Researcher','White Female Researcher','White Male Researcher']:

    #Folder = "Asian Female Researcher"
    dir_list = os.listdir("Pilot Sample/"+Folder+"/Additional Images")
    researches = []
    image_names = []
    output = pd.DataFrame()
    count = 0
    print(Folder)
    for image in dir_list:
        research_score = detect_labels("Pilot Sample/"+Folder+"/Additional Images"+"/"+image)
        print(count)
        output = output.append(research_score, ignore_index=True)
        image_names.append(image)
        count+=1
    output.index = image_names
    #     researches.append(research_score)
    #
    # df = pd.DataFrame({"image names":image_names, "Research":researches},
    #                   columns = ["image names","Research"])
    output.to_csv(f'PilotResults_LabelDetection_additionImages/{Folder}_LabelDetectionResults.csv')