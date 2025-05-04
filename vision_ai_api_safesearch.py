import io
import os
from google.cloud import vision
import pandas as pd
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

credentials_path = config.get('google', 'credentials_path')
def detect_safe_search(path):
    """Detects unsafe features in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

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
    return likelihood_name[safe.adult], likelihood_name[safe.medical], likelihood_name[safe.spoof], likelihood_name[safe.violence], likelihood_name[safe.racy]

for Folder in ['Asian Female Researcher','Asian Male Researcher','Black Female Researcher','Black Male Researcher','Hispanic Female Researcher',
               'Hispanic Male Researcher','White Female Researcher','White Male Researcher']:

    #detect_safe_search("resources/researcher.jpg")
    #Folder = "Asian Female Researcher"
    dir_list = os.listdir("Pilot Sample/"+Folder+"/Additional Images")
    adults = []
    medicals = []
    spoofs = []
    violences = []
    racys = []
    image_names = []
    print('Safe search analysing:'+Folder)
    n = 1
    for image in dir_list:
        adult, medical, spoof, violence, racy = detect_safe_search("Pilot Sample/"+Folder+"/Additional Images"+"/"+image)
        print("Image "+str(n))
        n +=1
        image_names.append(image)
        adults.append(adult)
        medicals.append(medical)
        spoofs.append(spoof)
        violences.append(violence)
        racys.append(racy)

    df = pd.DataFrame({"image names":image_names, "adults":adults, "medicals":medicals, "spoofs": spoofs, "violences": violences, "racys": racys},
                      columns = ["image names","adults","medicals","spoofs","violences","racys"])

    print(df.head())
    df.to_csv(f'PilotResults_SafeSearchDetection_additionImages/{Folder}_SafeSearchResults.csv')




