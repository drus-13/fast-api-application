from urllib import response
import numpy as np
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_read_item():
    response = client.get("/")
    assert response.status_code == 200

def test_read_item_bad_token():
    response = client.get("/")
    assert response.status_code == 400

def test_read_inexistent_item():
    response = client.get("/")
    assert response.status_code == 404

# def test_content():
#     response = client.get("/")
#     content = """
#         <head>
#         <meta name="viewport" content="width=device-width, initial-scale=1"/>
#         </head>
#         <body style="background-color:powderblue;">
#         <center>
#             <marquee width="525" behavior="alternate"><h1 style="color:red;font-family:Arial">Please Upload Your Scenes!</h1></marquee>
#             <h3 style="font-family:Arial">We'll Try to Predict Which of These Categories They Are:</h3><br>
#             <table align="center"><tr><td><img height="80" src="/static/original/building_default.jpg" ></td><td style="text-align:center">Building</td></tr><tr><td><img height="80" src="/static/original/forest_default.jpg" ></td><td style="text-align:center">Forest</td></tr><tr><td><img height="80" src="/static/original/glacier_default.jpg" ></td><td style="text-align:center">Glacier</td></tr><tr><td><img height="80" src="/static/original/mountain_default.jpg" ></td><td style="text-align:center">Mountain</td></tr><tr><td><img height="80" src="/static/original/sea_default.jpg" ></td><td style="text-align:center">Sea</td></tr><tr><td><img height="80" src="/static/original/street_default.jpg" ></td><td style="text-align:center">Street</td></tr></table>
#             <br/>
#             <br/>
#             <form  action="/uploadfiles" enctype="multipart/form-data" method="post">
#             <input name="files" type="file" multiple>
#             <input type="submit">
#             </form>
#             </body>
#             """
    