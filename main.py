from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from typing import List, Dict
import os
from pydantic import BaseModel
import pickle

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
image_descriptions = {}
images = {}
similarity = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
class ImageUpload(BaseModel):
    filename: str
    contents: bytes

def generate_description(image):
    image = Image.open(image)
    description = image_to_text(image)
    return description[0]['generated_text']


async def process_file(file, db):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    opencv_image = cv2.imdecode(nparr, 1)
    description = generate_description(BytesIO(contents))
    description_enc = similarity.encode(description)
    images[file.filename] = opencv_image.tolist()
    image_descriptions[file.filename] = description_enc.tolist()
    print("done!")
    file_path = os.path.join("./static/images/", file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    db[file.filename] = image_descriptions[file.filename]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    num_files_uploaded = 0
    
    if os.path.exists('embeddings.pkl'):
        with open('embeddings.pkl', 'rb') as f:
            db = pickle.load(f)       
            for file in files:
                if file.filename in db:
                    if file.filename in image_descriptions and image_descriptions[file.filename] != db[file.filename]:
                        print("updating")
                        image_descriptions[file.filename] += db[file.filename]
                    else:
                        image_descriptions[file.filename] = db[file.filename]
                else:
                    await process_file(file, db)
        with open('embeddings.pkl', 'wb') as f_db:
            pickle.dump(db, f_db)
        num_files_uploaded += 1
        return JSONResponse(content={"files": num_files_uploaded}, status_code=200)
    else:
        db = {}
        for file in files:
            await process_file(file, db)
        with open('embeddings.pkl', 'wb+') as f_db:
            pickle.dump(db, f_db)
        num_files_uploaded += 1
        return JSONResponse(content={"files": num_files_uploaded}, status_code=200)


class SearchQuery(BaseModel):
    query: str

@app.post("/search")
async def search(search_query: SearchQuery):
    query = search_query.query
    search_query_enc = similarity.encode(query)
    similarities = {}
    with open('embeddings.pkl', 'rb') as f:
        db = pickle.load(f)
        for filename, description_enc in db.items():
            similarities[filename] = cosine_similarity(np.array(search_query_enc).reshape(1, -1), np.array(description_enc).reshape(1, -1))[0][0]
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        filtered_similarities = [t for t in sorted_similarities if t[1] > 0.3]
        if len(filtered_similarities) > 4:
            return filtered_similarities
        else:
            return sorted_similarities[:4]

@app.get("/uploaded-files", response_model=List[str])
async def get_uploaded_files():
    with open('embeddings.pkl', 'rb') as f:
        db = pickle.load(f) 
    print(list(db.keys()))
    return list(db.keys())


@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    if os.path.exists('embeddings.pkl'):
        with open('embeddings.pkl', 'rb') as f:
            db = pickle.load(f)
        if filename in db:
            del db[filename]
            with open('embeddings.pkl', 'wb') as f:
                pickle.dump(db, f)
            return {"message": "File deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    else:
        raise HTTPException(status_code=500, detail="Pickle file not found")