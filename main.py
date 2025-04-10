from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import face_recognition
import numpy as np
import cv2
import pickle
import os
from pathlib import Path
from database import SessionLocal, engine, get_db
from models import FaceEmbedding, User  # ✅ Import User model
from fastapi.responses import JSONResponse
from config import settings  # ✅ Imported settings

app = FastAPI()

# ✅ CORS Setup using settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


# Create database tables
from models import Base
Base.metadata.create_all(bind=engine)

# Train Faces & Store in PostgreSQL
@app.post("/train_faces/")
async def train_faces(
    employee_id: str = Form(...), 
    name: str = Form(...),  # Require name in the request
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image format!")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image_rgb, model="hog")
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

    if not face_encodings:
        raise HTTPException(status_code=400, detail="No face detected!")

    encoding_bytes = pickle.dumps(face_encodings[0])  # Convert array to binary
    face_entry = FaceEmbedding(employee_id=employee_id, name=name, encoding=encoding_bytes)  # Fix applied
    db.add(face_entry)
    db.commit()

    return {"message": f"Face for {employee_id} trained successfully!"}

# Recognize Face & Return Employee ID
@app.post("/recognize_face/")
async def recognize_face(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "No face detected!"})

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image_rgb, model="hog")
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

    if not face_encodings:
        return JSONResponse(status_code=400, content={"error": "No face detected!"})

    input_embedding = face_encodings[0]
    known_faces = db.query(FaceEmbedding).all()
    for face in known_faces:
        stored_encoding = pickle.loads(face.encoding)
        match = face_recognition.compare_faces([stored_encoding], input_embedding, tolerance=0.6)
        if match[0]:
            return {"employeeId": face.employee_id}  # Fixed response format

    return JSONResponse(status_code=401, content={"error": "Face not recognized!"})

FACE_STORAGE_DIR = Path(settings.face_storage_dir)  # Set directory for storing images
FACE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

@app.post("/update_face/")
async def update_face(
    employee_id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # ✅ Read uploaded file
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="❌ File is empty!")

    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="❌ Invalid image format!")

    # ✅ Detect face
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image_rgb, model="hog")
    face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

    if not face_encodings:
        raise HTTPException(status_code=400, detail="No face detected!")

    encoding_bytes = pickle.dumps(face_encodings[0])  # ✅ Convert encoding to binary

    # ✅ Check if employee exists in FaceEmbedding & User table
    existing_face = db.query(FaceEmbedding).filter(FaceEmbedding.employee_id == employee_id).first()
    user = db.query(User).filter(User.employee_id == employee_id).first()

    if not existing_face or not user:
        raise HTTPException(status_code=404, detail="Employee not found!")

    # ✅ Delete old image if it exists
    if user.image_path and os.path.exists(user.image_path):
        os.remove(user.image_path)

    # ✅ Save new image with `employeeId_filename` format
    new_filename = f"{employee_id}_{file.filename}"
    save_path = FACE_STORAGE_DIR / new_filename

    with open(save_path, "wb") as image_file:
        image_file.write(contents)

    # ✅ Update the database with new image path & encoding
    existing_face.encoding = encoding_bytes  # Update encoding
    existing_face.name = name  # Update name if changed
    user.image_path = str(save_path)  # Update image path in DB

    db.commit()

    return {"message": f"✅ Face for {employee_id} updated successfully!", "saved_image": str(save_path)}

# Run FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
