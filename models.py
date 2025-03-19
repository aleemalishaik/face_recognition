from database import Base  # ✅ Import Base from database.py
from sqlalchemy import Column, String, LargeBinary
from sqlalchemy import Column, String, BigInteger, ForeignKey, TIMESTAMP
from sqlalchemy.sql import func

class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    employee_id = Column(String, primary_key=True, index=True)  # ✅ Now primary key
    name = Column(String, index=True)
    encoding = Column(LargeBinary)

class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    employee_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    image_path = Column(String(255), nullable=False)
