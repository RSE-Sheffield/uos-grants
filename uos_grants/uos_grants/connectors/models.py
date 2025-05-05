# connectors/models.py

from sqlalchemy import Column, Integer, String, Text
from .db import Base

class Researcher(Base):
    __tablename__ = "researchers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    department = Column(String, nullable=True)
    email = Column(String, unique=False, nullable=True)
    telephone = Column(String, unique=False, nullable=True)
    address = Column(Text, nullable=True)
    url = Column(String, unique=True)
    main_role = Column(String, nullable=True)
    additional_roles = Column(Text, nullable=True)
    profile = Column(Text, nullable=True)
    research_interests = Column(Text, nullable=True)
    last_response = Column(String, nullable=False)

class StaffUrls(Base):
    __tablename__ = "staff_urls"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, nullable=False)
    last_response = Column(String, nullable=False)
