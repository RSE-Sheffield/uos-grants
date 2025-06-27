# connectors/models.py

from sqlalchemy import Column, Integer, String, Text, DateTime, UniqueConstraint
try:
    from open_webui.internal.db import Base
except:
    from uos_grants.connectors.db import Base
from datetime import datetime


class Researcher(Base):
    __tablename__ = "researchers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    department = Column(String, nullable=True)
    email = Column(String, nullable=True)
    telephone = Column(String, nullable=True)
    address = Column(Text, nullable=True)
    url = Column(String, unique=True, index=True, nullable=False)
    main_role = Column(String, nullable=True)
    additional_roles = Column(Text, nullable=True)
    profile = Column(Text, nullable=True)
    research_interests = Column(Text, nullable=True)
    last_response = Column(String, nullable=False)  # e.g. 200/404 for validation
    last_modified = Column(String, nullable=True)  # From sitemap <lastmod>
    last_scraped = Column(DateTime, default=datetime.utcnow)  # Last time we scraped the page

    __table_args__ = (
        UniqueConstraint('url', name='uq_researcher_url'),
    )

class StaffUrls(Base):
    __tablename__ = "staff_urls"
    
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, nullable=False)
    last_response = Column(String, nullable=False)
