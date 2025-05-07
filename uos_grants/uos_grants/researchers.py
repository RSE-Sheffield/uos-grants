import asyncio
import re
from typing import List

import aiohttp
import requests
from bs4 import BeautifulSoup, element
from langchain_core.utils.html import extract_sub_links


class Publication:
    def __init__(self, soup: element.Tag):
        self.soup = soup
        self.title = self._retrieve_title()
        self.author = self._retrieve_authors()
        self.journal = self._retrieve_journal()

    def _retrieve_title(self):
        title = self.soup.find("a", target="_top")
        if title is None:
            author = self.soup.find("span", class_="author")
            title = author.find_next_sibling(string=True).strip()
            return title
        try:
            self.doi = title.get("href")
        except AttributeError:
            self.doi = ""
        return title.text

    def _retrieve_authors(self):
        authors = self.soup.find("span", class_="author")
        return authors.text

    def _retrieve_journal(self):
        journal = self.soup.find("cite")
        return journal.text if journal else ""

    def __str__(self):
        return f"Title: {self.title}\nAuthors: {self.author}\nJournal: {self.journal}"


from bs4 import BeautifulSoup
import re
from typing import List, Optional


class Researcher:
    def __init__(self, html: str, url: str):
        """
        Represents a University of Sheffield researcher profile page.

        Args:
            html (str): HTML content of the webpage.
            url (str): URL of the profile page.
        """
        self.html = html
        self.url = url
        self.soup = BeautifulSoup(html, "lxml")
        self.person_info = self.soup.find("div", class_=re.compile("personinfo"))
        self.contact_info = self.soup.find("div", class_=re.compile("contactinfo"))

    @property
    def name(self) -> Optional[str]:
        heading = self.person_info.find("h1")
        if heading:
            return re.sub(r"\s+", " ", heading.text.strip())
        return None

    @property
    def department(self) -> Optional[str]:
        dept = self.soup.find("div", class_="deptname")
        return dept.text.strip() if dept else None

    @property
    def main_role(self) -> Optional[str]:
        role = self.person_info.find("p", class_="mainrole")
        return role.text.strip() if role else None

    @property
    def additional_roles(self) -> Optional[str]:
        roles = self.person_info.find_all("p", class_="roles")
        return ", ".join(r.text.strip() for r in roles) if roles else None

    @property
    def email(self) -> Optional[str]:
        email_tag = self.contact_info.find("a", title="Email")
        return email_tag.text.strip() if email_tag else None

    @property
    def telephone(self) -> Optional[str]:
        phone_tag = self.contact_info.find("a", title="Telephone")
        return phone_tag.text.strip() if phone_tag else None

    @property
    def address(self) -> Optional[str]:
        addr = self.contact_info.find("div", class_="address")
        if addr:
            cleaned = re.sub(r"\s+", " ", addr.text).strip()
            return cleaned.replace(self.name or "", "").strip()
        return None

    @property
    def profile(self) -> Optional[str]:
        profile = self.soup.find("dl", class_=re.compile("profiletext"))
        if profile:
            return "\n\n".join(p.text.strip() for p in profile.find_all("p"))
        return None

    @property
    def research_interests(self) -> Optional[str]:
        ri = self.soup.find("dl", re.compile("focusresinterests"))
        if ri:
            return "\n\n".join(p.text.strip() for p in ri.find_all("p"))
        return None

    @property
    def qualifications(self) -> Optional[str]:
        quals = self.soup.find("dl", re.compile("focusqualifications"))
        if quals:
            return "\n".join(q.text.strip() for q in quals.find_all("li"))
        return None

    @property
    def publications(self) -> Optional[str]:
        pub_list = self.soup.find("ul", re.compile("publicationsList"))
        if pub_list:
            return "\n\n".join(li.text.strip() for li in pub_list.find_all("li"))
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "department": self.department,
            "email": self.email,
            "telephone": self.telephone,
            "address": self.address,
            "url": self.url,
            "main_role": self.main_role,
            "additional_roles": self.additional_roles,
            "profile": self.profile,
            "research_interests": self.research_interests,
        }

    def __str__(self) -> str:
        fields = [
            f"Name: {self.name}" if self.name else None,
            f"Department: {self.department}" if self.department else None,
            f"URL: {self.url}",
            f"Main role: {self.main_role}" if self.main_role else None,
            f"Additional roles: {self.additional_roles}" if self.additional_roles else None,
            f"Email: {self.email}" if self.email else None,
            f"Telephone: {self.telephone}" if self.telephone else None,
            f"Address: {self.address}" if self.address else None,
            f"Profile:\n{self.profile}" if self.profile else None,
            f"Research Interests:\n{self.research_interests}" if self.research_interests else None,
            f"Qualifications:\n{self.qualifications}" if self.qualifications else None,
        ]
        return "\n\n".join(filter(None, fields))


class ResearcherAsync:
    def __init__(self, url: str, html: str = None):
        """
        EXPERIMENTAL: MAY NOT BEHAVE AS EXPECTED
        Async class to fetch and parse researcher profile from a UoS staff webpage.

        Args:
            url (str): URL of the researcher's profile page.
        """
        self.url = url
        self.response = requests.get(url)
        if self.response.status_code == 200:
            self.html = self.response.text
            self.soup = BeautifulSoup(self.html, "lxml")
            self.person_info = self.soup.find(
                "div", class_=re.compile("personinfo")
            )
            self.contact_info = self.soup.find(
                "div", class_=re.compile("contactinfo")
            )
            self.page_exists = True
        else:
            self.page_exists = False

    async def fetch_page(url):
        """
        Fetches the page asynchronously with timeout handling.
        """  # Set a 10-second timeout
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
            except asyncio.TimeoutError:
                print(f"Timeout error while fetching {url}")
            except aiohttp.ClientError as e:
                print(f"HTTP error {e} while fetching {url}")

        return None  # Return None if fetching fails

    @classmethod
    async def create(cls, url: str):
        html = await cls.fetch_page(url)
        if html:
            return cls(url, html)
        else:
            return cls(url)

    @property
    def profile(self) -> str:
        profile = self.soup.find("dl", class_=re.compile("profiletext"))
        return (
            "\n\n".join(
                [paragraph.text for paragraph in profile.find_all("p")]
            )
            if profile
            else None
        )

    @property
    def name(self) -> str:
        person_name = (
            self.person_info.find("h1").text.strip()
            if self.person_info
            else None
        )
        return (
            re.sub(r"\s+", " ", person_name).strip() if person_name else None
        )

    @property
    def department(self) -> str:
        department = self.soup.find("div", class_="deptname")
        return department.text.strip() if department else None

    @property
    def main_role(self) -> str:
        if not self.person_info:
            return None
        main_role = (
            self.person_info.find("p", class_="mainrole").text
            if self.person_info.find("p", class_="mainrole")
            else None
        )
        roles_string = f"{main_role}" if main_role else ""
        return roles_string if roles_string else None

    @property
    def additional_roles(self) -> str:
        if not self.person_info:
            return None
        additional_roles = [
            role.text.strip()
            for role in self.person_info.find_all("p", class_="roles")
        ]
        if additional_roles:
            roles_string += (
                f"{', '.join(additional_roles)}"
            )
        return roles_string if roles_string else None

    @property
    def roles(self) -> str:
        roles = []
        if self.main_role:
            roles.append(self.main_role)
        if self.additional_roles:
            roles.append(self.additional_roles)
        return f"Main Role: {roles[0]}\nAdditional Roles: {', '.join(roles[1])} " if roles else ""

    @property
    def email(self) -> str:
        email = (
            self.contact_info.find("a", title="Email")
            if self.contact_info
            else None
        )
        return email.text if email else None

    @property
    def telephone(self) -> str:
        telephone = (
            self.contact_info.find("a", title="Telephone")
            if self.contact_info
            else None
        )
        return telephone.text if telephone else None

    @property
    def address(self) -> str:
        address = (
            self.contact_info.find("div", class_="address")
            if self.contact_info
            else None
        )
        return (
            re.sub(r"\s+", " ", address.text)
            .strip()
            .replace(self.name, "")
            .strip()
            if address
            else None
        )

    @property
    def research_interests(self) -> str:
        research_interests = self.soup.find(
            "dl", re.compile("focusresinterests")
        )
        if research_interests:
            research_interests = research_interests.find_all("p")
        return (
            "\n\n".join([interest.text for interest in research_interests])
            if research_interests
            else None
        )

    def __dict__(self):
        return {
            "name": self.name,
            "department": self.department,
            "url": self.url,
            "main_role": self.main_role,
            "additional_roles": self.additional_roles,
            "email": self.email,
            "telephone": self.telephone,
            "address": self.address,
            "profile": self.profile,
            "research_interests": self.research_interests,
        }

    def __str__(self):
        return (
            "\n\n".join(
                filter(
                    None,
                    [
                        f"Name: {self.name}" if self.name else None,
                        (
                            f"Department: {self.department}"
                            if self.department
                            else None
                        ),
                        f"Website: {self.url}" if self.url else None,
                        f"Roles:\n {self.roles}" if self.roles else None,
                        f"Email: {self.email}" if self.email else None,
                        (
                            f"Telephone: {self.telephone}"
                            if self.telephone
                            else None
                        ),
                        f"Address: {self.address}" if self.address else None,
                        f"Profile: {self.profile}" if self.profile else None,
                        (
                            f"Research Interests: {self.research_interests}"
                            if self.research_interests
                            else None
                        ),
                    ],
                )
            )
            if self.research_interests
            else ""
        )
