
import logging
from typing import Optional

from open_webui.internal.db import get_db
from open_webui.models.uos_feedback import (
    GrantsFeedbackBase,
)

from open_webui.models.users import User
from fastapi.responses import HTMLResponse

from fastapi.templating import Jinja2Templates
from fastapi import APIRouter, Depends, HTTPException, status, Request

from datetime import datetime

templates = Jinja2Templates(directory="open_webui/templates")

router = APIRouter()

@router.get("/feedback/ui", response_class=HTMLResponse)
async def feedback_ui(request: Request):
    with get_db() as db:
        feedbacks = db.query(GrantsFeedbackBase).all()
        users = db.query(User).all()
        user_dict = {user.id: user.name for user in users}

        feedback_list = [
            {
                "user_name": user_dict.get(f.user_id, "Unknown User"),
                "chat_id": f"http://uos-grants/c/{f.chat_id}",
                "created_at": datetime.fromtimestamp(f.created_at).strftime("%Y-%m-%d %H:%M:%S"),
                "feedback_str": f.feedback_str,
            }
            for f in feedbacks
        ]

    return templates.TemplateResponse("feedback.html", {
        "request": request,
        "feedbacks": feedback_list
    })