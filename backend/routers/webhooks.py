# GitHub webhook receiver — verify signature, enqueue review job
import hashlib
import hmac
import os

from fastapi import APIRouter, Header, HTTPException, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import get_db
from backend.models.review import Review
from backend.services.github_service import parse_pr_url
from backend.worker import run_review_task

router = APIRouter()

WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")


def verify_signature(payload: bytes, signature: str) -> bool:
    if not WEBHOOK_SECRET:
        return True
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/github", status_code=202)
async def github_webhook(
    request: Request,
    db: AsyncSession = Depends(get_db),
    x_hub_signature_256: str = Header(default=""),
    x_github_event: str = Header(default=""),
):
    body = await request.body()

    if WEBHOOK_SECRET and not verify_signature(body, x_hub_signature_256):
        raise HTTPException(status_code=403, detail="Invalid signature")

    if x_github_event != "pull_request":
        return {"status": "ignored", "reason": f"event type: {x_github_event}"}

    payload = await request.json()
    action = payload.get("action", "")

    if action not in ("opened", "synchronize"):
        return {"status": "ignored", "reason": f"action: {action}"}

    pr_url = payload["pull_request"]["html_url"]
    owner, repo, pr_num = parse_pr_url(pr_url)

    review = Review(
        pr_url=pr_url,
        owner=owner,
        repo=repo,
        pr_number=int(pr_num),
        status="pending",
    )
    db.add(review)
    await db.commit()
    await db.refresh(review)

    # Enqueue — returns immediately, worker picks it up asynchronously
    run_review_task.delay(review.id, pr_url)

    return {"status": "accepted", "review_id": review.id}
