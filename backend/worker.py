# Celery worker — async review pipeline decoupled from the web process
import os
from datetime import datetime, timezone

from celery import Celery
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "review_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)


def _sync_session() -> Session:
    db_url = os.getenv("DATABASE_URL", "sqlite:///./reviews.db").replace("+aiosqlite", "")
    engine = create_engine(db_url)
    return Session(engine)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def run_review_task(self, review_id: int, pr_url: str, model: str | None = None):
    """Run a full PR review and persist results. Called by webhook + trigger endpoints."""
    from backend.models.review import Review, ReviewComment
    from backend.services.review_agent import run_review

    session = _sync_session()
    try:
        review = session.get(Review, review_id)
        if not review:
            return {"error": f"Review {review_id} not found"}

        review.status = "in_progress"
        session.commit()

        comments, _ = run_review(pr_url, model=model)

        for c in comments:
            session.add(ReviewComment(
                review_id=review_id,
                file=c.file,
                line=c.line,
                severity=c.severity,
                category=c.category,
                comment=c.comment,
                suggestion=c.suggestion,
                reproduction=getattr(c, "reproduction", None),
            ))

        review.status = "completed"
        review.completed_at = datetime.now(timezone.utc)
        session.commit()

        return {"review_id": review_id, "comment_count": len(comments)}

    except Exception as exc:
        session.rollback()
        try:
            review = session.get(Review, review_id)
            if review:
                review.status = "failed"
                session.commit()
        except Exception:
            pass
        raise self.retry(exc=exc)
    finally:
        session.close()
