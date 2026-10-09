"""
数据库工具函数
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy.orm import Session

from ..config import get_config


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """获取数据库会话的上下文管理器"""
    from ..models.database import init_database

    config = get_config()
    session = init_database(config.sqlite_path)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()