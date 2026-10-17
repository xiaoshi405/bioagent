"""Configure console and file logging."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import Config

def get_logger(name: str, log_file: str) -> logging.Logger:
    """Configure a logger once, with a console fallback if file logging fails."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    try:
        # Do not add duplicate handlers.
        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Keep console logging if the log file is unavailable.
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        try:
            log_dir = Path(Config().LOG_DIR).expanduser()
            if not log_dir.is_absolute():
                project_root = Path(__file__).resolve().parent.parent.parent
                log_dir = project_root / log_dir
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                log_dir / log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as exc:
            logger.warning("文件日志初始化失败，已退化为控制台日志: %s", exc)

        return logger
    except Exception as exc:
        # Logging failures must not stop request handling.
        logging.getLogger().warning("Logger %s 初始化失败: %s", name, exc)
        return logger
