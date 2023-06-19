from pathlib import Path
import logging

from utilities.utils import setup_logging

setup_logging('data_preprocess.log')

logger = logging.getLogger(__name__)


def main(data_dir) -> None:
    logger.info('Preprocessing started...')


if __name__ == "__main__":
    data_dir = Path()
    main(data_dir)