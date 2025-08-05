import logging
import inspect


class Logger:
    def __init__(self, level=logging.INFO, name=__name__):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s - %(caller_filename)s:%(caller_lineno)d')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def set_level(self, level):
        self.logger.setLevel(level)

    def debug(self, message):
        self.logger.debug(message, extra=self._get_extra_info())

    def info(self, message):
        self.logger.info(message, extra=self._get_extra_info())

    def warning(self, message):
        self.logger.warning(message, extra=self._get_extra_info())

    def error(self, message):
        self.logger.error(message, extra=self._get_extra_info())

    def critical(self, message):
        self.logger.critical(message, extra=self._get_extra_info())

    def _get_extra_info(self):
        frame = inspect.currentframe().f_back.f_back
        caller_filename = inspect.getframeinfo(frame).filename
        caller_lineno = inspect.getframeinfo(frame).lineno
        return {'caller_filename': caller_filename, 'caller_lineno': caller_lineno}


logger = Logger()
logger.set_level(logging.INFO)


# 示例用法
# if __name__ == "__main__":
#     logger = Logger()
#     logger.info("This is an info message")
#     logger.debug("This is a debug message")

#     logger.set_level(logging.DEBUG)
#     logger.debug("This is a debug message with DEBUG level")

#     logger.set_level(logging.WARNING)
#     logger.info("This is an info message, but will not be shown because the log level is set to WARNING")
#     logger.warning("This is a warning message")
