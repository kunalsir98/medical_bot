import sys
from ai.logger import logging

class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.get_error_message_detail(error_message, error_detail)

    @staticmethod
    def get_error_message_detail(error_message: Exception, error_detail: sys):
        _, _, exec_tb = error_detail.exc_info()
        line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename

        error_message = (
            f"Error occurred in script: [{file_name}] at line number [{line_number}] "
            f"with error message: [{error_message}]"
        )
        return error_message

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return f"{CustomException.__name__}('{self.error_message}')"

if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)  # Configure logging
        a = 5
        b = 0
        a / b
    except Exception as e:
        logging.info("Zero DivisionError occurred")
        raise CustomException(e, sys)
