from typing import List


class Logger:
    """
    A simple logger using the Observer design pattern.
    Observers subscribe to receive log messages.
    """

    def __init__(self):
        self.observers: List = []

    def subscribe(self, observer):
        self.observers.append(observer)

    def unsubscribe(self, observer):
        self.observers.remove(observer)

    def notify(self, message: str):
        for observer in self.observers:
            observer.update(message)

    def log(self, message: str):
        """
        Log a message to console and notify observers.

        Args:
            message (str): The message to log.
        """
        print(message)
        self.notify(message)


class FileLogger:
    """
    Logs messages to a file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def update(self, message: str):
        """
        Append a log message to the log file.

        Args:
            message (str): The log message to be written to the log file.

        Raises:
            IOError: If there is an issue writing to the log file.
        """
        with open(self.file_path, "a") as log_file:
            log_file.write(message + "\n")