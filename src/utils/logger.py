from typing import List


class Logger:
    """
    A singleton logger using the Observer design pattern.
    Observers subscribe to receive log messages.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Ensure only one instance of Logger is created.
        """
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "observers"):
            self.observers: List = []

    def subscribe(self, observer):
        """
        Subscribe an observer to the logger.

        Args:
            observer: An object implementing the `update` method to handle log messages.
        """
        self.observers.append(observer)

    def unsubscribe(self, observer):
        """
        Unsubscribe an observer from the logger.

        Args:
            observer: The observer to be removed.
        """
        self.observers.remove(observer)

    def notify(self, message: str):
        """
        Notify all subscribed observers with a log message.

        Args:
            message (str): The log message.
        """
        for observer in self.observers:
            observer.update(message)

    def log(self, message: str):
        """
        Log a message to the console and notify observers.

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
        """
        Initialize the FileLogger with the specified file path.

        Args:
            file_path (str): Path to the log file.
        """
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


# Initialize a singleton instance of logger
logger = Logger()
