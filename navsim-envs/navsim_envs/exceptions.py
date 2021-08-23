class EnvNotInitializedError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        message = "Environment not initialized. Please call reset()."
        if self.message:
            message += self.message
        else:
            return message
