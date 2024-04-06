from jproperties import Properties


class ConfigLoader:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls._load_config()
        return cls._instance

    @staticmethod
    def _load_config():
        config = Properties()
        with open('config.properties', 'rb') as f:
            config.load(f, 'utf-8')
        return config
