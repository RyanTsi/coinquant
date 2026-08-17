from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=['configs/config.json'],
)