''' helpers to load configuration without dealing with paths in notebook '''

from pathlib import Path, PurePath

yaml_lookup = {
    'BlueFors': 'BFXLD1.yaml',
    '4K_magnet': 'oxford_4K_magnet_station.yaml',
}

def get_station_config_path(system_name):
    ''' load config path by system name '''

    import shockley
    if system_name in yaml_lookup:
        module_path = Path(shockley.__file__).parent
        config_path = module_path / Path(
                'station_configurations/{0:s}'.format(yaml_lookup[system_name])
            )
        return str(config_path)
    else:
        raise ValueError(f'{system_name} is not a recognized system name.')
