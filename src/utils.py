import json


def write_file(file_name):

    data = {'file_name': file_name}

    files = {
        'json': ('json_data', json.dumps(data), 'application/json'),
        'model': (file_name, open(file_name, "rb"), 'application/octet-stream')}

    return files
