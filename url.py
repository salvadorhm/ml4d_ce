import urllib.request
import json

try:
    data = {'colum_0':'value0', 'column_1':'value_1'}

    body = str.encode(json.dumps(data))

    url = 'http://www.ml4d.com/api'
    headers = {'Content-Type':'application/json'}

    req = urllib.request.Request(url, body, headers)
    response = urllib.request.urlopen(req)
    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))

