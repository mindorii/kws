import flask

app = flask.Flask(__name__)

import cStringIO as StringIO
import os
import sys

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

import models

THRESHOLD = 20.0
kws = models.KWS(
        save_path="/home/awni/models/speech/kws/",
        keyword="olivia",
        window_size=700,
        step_size=200)

@app.route('/', methods=['POST'])
def index():
    request = flask.request

    content_type = request.content_type
    assert content_type.split(";")[0] == "audio/wav", \
            "Bad file type."

    fh = StringIO.StringIO(request.get_data())
    scores = kws.evaluate_wave(fh)
    print(scores)
    if min(scores) < THRESHOLD:
        text = "yes"
        confidence = 1
    else:
        text = "no"
        confidence = 1

    alternatives = [ {'transcript' : text,
                      'confidence' : confidence} ]
    result_dict = {'result_index' : 0,
                   'result' :
                   [ {'alternative' : alternatives,
                      'final' : True} ] }

    return flask.json.htmlsafe_dumps(result_dict)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2375, debug=True)
