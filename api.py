from pickle import GET
import flask
from flask import request, jsonify
from test2 import cameraStart, diceEyes, get_blobs, get_dice_from_blobs


app = flask.Flask(__name__)
app.config["DEBUG"] = False

# Create some test data for our catalog in the form of a list of dictionaries.
#books = [
   # {'id': 0,
   #  'title': 'A Fire Upon the Deep',
   #  'author': 'Vernor Vinge',
   #  'first_sentence': 'The coldsleep itself was dreamless.',
   #  'year_published': '1992'},rfaew
   # {'id': 1,
   #  'title': 'The Ones Who Walk Away From Omelas',
   #  'author': 'Ursula K. Le Guin',
   #  'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
   #  'published': '1973'},
    #{'id': 2,
     #'title': 'Dhalgren',
   #  'author': 'Samuel R. Delany',
  #   'first_sentence': 'to wound the autumnal city.',
 #    'published': '1975'}
#]


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Archive</h1>
    <p>A prototype API for distant reading of science fiction novels.</p>'''


# A route to return all of the available entries in our catalog.
@app.route('/api/v1/resources/books/all', methods=['GET'])
def api_all():
        return jsonify(books)

@app.route('/cameradice', methods=['GET'])
def api_lol():
       
        cameraStart()
        return('xD')
        
@app.route('/deletedate', methods=['GET'])
def api_xD():
    
        diceEyes.clear()
        return('xDD')
        

@app.route('/diceresult', methods =['GET'])
def api_diceresults():
    print(diceEyes)
    diceresult = diceEyes[len(diceEyes)-2] + diceEyes[len(diceEyes)-1]
    return jsonify(diceresult)

if __name__ == "__main__":
        app.run(host = '0.0.0.0', port=5000)