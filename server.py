
from gevent import monkey
monkey.patch_all()
import model 
from flask import Flask, request, render_template,jsonify

from gevent.pywsgi import WSGIServer 

app = Flask(__name__,template_folder="templates")


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
def results():
    try:
        smiles = request.args.get('smiles')
        mode = request.args.get('mode') 
        cell_line = request.args.get('cell_line') 
        result_df = model.get_results(smiles, mode, cell_line)
        return render_template('output.html', tables=result_df.to_html())
    except:
        return render_template('output.html', tables='The chemical you\'ve entered cannot be synthesized try again')
      

