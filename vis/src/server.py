from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS, cross_origin
import json
import os,sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)
from utils.simple import show
from utils.process_data import data_reader, numpy_to_vtp, uniform_sample
try:
    data_path = os.environ['data']
except KeyError:
    data_path = './data'


app = Flask(__name__, static_folder='../build')
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# running parameters
data_type = 'fpm'
latent_file = 'example/latent.npy'
data_file = 'example/025.vtu'
data_file = None
sample_type = 'random' # 'even'
sample_size = 2000
perplexity = 30
num_process = 16

if data_file is None:
    if data_type =='jet3b':
        data_file = "jet3b/run3g_50Am_jet3b_sph.3400"
    elif data_type == 'fpm':
        data_file = "2016_scivis_fpm/0.44/run41/025.vtu"
    elif data_type == 'cos':
        data_file = 'ds14_scivis_0128/raw/ds14_scivis_0128_e4_dt04_0.4900'
    elif data_type == 'fpm_h':
        data_type = 'fpm'
        data_file = '2016_scivis_fpm/0.20/run03/025.vtu'
    data = data_reader(os.path.join(data_path,data_file),data_type)
else:
    data = data_reader(data_file,data_type)


# init the cluster
clu = {
    'children':[],
    'idx': list(range(len(data)))
}

# load latent vectors
latent = np.load(latent_file)
print('Making sure the right file is loaded')
print(latent.shape)
print(data.shape)
assert len(data) == len(latent)

# load clusters
if sample_type =='even':
    choice = uniform_sample(sample_size,data[:,3:])
else:
    choice = np.random.choice(len(data),sample_size,replace=False) 
choice = np.array(choice).tolist()

# calculate tsne projection
ts = TSNE(2,perplexity=perplexity,n_jobs=num_process)
tsne = ts.fit_transform(latent[choice]).tolist()

@app.route("/get_root")
@cross_origin()
def get_root():
    ret = {
        'clu': clu,
        'tsne': tsne,
        'choice': choice,
    }
    return jsonify(ret)

@app.route("/get_children")
@cross_origin()
def get_children():
    values = request.args
    trace = values.getlist('trace[]')
    num_clu = int(values.get('num_clu'))
    cur = clu
    for t in trace:
        cur = cur['children'][int(t)]
    if len(cur['children']) == 0:
        km = KMeans(num_clu,n_init=3,n_jobs=num_process)
        idx = np.array(cur['idx'])
        clu_array = km.fit_predict(latent[idx])
        for i in range(num_clu): 
            new_node = {
                'children': [],
                'idx': idx[clu_array==i].tolist(),
            }
            cur['children'].append(new_node)
    return jsonify(cur['children'])


@app.route("/rm_children")
@cross_origin()
def rm_children():
    values = request.args
    trace = values.getlist('trace[]')
    cur = clu
    for t in trace:
        cur = cur['children'][int(t)]
    cur['children'] = []
    return jsonify(1)

@app.route("/vtk")
@cross_origin()
def vtk():
    values = request.args
    trace = values.getlist('trace[]')
    cur = clu
    for t in trace:
        cur = cur['children'][int(t)]
    idx = cur['idx']
    coord = data[idx,:3]
    if data_type == 'fpm':
        array_dict = {
            "concentration":data[idx,3],
            "velocity":data[idx,4:],
        }
    elif data_type == 'cos':
        array_dict = {
            "phi":data[idx,-1],
            "velocity":data[idx,3:6],
            "acceleration":data[idx,6:9],
        }
    elif data_type == 'jet3b':
        array_dict = {
            'rho': data[idx,3],
            'temp': data[idx,4],
        }
    vtk_data = numpy_to_vtp(coord,array_dict)
    show(vtk_data,time='',data_type=data_type,show=True)

    return jsonify(0)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
@cross_origin()
def serve(path):
    print(path)
    # chdir only apply to os package but not flask
    if path != "" and os.path.exists("./vis/build/" + path):
        return send_from_directory('../build/', path)
    else:
        return send_from_directory('../build/', 'index.html')
