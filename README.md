# Local Latent Representation based on Geometric Convolution for Particle Data Feature Exploration

Latent Representations are generated for particle data using a Geometric Convolution based autoencoder. Latent vectors are used for feature exploration through hierarchical clustering and tracking through mean-shift. 

## Dependencies

Insatall all dependencies:
```
pip install -r requirements.txt
```

## Model Training

```
python main.py -d 'fpm' --ball --result-dir result_example
```
* Find the example trained model and data at example/
* Example model analysis code at "vis.ipynb"
* Get 2016 SciVis Contest data (FPM) at: https://www.uni-kl.de/sciviscontest/
* Get 2016 SciVis Contest data (cosmology) at: https://darksky.slac.stanford.edu/scivis2015/

## VAST System

Configure the file "vis/src/server.py"

Setting flask app path:
```
cd /path/to/project/root/
$Env:FLASK_APP='./vis/src/server.py'
flask run
```
Visit http://127.0.0.1:5000/ for the system.


## Tracking

Configure the file "mean_shift.py"

```
python mean_shift.py
```

## Radius estimation

Configure the file "h_search.py"

```
python h_search.py
```