# Discrete Time-Series Prediction Viewer
A web based viewer for discrete event prediction in time-series.

Demo: <https://intense-waters-97573.herokuapp.com/>

## How to run on Heroku

After download files and put them into `deploy_web` folder. And inside the folder:
```
git init
heroku create --buildpack https://github.com/thenovices/heroku-buildpack-scipy
echo -e "numpy==1.9.2\nscipy==0.15.1" > requirements.txt
git add requirements.txt
git commit -m 'Added requirements'
git push heroku master
```
