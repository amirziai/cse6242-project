[![Build Status](https://travis-ci.org/amirziai/flatten.svg?branch=master)](https://travis-ci.org/amirziai/flatten)

# cse6242-project
Georgia Tech CSE6242 Team 14 Project, interactive document clustering

### Sample
Get started with [sample1.ipynb](sample1.ipynb)

### TODO
- create a test module
- create a module for each function 

### Deployment
```bash
sudo apt-get update
sudo apt-get install python-pip python-dev nginx -y

pip install -r requirements.txt
pip install gunicorn
gunicorn -w 4 -b 127.0.0.1:80 app:app
```
