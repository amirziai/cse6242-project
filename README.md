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
sudo apt-get install python3-pip python-dev nginx supervisor3 gunicorn3 -y
sudo rm /etc/nginx/sites-enabled/default
sudo vim /etc/nginx/sites-available/cse6242.com

server {
	listen 80;

	location / {
		proxy_pass http://127.0.0.1:8000/;
	}
}

sudo ln -s /etc/nginx/sites-available/cse6242.com /etc/nginx/sites-enabled/cse6242.com

sudo service nginx restart

sudo apt-get install python3-venv -y

git clone https://github.com/amirziai/cse6242-project.git
git checkout origin/interactive
pip3 install -r requirements.txt
gunicorn3 app:app
sudo mkdir /var/log/cse6242

gunicorn3 app:app --daemon

write to /etc/supervisor/conf.d/cse6242.conf

[program:cse6242]
directory=/home/ubuntu/cse6242-project
command=gunicorn3 app:app
autostart=true
autorestart=true
stderr_logfile=/var/log/cse6242/cse6242.err.log
stdout_logfile=/var/log/cse6242/cse6242.out.log

sudo supervisorctl reread
sudo service supervisor restart
sudo supervisorctl status

sudo nginx -t
sudo service nginx restart
```
