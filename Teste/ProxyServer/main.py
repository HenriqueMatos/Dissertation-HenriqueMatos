# flask run -h localhost -p 3000
from flask import Flask, redirect, url_for, request

app = Flask(__name__)

@app.route('/')
def main():
   return "Hello Guys"
    

@app.route('/user',methods = ['POST'])
def user():
   print(request.json)
   return "redirect(url_for('success',name = user))"

@app.route('/vhost',methods = ['POST'])
def vhost():
   print(request.json)
   return "redirect(url_for('success',name = user))"

@app.route('/resource',methods = ['POST'])
def resource():
   print(request.json)
   return "redirect(url_for('success',name = user))"

@app.route('/topic',methods = ['POST'])
def topic():
   print(request.json)
   return "redirect(url_for('success',name = user))"

if __name__ == '__main__':
   app.run(debug = True)


# @app.route('/vhost', methods=['POST'])
# def vhost(args):

# @app.route('/resource', methods=['POST'])
# def resource(args):

# @app.route('/topic', methods=['POST'])
# def topic(args):
