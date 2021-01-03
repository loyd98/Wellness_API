VirtualEnv
====
Installation
----
```console
$ virtualenv API
$ source API/bin/activate
(API)$ pip install -r path/to/requirements.txt
```

CoreNLP
====

Download:http://nlp.stanford.edu/software/stanford-corenlp-latest.zip

Download: http://nlp.stanford.edu/software/stanford-corenlp-4.2.0-models-english.jar

Installation
----
```console
$ mv /path/to/stanford-corenlp-4.2.0-models-english.jar /path/to/stanford-corenlp-4.2.0
$ export CLASSPATH=$CLASSPATH:/path/to/stanford-corenlp-4.2.0/*:
```
Running the API
====
Run the CoreNLP Server first:
```console
$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```
Then run the python file using the virtual environment:
```
$ python api.py
```
