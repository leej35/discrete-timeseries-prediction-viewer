# -*- coding: utf-8 -*-
# Discrete Time-Series Prediction Viewer
# Nov 18. by Jeongmin Lee

import json
import numpy as np
import sys,os,json
import urlparse
import math
from optparse import OptionParser
from BaseHTTPServer import HTTPServer,BaseHTTPRequestHandler
from models.BaseLines import BaseLines

port = int(os.environ.get('PORT', 33507))

# load necessaties
labels = list(np.load('models/labels.npy'))
embedding_in = np.load('models/embedding_in_48h_event.npy')
embedding_out = np.load('models/embedding_out_48h_event.npy')
corpus_size = embedding_in.shape[0]
tc_freq = np.load('models/baseline_sampling_type__tr_smpl_typeevent_baseline_hrx48_target_hr_gap_0_corpus_size_868_nepch_1_n_tr_sample80000.pkl__target_context_freq.npy')
t_freq = np.load('models/baseline_sampling_type__tr_smpl_typeevent_baseline_hrx48_target_hr_gap_0_corpus_size_868_nepch_1_n_tr_sample80000.pkl__target_freq.npy')
bl = BaseLines(tc_freq, t_freq)


def cbow_pred(itemids, mode):
    embed_in_vec = embedding_in[itemids,:]
    embed_in_vec = embed_in_vec.mean(0)
    embed_out_vec = embedding_out[itemids,:]
    embed_out_vec = embed_out_vec.mean(0)
    if mode == 'in_out':
        embed_rank = embedding_out.dot(embed_in_vec)
    elif mode == 'in_in':
        embed_rank = embedding_in.dot(embed_in_vec)
    elif mode == 'out_out':
        embed_rank = embedding_out.dot(embed_out_vec)
    print 'embed_rank'
    print embed_rank[:10]
    # ps = np.exp(embed_rank)
    # embed_rank = ps / sum(ps)
    rankings = sorted(enumerate(embed_rank), key=lambda k:k[1], reverse=True)
    result = []
    i=1
    for r in rankings:
        result.append({"Rank":i,"Event":labels[r[0]], "Score":str(r[1])})
        i+=1
    return result


def nb_pred(itemids):
    embed_rank = bl.nb_pred_t1(itemids)
    rankings = sorted(enumerate(embed_rank), key=lambda k:k[1], reverse=True)
    result = []
    i=1
    for r in rankings:
        result.append({"Rank":i,"Event":labels[r[0]], "Score":str(r[1])})
        i+=1
    return result


class http_server:
    def __init__(self, port):
        def handler(*args):
            CBOWHandler(*args)
        server = HTTPServer(('', port), handler)
        print 'Starting server at port {}, use <Ctrl-C> to stop'.format(port)
        server.serve_forever()

class CBOWHandler(BaseHTTPRequestHandler):
    def __init__(self,*args):
        BaseHTTPRequestHandler.__init__(self, *args)
    def jsonResponse(self,content):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(content))
    def do_GET(self):
        try:
            if self.path.startswith('/predict'):
                parsed_path = urlparse.urlparse(self.path)
                item_ids = urlparse.parse_qs(urlparse.urlparse(self.path).query).get('item_ids', None)
                model_type = urlparse.parse_qs(urlparse.urlparse(self.path).query).get('model_type', None)
                print item_ids
                print model_type
                print model_type[0]
                print model_type[0] == 'cbow'
                item_ids = [int(item) for item in item_ids]
                print item_ids
                item_ranking = [(None,None)]
                if model_type[0] == 'cbow_in_out':
                    item_ranking = cbow_pred(item_ids, 'in_out')
                elif model_type[0] == 'cbow_in_in':
                    item_ranking = cbow_pred(item_ids, 'in_in')
                elif model_type[0] == 'cbow_out_out':
                    item_ranking = cbow_pred(item_ids, 'out_out')
                elif model_type[0] == 'naivebayes':
                    item_ranking = nb_pred(item_ids)
                """
                item_ranking: [ (item_id, item_score), .. ]
                """
                print item_ranking[:2]
                self.jsonResponse(item_ranking)
                return
            elif self.path.endswith("models/medvec_index_dic_med_lab_ablab_proc.json"):
                f = open("models/medvec_index_dic_med_lab_ablab_proc.json") #self.path has /test.html
                #note that this potentially makes every file on your computer readable by the internet
                self.send_response(200)
                self.send_header('Content-type',    'text/html')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
                return
            elif self.path == '/':
                f = open('index.html')
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
                return
        except IOError:
          self.send_error(404, 'file not found')



def printUsage():
    print 'Usage: index.py -m <modelfile> [-b]'

def main(argv):
    parser = OptionParser()
    parser.add_option("-p", "--port",
                      type="int", dest="port", default=8080,
                      help="server port")
    (options, args) = parser.parse_args()

    from BaseHTTPServer import HTTPServer
    server = http_server(port)

if __name__ == "__main__":
    main(sys.argv[1:])
