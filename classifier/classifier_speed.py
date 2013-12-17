from simple_db import DB_connection
from collections import defaultdict, Counter
from matplotlib.pyplot import hist, scatter, show
import re
from matplotlib.colors import LogNorm
from matplotlib.pylab import matshow, colorbar
from pylab import *
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.sparse import csr_matrix
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.preprocessing import scale
from datetime import datetime
import random
from bs4 import BeautifulSoup
import csv
import sys
import argparse


# globals
SW = stopwords.words('english')
STATES = dict([[re.sub(r'"','',t).lower() for t in re.split(r',',s)] for s in re.split(r'\n', open('states.csv','rb').read())[1:-1]])



# >> BINARY CLASSIFICATION OVER PAIRS OF STATUTES

# NOTE NOTE --- TO-DO LIST :: 11/26/2013:
#
# 1. clean up sub-sampling -> make into standardized testing framework
# 2. experiment with more types of features, hyper-parameters
# 3. load in abuse statutes as well
# 4. test cross-sub-domain
# 5. [download some of the bulk.resources corpus & begin experimenting with?]


# tokenize utility function
def tokenize(string, ps=None, stem=True):
  if ps is None:
    ps = PorterStemmer()
  s = lambda t : ps.stem(t) if stem else t
  return [s(t) for t in re.findall(r'[a-z]+',string.lower()) if len(t) > 2 and t not in SW]


# load word embeddings
WE_SOURCE = 'stat_vecs.txt'
def word_embeddings(we_source=WE_SOURCE, we_size=200):
  wes = defaultdict(lambda : np.zeros(we_size))
  with open(we_source, 'rb') as f:
    for line in f:
      split = re.split(r' ', line)
      wes[split[0]] = np.array([float(s) for s in split[1:-1]])
  return wes


# load annotations + annotated statutes from anno_stats_dto.txt
class StatuteInfo:
  def __init__(self, sid=None, label=None):
    self.id = sid
    self.label = label
    self.state = None
    self.title = None
    self.path_titles = None
    self.text = None

  def add_data(self, state, title, text, pid, ps=None):
    if ps is None:
      ps = PorterStemmer()
    self.state = state
    self.title = title
    self.title_toks = tokenize(title, ps=ps)
    self.title_toks_unstemmed = tokenize(title, stem=False)
    self.text = text
    self.text_toks= tokenize(text, ps=ps)
    self.text_toks_unstemmed = tokenize(text, stem=False)
    self.title_toks_in_text = list(set(self.title_toks).intersection(self.text_toks))
    self.pid = int(pid) if pid is not None else pid
     

SOURCE_FILE = 'anno_stats_dto.txt'
def get_annotated(source_file=SOURCE_FILE):
  ps = PorterStemmer()
  tfidf = TfidfVectorizer(stop_words=SW, ngram_range=(1,1), sublinear_tf=True)
  anno = []
  by_id = {}

  # basic data from file + db, also basic tokenized/stemmed forms
  with DB_connection() as handle:
    with open(SOURCE_FILE, 'rb') as f:
      for i,line in enumerate(f.readlines()):
        a = re.split(r'\s*\t\s*', line)
        if i > 0 and len(a) > 1:
          s = StatuteInfo(int(a[0]), a[1].strip())
          handle[1].execute("SELECT state,header,text,parent_component_id FROM annotate_statutecomponent WHERE id=%s", (s.id,))
          s.add_data(*handle[1].fetchone(), ps=ps)
          anno.append(s)
          by_id[s.id] = s

  # tfidf text vectors
  X=csr_matrix(tfidf.fit_transform([' '.join(s.title_toks + s.text_toks) for s in anno])).todense()
  for i,x in enumerate(X):
    anno[i].tfidf_vec = np.reshape(np.array(x), x.shape[1])

  # make sure all parent components are loaded
  with DB_connection() as handle:

    # recursive subfunction, returns list of titles, parent -> child, loading from db if needed
    def get_parent_titles(pid):
      
      # base case
      if pid is None:
        return []

      # parent already loaded from db
      elif by_id.has_key(pid):
        s = by_id[pid]
        return get_parent_titles(s.pid) + [s.title]
      
      # need to load parent from db
      else:
        s = StatuteInfo(pid, None)
        handle[1].execute("SELECT state,header,text,parent_component_id FROM annotate_statutecomponent WHERE id=%s", (pid,))
        s.add_data(*handle[1].fetchone(), ps=ps)
        by_id[s.id] = s
        return get_parent_titles(s.pid) + [s.title]

    # run recursive sub function
    for s in anno:
      s.path_titles = get_parent_titles(s.pid) + [s.title]
      s.path_title_toks = tokenize(' '.join(s.path_titles), ps=ps)
      s.path_title_toks_unstemmed = tokenize(' '.join(s.path_titles), stem=False)
      s.path_title_toks_in_text = list(set(s.path_title_toks).intersection(s.text_toks))

  return anno


# subfunction for code format permutations
def code_format_perms(code):
  DIVS = [' ', '-', '.', '/', '_']
  nums = re.findall(r'[0-9a-z]+', code, flags=re.I)
  return [d.join(nums) for d in DIVS]

# get speeding law statutes
# NOTE >> just use this + manual to get static csv list, then work from that...
SPEEDING_SOURCE = 'speed_laws.html'
def get_speeding_ids(source_file=SPEEDING_SOURCE):
  stats = []
  
  # get speeding law statute codes from html file
  with open(source_file, 'rb') as f:
    soup = BeautifulSoup(f.read(), 'lxml')
  state_codes = [(tr.find_all('td')[0].text, [re.sub(r'<[^>]+>','',s).strip() for s in re.split(r',|\t|\n', tr.find_all('td')[4].text) if len(s.strip())>0]) for tr in soup.find('table').find_all('tr')[1:]]

  # >> straight hard-coding of some...
  state_codes += [('michigan', ['mcl-257-'+str(n) for n in range(627,634)])]
  #state_codes += [('louisiana', 'mcl-257-'+str(n)) for n in range(627,634)]
  state_codes += [('georgia', ['40-6-'+str(n) for n in range(180,190)])]
  
  # get all statutes
  found_count = 0
  not_found = []
  with DB_connection() as handle:
    for state,codes in state_codes:

      # check for state abbreviation
      if STATES.has_key(state.strip().lower()):
        state_brev = STATES[state.strip().lower()]

        # >> Nebraska & DE format correction...
        if state_brev == 'ne':
          codes = ['60-6-213', '60-6-186']
        elif state_brev == 'de':
          codes = ['21-41-08', '21-41-09']
        
        # check that we have statutes for the state
        handle[1].execute("SELECT id FROM annotate_statutecomponent WHERE state LIKE '"+state_brev+"%' LIMIT 1")
        d = handle[1].fetchone()
        if d and len(d) > 0:
          for code in codes:

            # >> handle ranges
            # NOTE: To-do; hard-coded in at top for now...

            # >> for some states we need to go to the statute subcomponent table...
            # NOTE: To-do
            if state_brev in ['tx','or','ok','nd','sc']:
              not_found.append((state, code))
              continue

            # >> hard-coded for a few states...
            # - NOTE Can't find for: KS, IA, LA
            elif state_brev in ['vt','pa']:
              parts = re.findall(r'[0-9a-zA-Z]+', code)
              where_clause = "index_official LIKE '%title-"+parts[0]+"%' AND index_official LIKE '%/"+parts[1]+"%'"
            elif state_brev in ['ma']:
              parts = re.findall(r'[0-9a-zA-Z]+', code)
              where_clause = "index_official LIKE '%CHAPTER"+parts[1]+"/Section"+parts[2]+"%'"
            elif state_brev in ['me']:
              parts = re.findall(r'[0-9a-zA-Z]+', code)
              where_clause = "index_official LIKE '%title"+parts[0]+"%' AND index_official LIKE '%sec"+parts[1]+"%'"
            elif state_brev in ['ky']:
              parts = re.findall(r'[0-9a-zA-Z]+', code)
              where_clause = "index_official LIKE '%"+parts[0]+"-00/"+parts[1]+"%'"
            elif state_brev == 'de':
              parts = re.findall(r'[0-9a-zA-Z]+', code)
              where_clause = "index_official LIKE '%title"+parts[0]+"/c0"+parts[1]+"/c0"+parts[1]+"-sc"+parts[2]+"%'"
            
            # check for the particular code referenced, searching over several permutations
            # of format written
            else:
              where_clause = "(" + " OR ".join(["index_official LIKE '%"+c+"%'" for c in code_format_perms(code.strip())]) + ")"
            
            # try to fetch & store from db
            handle[1].execute("SELECT id FROM annotate_statutecomponent WHERE state LIKE '"+state_brev+"%' AND " + where_clause)
            sid = handle[1].fetchall()
            if sid and len(sid) > 0:
              found_count += 1
              stats += [r[0] for r in sid]
              
              # >> Screen manually for errors resulting in too many hits
              if len(sid) > 1:
                print "%s %s --> %s" % (state_brev, code, len(sid))
            else:
              not_found.append((state, code))
      #else:
      #  not_found.append((state,None))
  
  print 'Found %s percent of available:'%(100.0*(found_count/float(len(not_found)+found_count)),)
  print 'Found %s sids' % (len(stats),)
  
  # export to csv file & return
  with open('speeding_stat_ids.csv', 'wb') as f:
    writer = csv.writer(f)
    for s in stats:
      writer.writerow([int(s)])
  return not_found
    

# NOTE NOTE >> TO-DO:
# - navigate up tree hierarchies
# - etc. etc.


# simple class to check for duplicates via python dict
# NOTE: could upgrade to Bloom filter if need to do this at scale...
class Duplicate:
  def __init__(self):
    self.d = {}

  def check(self, string):
    if self.d.has_key(string):
      return True
    else:
      self.d[string] = len(self.d)
      return False


# get speeding statutes + all siblings of parent-nodes <levels-up>
# >> returns max <n_other> randomly sampled non-speeding sibling statutes as well
SOURCE_FILE_SPEEDING = 'speeding_stat_ids.csv'
def get_speeding_stats(levels_up=1, n_other=2000, source_file=SOURCE_FILE_SPEEDING):
  ps = PorterStemmer()
  tfidf = TfidfVectorizer(stop_words=SW, ngram_range=(1,1), sublinear_tf=True)
  stats = []
  by_id = {}
  duplicate = Duplicate()
  
  # basic data from file + db, also basic tokenized/stemmed forms
  with DB_connection() as handle:
    with open(source_file, 'rb') as f:
      for i,line in enumerate(f.readlines()):
        s = StatuteInfo(int(line.strip()), 'speed')
        handle[1].execute("SELECT state,header,text,parent_component_id FROM annotate_statutecomponent WHERE id=%s", (s.id,))
        s.add_data(*handle[1].fetchone(), ps=ps)

        # >> append to list only if not duplicate...
        if not duplicate.check(s.title + ' ' + s.text):
          stats.append(s)
          by_id[s.id] = s
    
    print 'Found %s speeding statutes' % (len(stats),)
    sys.stdout.flush()

    # get all siblings that are children of parent nodes <levels-up> up or less
    if levels_up > 0:

      # get all parent node ids at <levels_up> above starting level
      pids = [s.pid for s in stats if s.pid]
      for i in range(levels_up-1):
        handle[1].execute("SELECT parent_component_id FROM annotate_statutecomponent WHERE id IN ("+",".join([str(p) for p in pids])+")")
        pids += [r[0] for r in handle[1].fetchall() if r[0]]

      # get all children of these parent nodes --> ids only for now!
      cids = []
      pids = list(set(pids))
      while len(pids) > 0:
        print len(pids)
        sys.stdout.flush()
        handle[1].execute("SELECT id FROM annotate_statutecomponent WHERE parent_component_id IN ("+",".join([str(p) for p in pids])+")")
        cids_new = list(set([r[0] for r in handle[1].fetchall()]))
        cids += cids_new
        pids = cids_new

      # >> sub-sample randomly up to max_other statutes
      cids = random.sample(cids, n_other)
      for cid in cids:
        s = StatuteInfo(cid, 'not-speed')
        handle[1].execute("SELECT state,header,text,parent_component_id FROM annotate_statutecomponent WHERE id=%s", (s.id,))
        s.add_data(*handle[1].fetchone(), ps=ps)

        # >> append to list only if not duplicate...
        if not duplicate.check(s.title + ' ' + s.text):
          stats.append(s)
          by_id[s.id] = s

  # tfidf text vectors
  X=csr_matrix(tfidf.fit_transform([' '.join(s.title_toks + s.text_toks) for s in stats])).todense()
  for i,x in enumerate(X):
    stats[i].tfidf_vec = np.reshape(np.array(x), x.shape[1])

  print 'Loaded %s statutes' % (len(stats),)
  sys.stdout.flush()
  return stats


# get idf weightings
class IndexRow:
  def __init__(self):
    self.w = 0
    self.pos = []

def get_inv_index(stats):
  D = float(len(stats))
  inv_index = defaultdict(lambda : IndexRow())
  for s in stats:
    for t in s.title_toks_unstemmed + s.text_toks_unstemmed:
      inv_index[t].pos.append(s.id)
  for t,idx_row in inv_index.iteritems():
    inv_index[t].w = np.log(D/len(set(idx_row.pos)))
  return inv_index


# generate set of pair vectors using combination of different features to generate
FEATURES = [1,0,0,0,0]
def statute_pairs(stats, wes, inv_index, fs=FEATURES, as_pair=False):
  X = []
  Y = []

  # >> only match speeding with either speeding/non (no non-non pairs)
  for i,s1 in enumerate(stats):
    if s1.label == 'speed':
      for j,s2 in enumerate(stats):
        if s2.label == 'non-speed' or j >= i:
          if as_pair:
            f1 = np.array([])
            f2 = np.array([])
          else:
            f = np.array([])
          
          # >> F0 -- difference between tf-idf scaled one-hot vecs (leaf node title+text)
          if fs[0] > 0.0:
            if as_pair:
              f1 = np.append(f1, s1.tfidf_vec)
              f2 = np.append(f2, s2.tfidf_vec)
            else:
              f = np.append(f, np.abs(s1.tfidf_vec - s2.tfidf_vec))
          
          # >> F1 -- elementwise difference between word embedding vecs (leaf node title+text)
          if fs[1] > 0.0:
            s1_we = np.sum([inv_index[t].w*wes[t] for t in s1.text_toks_unstemmed+s1.title_toks_unstemmed], 0)
            s2_we = np.sum([inv_index[t].w*wes[t] for t in s2.text_toks_unstemmed+s2.title_toks_unstemmed], 0)
            if as_pair:
              f1 = np.append(f1, s1_we)
              f2 = np.append(f2, s2_we)
            else:
              f = np.append(f, np.abs(s2_we - s1_we))

          # >> F2 -- euclidean difference between word embedding vecs (leaf node text)
          if fs[2] > 0.0:
            s1_we = np.sum([wes[t] for t in s1.text_toks_unstemmed], 0)
            s2_we = np.sum([wes[t] for t in s2.text_toks_unstemmed], 0)
            if as_pair:
              f1 = np.append(f1, s1_we)
              f2 = np.append(f2, s2_we)
            else:
              f = np.append(f, euclidean(s2_we,s1_we))

          # >> F3 -- euclidean difference between word embedding vecs (leaf node title)
          if fs[3] > 0.0:
            s1t_we = np.sum([wes[t] for t in s1.title_toks_unstemmed], 0)
            s2t_we = np.sum([wes[t] for t in s2.title_toks_unstemmed], 0)
            if as_pair:
              f1 = np.append(f1, s1t_we)
              f2 = np.append(f2, s2t_we)
            else:
              f = np.append(f, euclidean(s2t_we,s1t_we))

          # >> F4 -- euclidean difference between word embedding vecs (path titles)
          if fs[4] > 0.0:
            s1p_we = np.sum([wes[t] for t in s1.path_title_toks_unstemmed], 0)
            s2p_we = np.sum([wes[t] for t in s2.path_title_toks_unstemmed], 0)
            if as_pair:
              f1 = np.append(f1, s1p_we)
              f2 = np.append(f2, s2p_we)
            else:
              f = np.append(f, euclidean(s2p_we,s1p_we))

          # append to X, also append binary aligned / not aligned label
          if as_pair:
            X.append([f1,f2])
          else:
            X.append(f)
          Y.append(1.0 if s1.label == s2.label else 0.0)

  return np.array(X), np.array(Y)


# utility function to write training / testing vector set to disk
def write_vec_pairs_out(X,Y,file_name):
  print X.shape

  # print to file + return
  with open(file_name, 'wb') as f:
    for i,pair in enumerate(zip(X,Y)):
      if i%10000 == 0:
        print i
      x,y = pair
      f.write('\t'.join([str(xx) for xx in x[0]])+'\n')
      f.write('\t'.join([str(xx) for xx in x[1]])+'\n')
      f.write(str(y)+'\n')
      f.write('\n')
  

# train simple binary classifier, performing grid search
N_FOLDS = 4
UM = 1
N_CORES = 1
def train_classifier(X,Y):
  now = datetime.now()
  print "Training classifier at %s:%s" % (now.hour, now.minute)

  # scale all X vectors
  max_x = np.max(X)
  X = [x/max_x for x in X]

  # create training / testing folds with undersampling of negatives so as to re-balance
  # >> only undersample for trainingm, *NOT* for testing (test on actual distr.)
  
  # split into random, class-proportional folds
  classes = (1.0, 0.0)
  yis = [[i for i,y in enumerate(Y) if y == c] for c in classes]
  for i,c in enumerate(classes):
    random.shuffle(yis[i])
  ycs = [int(float(len(c))/N_FOLDS) for c in yis]

  # take N_FOLDS-1 folds as training, subsample majority class for training, (not for testing)
  skf = []
  for i in range(N_FOLDS):
    test_i = np.concatenate([yic[i*ycs[j]:(i+1)*ycs[j]] for j,yic in enumerate(yis)])
    train_i = yis[0][:i*ycs[0]] + yis[0][(i+1)*ycs[0]:]
    train_i += random.sample(yis[1][:i*ycs[1]] + yis[1][(i+1)*ycs[1]:], int(UM*len(train_i)))
    skf.append((train_i, test_i))

  # manual parameter grid / cross-validation search + F1-scoring
  #grid_scores = []
  #C_range = 2.0 ** np.arange(-3,10)
  #for c in C_range:

  scores = []
  for train_i, test_i in skf:
    #clf = LinearSVC(C=c, loss='l2', penalty='l2', tol=1e-3)
    clf = LinearSVC(loss='l2', penalty='l2', tol=1e-3)
    clf.fit([X[i] for i in train_i], [Y[i] for i in train_i])
    Y_est = clf.predict([X[i] for i in test_i])
    scores.append(f1_score([Y[i] for i in test_i], Y_est))
  now = datetime.now()
  print ">> (F1) score: %s   at %s:%s" % (np.mean(scores), now.hour, now.minute)

  #grid_scores.append((c, np.mean(scores)))
  #grid_scores.sort(key=lambda x : -x[1])
  # print score of best parameters
  #now = datetime.now()
  #print "Best (F1) score: %s   at %s:%s" % (grid_scores[0][1], now.hour, now.minute)

  # train & return best classifier on all data and return
  #clf = LinearSVC(C=grid_scores[0][0], loss='l2', penalty='l2', tol=1e-3)
  clf = LinearSVC(loss='l2', penalty='l2', tol=1e-3)
  clf.fit(X, Y)
  return clf


def print_stat(s):
  print s.title
  print s.state
  print s.label
  print '\n\n'
  print s.text

def run_all():
  print 'Loading word embeddings...'
  wes = word_embeddings()
  print 'Loading & processing annotated statutes...'
  stats = get_speeding_stats()
  print 'Computing inverted index...'
  inv_index = get_inv_index(stats)
  print 'Computing alignment pair feature vectors...'
  pairs = statute_pairs(stats, wes, inv_index)
  print 'Training & testing classifier...'
  clf = train_classifier(*pairs)

def write_vecs():
  print 'Loading word embeddings...'
  wes = word_embeddings()
  print 'Loading & processing annotated statutes...'
  stats = get_speeding_stats()
  print 'Computing inverted index...'
  inv_index = get_inv_index(stats)

  # >> only take an approx. proportionate amount of non-speed stats (for now)
  stats_pos = [s for s in stats if s.label=='speed']
  stats_neg = random.sample([s for s in stats if s.label=='not-speed'], len(stats_pos))
  stats = stats_pos + stats_neg

  print 'Computing alignment pair feature vectors...'
  X,Y = statute_pairs(stats, wes, inv_index, as_pair=True)
  print 'Writing to disk...'
  write_vec_pairs_out(X,Y,'speed_pair_vecs.all.lab.model1')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-r", "--run", default=False, action="store_true")
  parser.add_argument("-w", "--write", default=False, action="store_true")
  args = parser.parse_args()
  if args.run:
    run_all()
  elif args.write:
    write_vecs()
