
class BaseLines(object):
	def __init__(self, tc_freq, t_freq):
		# get probability of context given target <likelihood>
		self.row_sums = tc_freq.astype(float).sum(axis=1)
		self.p_c_given_t = (tc_freq.astype(float)+1) / (self.row_sums[:, np.newaxis] + 1 + corpus_size)
		# get probability of target <prior>
		self.p_t = t_freq.astype(float) / sum(t_freq)
		
	## Type0: prior probability :: same as returning p_t
	def prior(self):
		return self.p_t

	## Type1: consider only observed in the context
	def nb_pred_t1(self, context_ids):
		target_prob = np.zeros((self.p_t.shape))
		for target_id in range(corpus_size):
			# skip if the target has not seen in training set
			if self.row_sums[target_id] == 0:
				continue
			log_sum_xj = 0
			for context_id in context_ids:
				# if p_c_given_t[target_id, context_id] == 0:
				# 	continue
				log_sum_xj += math.log(self.p_c_given_t[target_id, context_id])
			# old way::error when p_t[target_id] == 0
			# target_prob[target_id] = math.exp(log_sum_xj + math.log(p_t[target_id]))
			# new way:: decompose using exp(a+b) = exp(a) . exp(b)
			target_prob[target_id] = math.exp(log_sum_xj) * self.p_t[target_id]
		target_prob = target_prob / sum(target_prob)
		return target_prob

	## Type3: randomly pick any
	def random_pred(self):
		pred = np.random.rand(corpus_size,1)[:,0]
		pred = pred / sum(pred)
		return pred
