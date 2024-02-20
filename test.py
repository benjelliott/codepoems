from gensim.test.utils import common_texts
import scipy.stats as stats
multi = stats.multinomial(1, [1/3, 1/3, 1/3])
print(common_texts)